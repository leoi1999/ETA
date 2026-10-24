# Reproduction guide

## Environment and installation

The supplied reference environment is Python 3.8.20, PyTorch 2.4.1, and CUDA 12.1. Run the Bash launchers on Linux or WSL with a compatible NVIDIA driver and GPU.

```bash
conda create -n eta python=3.8.20 -y
conda activate eta
python -m pip install --upgrade pip
python -m pip install -r requirements-cu121.txt
```

`requirements-cu121.txt` selects the PyTorch CUDA wheel index and PyTorch Scatter wheel source. `requirements.txt` lists the direct runtime dependencies. Training and feature extraction use one GPU by default; select another physical GPU through `GPU=N`.

## Prepare data and models

```bash
python tools/download_data.py
python tools/download_data.py --check
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='google-bert/bert-base-uncased', local_dir='pretrained_models/bert')"
```

Feature encoders load model files locally. The BERT directory must contain its configuration, weights, vocabulary, and tokenizer files. Dataset details are in [DATASETS.md](DATASETS.md).

The supplied visual features are used by default. Only to regenerate them, download [CLIP ViT-B/16](https://huggingface.co/openai/clip-vit-base-patch16/tree/main) into `pretrained_models/clip-vit-base-patch16/` and prepare the entity images under each dataset's `images_random_split_one_seed42/` directory. Consult `python encoder/encode_visual.py --help` for image manifests, stability settings, and output options. Original MMKB VGG features are not replacements for these stability-aware CLIP features.

## Generate attribute features

`run_main.sh` creates missing attribute features automatically. Before using `run_pseudo.sh` independently, generate them once for each required dataset:

```bash
for dataset in FB15K_DB15K FB15K_YAGO15K; do
  CUDA_VISIBLE_DEVICES=0 python encoder/encode_attribute.py \
    --dataset-dir "data/$dataset" \
    --attribute-file "data/$dataset/attr" \
    --model-dir pretrained_models/bert \
    --output-dir "data/$dataset/bert_attr_from_mmkb_unified" \
    --output-name attr_name_value_bert_cls_embeddings.pkl \
    --device cuda:0 --batch-size 128 --max-length 128 \
    --text-content name-value --pooling cls --format-mode unified --fp16
done
```

Keep just one dataset name to prepare only that dataset. Generated features can be reused across training ratios.

## Non-iterative experiments

```bash
bash scripts/run_main.sh FB_DB 0.2
bash scripts/run_main.sh FB_YAGO 0.5
GPU=1 bash scripts/run_main.sh FB_DB 0.8
```

Each invocation trains graph, attribute, and visual stages, then trains decision fusion using their checkpoints. The dataset argument is `FB_DB` or `FB_YAGO`; supported ratios are `0.2`, `0.5`, and `0.8`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPU` | `0` | Physical GPU index |
| `PYTHON_BIN` | `python` | Python interpreter |
| `DATA_ROOT` | `data` | Dataset root |
| `BERT_DIR` | `pretrained_models/bert` | Complete local BERT model |
| `CLIP_DIR` | `pretrained_models/clip-vit-base-patch16` | Optional local CLIP model |
| `LOG_ROOT` | `logs/main_normvotes` | Output root |
| `BASE_EPOCHS` | `70` | Maximum epochs for each single-modality stage |
| `FUSION_EPOCHS` | `30` | Maximum fusion epochs |
| `EARLY_STOP_PATIENCE` | `5` | Evaluations without improvement before early stopping |

For example:

```bash
GPU=0 BERT_DIR=/models/bert DATA_ROOT=/datasets/eta LOG_ROOT=/runs/eta \
  bash scripts/run_main.sh FB_DB 0.2
```

Existing features are reused. Set `FORCE_REENCODE_ATTRIBUTES=1` or `FORCE_REENCODE_VISUALS=1` to regenerate them after preparing the required inputs. Final fusion metrics are in the `multimodal-fusion-*` stage's `train.log`; stage directories also contain `config.json` and checkpoints.

## Pseudo-label experiments

After preparing attribute and visual features:

```bash
bash scripts/run_pseudo.sh FB_DB 0.2
GPU=1 bash scripts/run_pseudo.sh FB_YAGO 0.8
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `RUN_ROOT` | Timestamped directory under `logs/` | Must be a new directory |
| `GRAPH_EPOCHS` | `30` | Maximum graph epochs |
| `EPOCHS` | `30` | Maximum attribute and visual epochs |
| `FUSION_EPOCHS` | `30` | Maximum fusion epochs |
| `EARLY_STOP_PATIENCE` | `20` | Early-stopping patience |
| `PSEUDO_THRESHOLD` | `0.70` | Pseudo-pair score threshold |
| `MAX_PSEUDO_PAIRS` | `25` | Maximum newly accepted pairs per round |
| `SPLIT_SEED` | `43` | Train/validation/test split seed |

`GPU`, `PYTHON_BIN`, and `DATA_ROOT` work as above. Graph and fusion stages can add pseudo-labels; attribute and visual stages consume the existing graph-derived pool. A pseudo-label audit runs after training.

## Evaluation protocols

The two launchers use distinct protocols implemented in `src/dataset/loader.py` and `src/dataset/splits.py`:

- **Non-iterative:** `alignment_split=auto` resolves to `legacy`; the provided `seed<ratio>/sup_ent_ids` are training pairs and `dev_ent_ids` are the evaluation pairs. This path does not create an independent held-out test partition.
- **Pseudo-label:** `--alignment-split dmea --split-seed 43` shuffles the full `ref_ent_ids` pairs with Python's seeded RNG. With `N` pairs and ratio `r`, train size is `floor(N*r)`, validation size is `floor((N-train)/2)`, and the rest is test. Test endpoints are excluded from the validation-derived pseudo-label candidate pool. Checkpoints are selected using validation results; the fusion stage requests final test evaluation.

Graph resources remain transductive. The training seed specified by both launchers is `1029`; the independent pseudo-label split seed defaults to `43`. Reproduce both the launch command and the evaluation protocol when reporting comparisons. Hits@1, Hits@3, Hits@10, MR, and MRR are logged by the evaluator.

## Validation scope for this release

Release preparation checks Python syntax, Bash syntax, dry runs for all twelve dataset/ratio/launcher combinations, data checksums, graph/entity consistency, and alignment-split consistency. GitHub Actions repeats syntax and dry-run checks without downloading the datasets or installing GPU dependencies.

These checks do not reproduce full GPU training or independently verify the manuscript's numerical results. No trained ETA checkpoint or experiment log was supplied in the release archive.

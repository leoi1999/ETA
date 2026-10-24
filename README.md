# ETA: Select Within, Fuse Across

**Reliable Decision Fusion for Multi-Modal Entity Alignment**

[简体中文](README.zh-CN.md) · [Framework](assets/framework.pdf) · [Datasets](docs/DATASETS.md) · [Reproduction guide](docs/REPRODUCIBILITY.md) · [Citation](CITATION.bib)

ETA identifies entities that refer to the same real-world object across multi-modal knowledge graphs. It follows a **select-within, fuse-across** design: select discriminative information within individual modalities, then combine their alignment decisions using weights adapted to each candidate entity pair.

![Overview of ETA: graph information selection, visual patch reweighting, attribute encoding, and dynamic decision fusion.](assets/framework.png)

## Method

- **Graph information selection:** dimension-wise message filtering, relation-aware neighbor weighting, and prior-regularized adaptive propagation-layer weighting improve structural representations.
- **Visual information selection:** Gaussian low-pass filtering measures local patch stability; stability-based patch reweighting emphasizes informative visual cues.
- **Attribute encoding:** BERT encodes textualized attribute names and values.
- **Dynamic decision fusion:** an MLP gating network produces pair-specific modality weights to combine graph, attribute, and visual alignment scores.

This repository accompanies *Select Within, Fuse Across: Reliable Decision Fusion for Multi-Modal Entity Alignment*. The supplied manuscript reports experiments on five benchmarks. **This release contains the code and prepared data for FB15K-DB15K and FB15K-YAGO15K**, with seed ratios of 20%, 50%, and 80%, in non-iterative and pseudo-label settings. The DBP15K ZH-EN, JA-EN, and FR-EN experiments, baseline implementations, ablations, and case studies are outside this release.

## Quick start

Use Linux or WSL with Bash and a CUDA-capable NVIDIA GPU. The reference environment is Python 3.8.20, PyTorch 2.4.1, and CUDA 12.1.

```bash
git clone https://github.com/leoi1999/ETA.git
cd ETA
conda create -n eta python=3.8.20 -y
conda activate eta
python -m pip install --upgrade pip
python -m pip install -r requirements-cu121.txt

# Download the prepared datasets and verify their SHA-256 checksums.
python tools/download_data.py
```

Download the complete [BERT base uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/main) model into `pretrained_models/bert/`, including weights, configuration, and tokenizer files. For example, after installing the runtime dependencies:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='google-bert/bert-base-uncased', local_dir='pretrained_models/bert')"

# Train graph, attribute, visual, and fusion stages in sequence.
bash scripts/run_main.sh FB_DB 0.2
```

The main script generates missing attribute features and reuses the included visual features. It does not need raw images or CLIP weights for this default path. Logs, configurations, and checkpoints are written under `logs/main_normvotes/`.

## Datasets and downloads

| Benchmark | Command-line key | Knowledge graphs | Seed ratios |
| --- | --- | --- | --- |
| FB15K-DB15K | `FB_DB` | Freebase and DBpedia | `0.2`, `0.5`, `0.8` |
| FB15K-YAGO15K | `FB_YAGO` | Freebase and YAGO | `0.2`, `0.5`, `0.8` |

**[Download the prepared ETA data](https://github.com/leoi1999/ETA/releases/download/data-v1/eta-data-v1.zip)**, or run `python tools/download_data.py`. The archive contains graph triples, entity mappings, raw attributes, alignment splits, and the supplied 512-dimensional visual features. Extract it in the repository root to create `data/FB15K_DB15K/` and `data/FB15K_YAGO15K/`.

For the original benchmark resources, see [MMKG/MMKB](https://github.com/mniepert/mmkb), its [archival data record](https://doi.org/10.5281/zenodo.1245698), and [MMEA](https://github.com/liyichen-cly/MMEA). For the bilingual DBP15K benchmarks discussed in the manuscript, see the [MEAformer data instructions](https://github.com/zjukg/MEAformer#-dataset) and [EVA data instructions](https://github.com/cambridgeltl/eva#data). Their data formats and visual representations are not interchangeable with ETA's prepared files without conversion.

See [dataset documentation](docs/DATASETS.md) for measured dataset statistics, file formats, checksums, upstream attribution, and optional raw-image preparation. Raw images, attribute embeddings, pretrained weights, and trained ETA checkpoints are not bundled.

## Run experiments

Run commands from the repository root. Each invocation selects one dataset and one seed ratio.

```bash
# Non-iterative experiments
bash scripts/run_main.sh FB_DB 0.2
bash scripts/run_main.sh FB_YAGO 0.5
GPU=1 bash scripts/run_main.sh FB_DB 0.8

# Inspect commands without installing PyTorch or starting training.
bash scripts/run_main.sh FB_DB 0.2 --dry-run
bash scripts/run_pseudo.sh FB_YAGO 0.2 --dry-run
```

For pseudo-label experiments, first generate the attribute features as described in the [reproduction guide](docs/REPRODUCIBILITY.md), or run the main script once for that dataset. Then:

```bash
bash scripts/run_pseudo.sh FB_DB 0.2
GPU=1 bash scripts/run_pseudo.sh FB_YAGO 0.8
```

**Evaluation protocols differ:** the non-iterative script uses the provided `sup_ent_ids`/`dev_ent_ids` splits. Pseudo-label runs deterministically re-split reference pairs into train/validation/test partitions, select checkpoints on validation data, and evaluate the final fusion model on held-out test data. See [protocol details](docs/REPRODUCIBILITY.md#evaluation-protocols) before comparing scores.

## Repository structure

```text
ETA/
|-- assets/                       # Framework PNG and original PDF
|-- data/                         # Downloaded datasets; not stored in Git
|-- docs/                         # Data and reproduction documentation
|-- encoder/
|   |-- encode_attribute.py        # BERT attribute feature extraction
|   |-- encode_visual.py           # Stability-aware CLIP feature extraction
|   `-- stability_clip.py
|-- scripts/
|   |-- run_main.sh                # Non-iterative experiment pipeline
|   `-- run_pseudo.sh              # Pseudo-label experiment pipeline
|-- src/
|   |-- dataset/                  # Loading, splits, and pseudo-label state
|   |-- model/                    # Graph, attribute, visual, and fusion modules
|   |-- runner/                   # Training and evaluation
|   |-- config.py
|   `-- run.py
|-- tools/download_data.py         # Data download, extraction, and verification
|-- requirements.txt
`-- requirements-cu121.txt
```

## Citation and acknowledgments

Repository citation metadata is provided in [CITATION.bib](CITATION.bib). A published-paper citation can be added when its bibliographic details are available.

We thank the maintainers of [MMKG/MMKB](https://github.com/mniepert/mmkb), [MMEA](https://github.com/liyichen-cly/MMEA), [EVA](https://github.com/cambridgeltl/eva), and [MEAformer](https://github.com/zjukg/MEAformer) for their benchmark resources and public research repositories. See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for data attribution and the repositories consulted for documentation structure. Questions and reproducibility reports can be submitted through [GitHub Issues](https://github.com/leoi1999/ETA/issues).

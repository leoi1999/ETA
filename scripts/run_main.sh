#!/usr/bin/env bash
set -Eeuo pipefail

# Run one dataset/ratio through four sequential stages on one GPU.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    echo "Usage: bash scripts/run_main.sh DATASET RATIO [--dry-run]"
    echo "  DATASET: FB_DB or FB_YAGO; RATIO: 0.2, 0.5 or 0.8"
    echo "  GPU defaults to 0; set GPU=N to select another single GPU."
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if [[ $# -lt 2 || $# -gt 3 ]]; then usage >&2; exit 2; fi
DATASET="$1"
RATIO="$2"
case "$DATASET" in FB_DB|FB_YAGO) ;; *) usage >&2; exit 2;; esac
case "$RATIO" in 0.2|0.5|0.8) ;; *) usage >&2; exit 2;; esac
DRY_RUN=0
if [[ $# == 3 ]]; then
    [[ "$3" == --dry-run ]] || { usage >&2; exit 2; }
    DRY_RUN=1
fi
GPU="${GPU:-0}"
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "GPU must be one non-negative device index" >&2; exit 2; }

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/main_normvotes}"
BASE_EPOCHS="${BASE_EPOCHS:-70}"
FUSION_EPOCHS="${FUSION_EPOCHS:-30}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
BERT_DIR="${BERT_DIR:-${REPO_ROOT}/pretrained_models/bert}"
ATTRIBUTE_BATCH_SIZE="${ATTRIBUTE_BATCH_SIZE:-128}"
ATTRIBUTE_MAX_LENGTH="${ATTRIBUTE_MAX_LENGTH:-128}"
CLIP_DIR="${CLIP_DIR:-${REPO_ROOT}/pretrained_models/clip-vit-base-patch16}"
VISUAL_BATCH_SIZE="${VISUAL_BATCH_SIZE:-256}"
FORCE_REENCODE_VISUALS="${FORCE_REENCODE_VISUALS:-0}"
FORCE_REENCODE_ATTRIBUTES="${FORCE_REENCODE_ATTRIBUTES:-0}"

ATTRIBUTE_RELATIVE_PATH="${ATTRIBUTE_RELATIVE_PATH:-bert_attr_from_mmkb_unified/attr_name_value_bert_cls_embeddings.pkl}"
FB_DB_VISUAL_RELATIVE_PATH="${FB_DB_VISUAL_RELATIVE_PATH:-image/image.pkl}"
FB_YAGO_VISUAL_RELATIVE_PATH="${FB_YAGO_VISUAL_RELATIVE_PATH:-image/image.pkl}"
ATTRIBUTE_RELATIVE_DIR="$(dirname -- "${ATTRIBUTE_RELATIVE_PATH}")"
ATTRIBUTE_OUTPUT_NAME="$(basename -- "${ATTRIBUTE_RELATIVE_PATH}")"

require_file() {
    local path="$1"
    [[ "$DRY_RUN" == 1 ]] && return 0
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: required file does not exist: ${path}" >&2
        exit 1
    fi
}

require_dir() {
    local path="$1"
    [[ "$DRY_RUN" == 1 ]] && return 0
    if [[ ! -d "${path}" ]]; then
        echo "ERROR: required directory does not exist: ${path}" >&2
        exit 1
    fi
}

for source_file in \
    src/__init__.py \
    src/config.py \
    src/run.py \
    encoder/encode_attribute.py \
    encoder/encode_visual.py \
    src/dataset/loader.py \
    src/model/alignment_model.py \
    src/model/graph/graph_encoder.py \
    src/model/graph/loss.py \
    src/model/attribute/module.py \
    src/model/visual/module.py \
    src/model/decision_fusion/strategy.py; do
    require_file "${source_file}"
done

if [[ "$DATASET" == FB_DB ]]; then
    dataset_dir=FB15K_DB15K
    dataset_tag=fb-db
    visual_relative_path="$FB_DB_VISUAL_RELATIVE_PATH"
    top_k=30
else
    dataset_dir=FB15K_YAGO15K
    dataset_tag=fb-yago
    visual_relative_path="$FB_YAGO_VISUAL_RELATIVE_PATH"
    top_k=20
fi
dataset_path="$DATA_ROOT/$dataset_dir"
for name in ent_ids_1 ent_ids_2 triples_1 triples_2; do
    require_file "$dataset_path/$name"
done
require_file "$dataset_path/seed$RATIO/sup_ent_ids"
require_file "$dataset_path/seed$RATIO/dev_ent_ids"

run_command() {
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU"
        printf ' %q' "$@"
        printf '\n'
    else
        CUDA_VISIBLE_DEVICES="$GPU" "$@"
    fi
}

encode_visual_dataset() {
    local gpu="$1" dataset_path="$2" relative_path="$3" top_k="$4"
    local output_path="${dataset_path}/${relative_path}"
    if [[ "${FORCE_REENCODE_VISUALS}" != "1" && -s "${output_path}" ]]; then
        echo "[GPU ${gpu}] Reusing normalized-vote visual embeddings: ${output_path}"
        return 0
    fi
    require_file "${CLIP_DIR}/config.json"
    require_dir "${dataset_path}/images_random_split_one_seed42"
    [[ "$DRY_RUN" == 1 ]] || mkdir -p "$(dirname -- "${output_path}")"
    run_command "${PYTHON_BIN}" -u encoder/encode_visual.py \
        --dataset-dir "${dataset_path}" \
        --images-dir "${dataset_path}/images_random_split_one_seed42" \
        --clip-dir "${CLIP_DIR}" \
        --output-dir "$(dirname -- "${output_path}")" \
        --output-name "$(basename -- "${output_path}")" \
        --device cuda:0 --batch-size "${VISUAL_BATCH_SIZE}" \
        --images-per-entity 1 --random-seed 42 --pooling mean --fp16 \
        --stability-layers 1,5,8 --stability-top-k "${top_k}" \
        --stability-epsilon 1e-6
    require_file "${output_path}"
}

prepare_dataset_features() {
    local gpu="$1" dataset_path="$2" relative_path="$3" top_k="$4"
    encode_attribute_dataset "${gpu}" "${dataset_path}"
    encode_visual_dataset "${gpu}" "${dataset_path}" "${relative_path}" "${top_k}"
}

encode_attribute_dataset() {
    local gpu="$1"
    local dataset_path="$2"
    local output_dir="${dataset_path}/${ATTRIBUTE_RELATIVE_DIR}"
    local output_path="${dataset_path}/${ATTRIBUTE_RELATIVE_PATH}"

    if [[ "${FORCE_REENCODE_ATTRIBUTES}" != "1" && -s "${output_path}" ]]; then
        echo "[GPU ${gpu}] Reusing attribute embeddings: ${output_path}"
        return 0
    fi

    require_file "${BERT_DIR}/config.json"
    require_file "${dataset_path}/attr"
    [[ "$DRY_RUN" == 1 ]] || mkdir -p "${output_dir}"
    echo "[GPU ${gpu}] Encoding unified name-value attributes for ${dataset_path}"
    run_command "${PYTHON_BIN}" encoder/encode_attribute.py \
        --dataset-dir "${dataset_path}" \
        --attribute-file "${dataset_path}/attr" \
        --model-dir "${BERT_DIR}" \
        --output-dir "${output_dir}" \
        --output-name "${ATTRIBUTE_OUTPUT_NAME}" \
        --device cuda:0 \
        --batch-size "${ATTRIBUTE_BATCH_SIZE}" \
        --max-length "${ATTRIBUTE_MAX_LENGTH}" \
        --text-content name-value \
        --pooling cls \
        --format-mode unified \
        --fp16
}

run_experiment() {
    local gpu="$1"
    local dataset_key="$2"
    local dataset_tag="$3"
    local ratio="$RATIO"
    local ratio_tag graph_log_name graph_log_dir attribute_log_name attribute_log_dir
    local visual_log_name visual_log_dir fusion_log_name visual_relative_path
    local graph_reg attribute_dropout attribute_lr visual_lr fusion_balance

    if [[ "${dataset_key}" == "FB_DB" ]]; then
        # Confirmed FB-DB fine-search winner.
        visual_relative_path="${FB_DB_VISUAL_RELATIVE_PATH}"
        graph_reg="5"
        attribute_dropout="0.0"
        attribute_lr="0.001"
        visual_lr="0.0003"
        fusion_balance="0.02"
    else
        visual_relative_path="${FB_YAGO_VISUAL_RELATIVE_PATH}"
        # The broad r02 search is the available YAGO result.
        graph_reg="30"
        attribute_dropout="0.1"
        attribute_lr="0.001"
        visual_lr="0.001"
        fusion_balance="0.0"
    fi

        ratio_tag="${ratio/./}"
        graph_log_name="multimodal-graph-${dataset_tag}-r${ratio_tag}-${RUN_TAG}"
        graph_log_dir="${LOG_ROOT}/${graph_log_name}"
        attribute_log_name="multimodal-attribute-${dataset_tag}-r${ratio_tag}-${RUN_TAG}"
        attribute_log_dir="${LOG_ROOT}/${attribute_log_name}"
        visual_log_name="multimodal-visual-${dataset_tag}-r${ratio_tag}-${RUN_TAG}"
        visual_log_dir="${LOG_ROOT}/${visual_log_name}"
        fusion_log_name="multimodal-fusion-${dataset_tag}-r${ratio_tag}-${RUN_TAG}"

        echo "[GPU ${gpu}] Stage 1/4: training graph encoder for ${dataset_key}, ratio=${ratio}"
        run_command "${PYTHON_BIN}" -m src.run \
            "${dataset_key}" "${ratio}" \
            --data-root "${DATA_ROOT}" \
            --device cuda:0 \
            --modules graph \
            --lr 0.005 \
            --depth 3 \
            --node-dim 128 \
            --relation-dim 128 \
            --message-filter-scale 2.0 \
            --layer-weight-mode learned \
            --layer-prior-type folded_normal \
            --layer-regularization-weight "${graph_reg}" \
            --decision-fusion-loss-weight 0 \
            --semi-prediction false \
            --epochs "${BASE_EPOCHS}" \
            --early-stop-patience "${EARLY_STOP_PATIENCE}" \
            --batch-size 512 \
            --seed 1029 \
            --log-root "${LOG_ROOT}" \
            --log-name "${graph_log_name}"

        require_file "${graph_log_dir}/best_graph_model.pt"

        echo "[GPU ${gpu}] Stage 2/4: training attribute encoder for ${dataset_key}, ratio=${ratio}"
        run_command "${PYTHON_BIN}" -m src.run \
            "${dataset_key}" "${ratio}" \
            --data-root "${DATA_ROOT}" \
            --device cuda:0 \
            --modules attribute \
            --attribute-embedding-path "${ATTRIBUTE_RELATIVE_PATH}" \
            --attribute-use-projector true \
            --attribute-projection-dim 256 \
            --attribute-dropout "${attribute_dropout}" \
            --attribute-temperature 0.07 \
            --lr "${attribute_lr}" \
            --decision-fusion-loss-weight 0 \
            --semi-prediction false \
            --epochs "${BASE_EPOCHS}" \
            --early-stop-patience "${EARLY_STOP_PATIENCE}" \
            --batch-size 512 \
            --seed 1029 \
            --log-root "${LOG_ROOT}" \
            --log-name "${attribute_log_name}"

        require_file "${attribute_log_dir}/best_attribute_model.pt"

        echo "[GPU ${gpu}] Stage 3/4: training visual encoder for ${dataset_key}, ratio=${ratio}"
        run_command "${PYTHON_BIN}" -m src.run \
            "${dataset_key}" "${ratio}" \
            --data-root "${DATA_ROOT}" \
            --device cuda:0 \
            --modules visual \
            --visual-embedding-path "${visual_relative_path}" \
            --visual-use-projector true \
            --visual-projection-dim 256 \
            --visual-dropout 0.1 \
            --visual-temperature 0.07 \
            --lr "${visual_lr}" \
            --decision-fusion-loss-weight 0 \
            --semi-prediction false \
            --epochs "${BASE_EPOCHS}" \
            --early-stop-patience "${EARLY_STOP_PATIENCE}" \
            --batch-size 512 \
            --seed 1029 \
            --log-root "${LOG_ROOT}" \
            --log-name "${visual_log_name}"

        require_file "${visual_log_dir}/best_visual_model.pt"

        echo "[GPU ${gpu}] Stage 4/4: training frozen-encoder fusion for ${dataset_key}, ratio=${ratio}"
        run_command "${PYTHON_BIN}" -m src.run \
            "${dataset_key}" "${ratio}" \
            --data-root "${DATA_ROOT}" \
            --device cuda:0 \
            --modules graph,attribute,visual \
            --attribute-embedding-path "${ATTRIBUTE_RELATIVE_PATH}" \
            --attribute-use-projector true \
            --attribute-dropout "${attribute_dropout}" \
            --visual-embedding-path "${visual_relative_path}" \
            --visual-use-projector true \
            --visual-projection-dim 256 \
            --visual-dropout 0.1 \
            --visual-temperature 0.07 \
            --load-graph-model "${graph_log_dir}/best_graph_model.pt" \
            --load-attribute-model "${attribute_log_dir}/best_attribute_model.pt" \
            --load-visual-model "${visual_log_dir}/best_visual_model.pt" \
            --train-decision-fusion-only true \
            --decision-fusion-hidden-dim 32 \
            --decision-fusion-temperature 1.0 \
            --decision-fusion-initial-weights 0.60 0.25 0.15 \
            --decision-score-temperature 0.05 \
            --decision-fusion-balance-weight "${fusion_balance}" \
            --decision-fusion-loss-weight 1.0 \
            --decision-fusion-eval-interval 1 \
            --lr 0.0001 \
            --depth 3 \
            --node-dim 128 \
            --relation-dim 128 \
            --message-filter-scale 2.0 \
            --layer-weight-mode learned \
            --layer-prior-type folded_normal \
            --layer-regularization-weight "${graph_reg}" \
            --semi-prediction false \
            --epochs "${FUSION_EPOCHS}" \
            --early-stop-patience "${EARLY_STOP_PATIENCE}" \
            --batch-size 512 \
            --seed 1029 \
            --log-root "${LOG_ROOT}" \
            --log-name "${fusion_log_name}"
}

echo "Running $DATASET ratio=$RATIO on GPU $GPU"
prepare_dataset_features "$GPU" "$dataset_path" "$visual_relative_path" "$top_k"
run_experiment "$GPU" "$DATASET" "$dataset_tag"
echo "Completed $DATASET ratio=$RATIO. Outputs: $LOG_ROOT"

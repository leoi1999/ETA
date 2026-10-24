#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
usage() {
    echo "Usage: bash scripts/run_pseudo.sh DATASET RATIO [--dry-run]"
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
RUN_ROOT="${RUN_ROOT:-$ROOT/logs/pseudo_${DATASET}_r${RATIO}_$(date +%Y%m%d_%H%M%S)_$$}"
EPOCHS="${EPOCHS:-30}"
GRAPH_EPOCHS="${GRAPH_EPOCHS:-30}"
FUSION_EPOCHS="${FUSION_EPOCHS:-30}"
PATIENCE="${EARLY_STOP_PATIENCE:-20}"
PSEUDO_THRESHOLD="${PSEUDO_THRESHOLD:-0.70}"
MAX_PSEUDO_PAIRS="${MAX_PSEUDO_PAIRS:-25}"
ATTR='bert_attr_from_mmkb_unified/attr_name_value_bert_cls_embeddings.pkl'
require_file() {
    [[ "$DRY_RUN" == 1 ]] && return 0
    [[ -f "$1" ]] || { echo "Missing required input: $1" >&2; exit 2; }
}
if [[ "$DATASET" == FB_DB ]]; then
    DATASET_DIR=FB15K_DB15K
    GRAPH_REG=5
    ATTRIBUTE_DROPOUT=0.0
    VISUAL_LR=0.0003
else
    DATASET_DIR=FB15K_YAGO15K
    GRAPH_REG=30
    ATTRIBUTE_DROPOUT=0.1
    VISUAL_LR=0.001
fi
VISUAL_PATH="image/image.pkl"
for name in ent_ids_1 ent_ids_2 triples_1 triples_2 "$ATTR" "$VISUAL_PATH"; do
    require_file "$DATA_ROOT/$DATASET_DIR/$name"
done
for name in sup_ent_ids dev_ent_ids ref_ent_ids; do
    require_file "$DATA_ROOT/$DATASET_DIR/seed$RATIO/$name"
done
if [[ "$DRY_RUN" != 1 ]]; then
    [[ ! -e "$RUN_ROOT" ]] || { echo "RUN_ROOT already exists; choose a fresh directory: $RUN_ROOT" >&2; exit 2; }
    mkdir -p "$RUN_ROOT"
fi

run_stage() { local dataset="$1" ratio="$2" modules="$3" name="$4" pool="$5"; shift 5
  # Only graph and fusion add labels. Attribute/visual consume the existing
  # graph-derived pool while retaining the same explicit three-way split.
  local semi=false epochs="$EPOCHS" final_test=false
  [[ "$modules" == graph || "$modules" == graph,attribute,visual ]] && semi=true
  [[ "$modules" == graph,attribute,visual ]] && final_test=true
  [[ "$modules" == graph ]] && epochs="$GRAPH_EPOCHS"
  [[ "$modules" == graph,attribute,visual ]] && epochs="$FUSION_EPOCHS"
  local -a cmd=("$PYTHON_BIN" -m src.run "$dataset" "$ratio" --data-root "$DATA_ROOT" --device cuda:0 --modules "$modules" --semi-prediction "$semi" --alignment-split dmea --split-seed "${SPLIT_SEED:-43}" --final-test "$final_test" --semi-start-epoch "${SEMI_START_EPOCH:-5}" --pseudo-pool-path "$pool" --threshold "$PSEUDO_THRESHOLD" --threshold-weight 1.0 --max-pseudo-pairs-per-round "$MAX_PSEUDO_PAIRS" --conflict-policy 2 --epochs "$epochs" --eval-interval 1 --early-stop-patience "$PATIENCE" --batch-size 512 --seed 1029 --log-root "$RUN_ROOT" --log-name "$name" "$@")
  if [[ "${DRY_RUN:-0}" == 1 ]]; then printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU"; printf ' %q' "${cmd[@]}"; printf '\n'; else CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}"; fi
}
run_one() { local dataset="$1" ratio="$2" visual="$3" reg="$4" adrop="$5" alr="$6" vlr="$7"; local base="${dataset,,}_r${ratio//./p}"; local pool="$RUN_ROOT/${base}_shared_pseudo_pool.pt"; local attr="$ATTR";
  run_stage "$dataset" "$ratio" graph "${base}-graph" "$pool" --lr 0.005 --depth 3 --node-dim 128 --relation-dim 128 --message-filter-scale 2 --layer-weight-mode learned --layer-prior-type folded_normal --layer-regularization-weight "$reg"
  run_stage "$dataset" "$ratio" attribute "${base}-attribute" "$pool" --attribute-embedding-path "$attr" --attribute-use-projector true --attribute-projection-dim 256 --attribute-dropout "$adrop" --attribute-temperature 0.07 --lr "$alr"
  run_stage "$dataset" "$ratio" visual "${base}-visual" "$pool" --visual-embedding-path "$visual" --visual-use-projector true --visual-projection-dim 256 --visual-dropout 0.1 --visual-temperature 0.07 --lr "$vlr"
  run_stage "$dataset" "$ratio" graph,attribute,visual "${base}-fusion" "$pool" --attribute-embedding-path "$attr" --visual-embedding-path "$visual" --load-graph-model "$RUN_ROOT/${base}-graph/best_graph_model.pt" --load-attribute-model "$RUN_ROOT/${base}-attribute/best_attribute_model.pt" --load-visual-model "$RUN_ROOT/${base}-visual/best_visual_model.pt" --train-decision-fusion-only true --decision-fusion-hidden-dim 32 --decision-fusion-temperature 1 --decision-fusion-initial-weights 0.60 0.25 0.15 --decision-score-temperature 0.05 --decision-fusion-balance-weight 0.02 --decision-fusion-loss-weight 1 --decision-fusion-eval-interval 1 --lr 0.0001 --depth 3 --node-dim 128 --relation-dim 128 --message-filter-scale 2 --layer-weight-mode learned --layer-prior-type folded_normal --layer-regularization-weight "$reg"
}
echo "Running pseudo-label experiment: $DATASET ratio=$RATIO on GPU $GPU"
run_one "$DATASET" "$RATIO" "$VISUAL_PATH" "$GRAPH_REG" "$ATTRIBUTE_DROPOUT" 0.001 "$VISUAL_LR"
if [[ "$DRY_RUN" != 1 ]]; then
    "$PYTHON_BIN" -m src.utils.pseudo_audit "$RUN_ROOT"
fi
echo "Completed $DATASET ratio=$RATIO. Outputs: $RUN_ROOT"

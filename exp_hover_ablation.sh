#!/bin/bash
# Hover vs no-hover, single-round, fixed_alpha, hotdog @ 80k/160k/320k splats.
# Single round = --rounds 1 (full budget in round 1, no growth) -- direct comparison
# of LMGModel (gs_type=lmg) vs LMGModelHover (gs_type=lmg_hover) at matched budget.
#
#   bash exp_hover_ablation.sh <gs_type: lmg|lmg_hover> <gpu>
#
# Idempotent: a config whose round_summary.json already has 1 round entry is skipped.

set -u
GS_TYPE="${1:?usage: exp_hover_ablation.sh <lmg|lmg_hover> <gpu>}"
export CUDA_VISIBLE_DEVICES="${2:?usage: exp_hover_ablation.sh <lmg|lmg_hover> <gpu>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/env.local.sh" ] && source "$SCRIPT_DIR/env.local.sh"
: "${DATASET_BASE_DIR:?Set DATASET_BASE_DIR in env.local.sh}"
: "${MESH_BASE_DIR:?Set MESH_BASE_DIR in env.local.sh}"

ITERS=32000
DATASET_DIR="$DATASET_BASE_DIR/hotdog"
MESH_FILE="$MESH_BASE_DIR/hotdog/hotdog.ply"
MESH_IMG_DIR="$MESH_BASE_DIR/hotdog"
PY() { conda run -n lmg python "$@"; }

for TOTAL_SPLATS in 80000 160000 320000; do
    tag="hotdog_${TOTAL_SPLATS}_${GS_TYPE}"
    OUT="output/0709_hover/$tag"
    LOG="log/0709_hover/${tag}.log"
    mkdir -p "$OUT" "$(dirname "$LOG")"

    done_rounds=0
    [ -f "$OUT/round_summary.json" ] && \
        done_rounds="$(PY -c "import json; print(len(json.load(open('$OUT/round_summary.json'))))" 2>/dev/null || echo 0)"
    if [ "$done_rounds" = "1" ]; then
        echo "[skip] $tag (round_summary.json already has 1 round)"
        continue
    fi

    echo "[run] $tag total_splats=$TOTAL_SPLATS gs_type=$GS_TYPE"
    PY train_progressive_orchestrator.py --eval -s "$DATASET_DIR" -m "$OUT" \
        --texture_obj_path "$MESH_FILE" --mesh_type milo --gs_type "$GS_TYPE" \
        --alloc_policy distortion_progressive --precaptured_mesh_img_path "$MESH_IMG_DIR" \
        --occlusion --mesh_rasterizer_type nvdiffrast --fixed_alpha \
        --rounds 1 --total_splats "$TOTAL_SPLATS" --iters_per_round "$ITERS" \
        --schedule linear --seed 0 > "$LOG" 2>&1 \
        || { echo "[FAIL] $tag (see $LOG)"; continue; }
    echo "[done] $tag"
done

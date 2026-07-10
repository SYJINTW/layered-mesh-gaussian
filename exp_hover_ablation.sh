#!/bin/bash
# Hover vs no-hover, single-round, fixed_alpha comparison.
# Single round = --rounds 1 (full budget in round 1, no growth) -- direct comparison
# of LMGModel (gs_type=lmg) vs LMGModelHover (gs_type=lmg_hover) at matched budget.
#
#   bash exp_hover_ablation.sh <gs_type: lmg|lmg_hover> <gpu> <scene: hotdog|hotdog_colmap|bicycle>
#
# Idempotent: a config whose round_summary.json already has 1 round entry is skipped.

set -u
GS_TYPE="${1:?usage: exp_hover_ablation.sh <lmg|lmg_hover> <gpu> <hotdog|hotdog_colmap|bicycle>}"
export CUDA_VISIBLE_DEVICES="${2:?usage: exp_hover_ablation.sh <lmg|lmg_hover> <gpu> <hotdog|hotdog_colmap|bicycle>}"
SCENE="${3:?usage: exp_hover_ablation.sh <lmg|lmg_hover> <gpu> <hotdog|hotdog_colmap|bicycle>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/env.local.sh" ] && source "$SCRIPT_DIR/env.local.sh"
: "${DATASET_BASE_DIR:?Set DATASET_BASE_DIR in env.local.sh}"
: "${MESH_BASE_DIR:?Set MESH_BASE_DIR in env.local.sh}"

ITERS=32000
IMAGES=""
case "$SCENE" in
    hotdog)
        DATASET_DIR="$DATASET_BASE_DIR/hotdog"
        MESH_FILE="$MESH_BASE_DIR/hotdog/hotdog.ply"
        MESH_IMG_DIR="$MESH_BASE_DIR/hotdog"
        MESH_TYPE="milo"
        BUDGETS=(80000 160000 320000)
        ;;
    hotdog_colmap)
        DATASET_DIR="$DATASET_BASE_DIR/hotdog"
        MESH_FILE="/mnt/data1/samk/NEU/dataset/hotdog/colmap_mesh/mesh.ply"
        MESH_IMG_DIR="/mnt/data1/samk/NEU/dataset/hotdog/colmap_mesh"
        MESH_TYPE="colmap"
        BUDGETS=(80000 160000 320000)
        ;;
    bicycle)
        DATASET_DIR="$DATASET_BASE_DIR/bicycle"
        MESH_FILE="$MESH_BASE_DIR/bicycle-dw50/bicycle-dw50.ply"
        MESH_IMG_DIR="$MESH_BASE_DIR/bicycle-dw50"
        MESH_TYPE="milo"
        IMAGES="-i images_4"
        BUDGETS=(320000)  # matches bicycle's established single-budget convention this session
        ;;
    bicycle_colmap)
        DATASET_DIR="$DATASET_BASE_DIR/bicycle"
        MESH_FILE="$SCRIPT_DIR/dataset/colmap/bicycle/downsampled_30/mesh.ply"
        MESH_IMG_DIR="$SCRIPT_DIR/dataset/colmap/bicycle/downsampled_30_lmg_precapture"
        MESH_TYPE="colmap"
        IMAGES="-i images_4"
        BUDGETS=(320000)  # matches bicycle's established single-budget convention this session
        ;;
    ship)
        DATASET_DIR="$DATASET_BASE_DIR/ship"
        MESH_FILE="$MESH_BASE_DIR/ship/ship.ply"
        MESH_IMG_DIR="$MESH_BASE_DIR/ship"
        MESH_TYPE="milo"
        BUDGETS=(80000 160000 320000)
        ;;
    *) echo "unknown scene: $SCENE (expected hotdog|hotdog_colmap|bicycle|bicycle_colmap|ship)"; exit 1 ;;
esac
PY() { conda run -n lmg python "$@"; }

for TOTAL_SPLATS in "${BUDGETS[@]}"; do
    tag="${SCENE}_${TOTAL_SPLATS}_${GS_TYPE}"
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
    PY train_progressive_orchestrator.py --eval -s "$DATASET_DIR" -m "$OUT" $IMAGES \
        --texture_obj_path "$MESH_FILE" --mesh_type "$MESH_TYPE" --gs_type "$GS_TYPE" \
        --alloc_policy distortion_progressive --precaptured_mesh_img_path "$MESH_IMG_DIR" \
        --occlusion --mesh_rasterizer_type nvdiffrast --fixed_alpha \
        --rounds 1 --total_splats "$TOTAL_SPLATS" --iters_per_round "$ITERS" \
        --schedule linear --seed 0 --debugging --debug_freq 4000 > "$LOG" 2>&1 \
        || { echo "[FAIL] $tag (see $LOG)"; continue; }
    echo "[done] $tag"
done

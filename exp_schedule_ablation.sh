#!/bin/bash
# Q1 control experiment (.claude/todo.md): does a `random` splat-growth schedule reach
# the same FINAL PSNR/SSIM as the current `linear` schedule, at the same final
# total_splats? If yes, the progressive freeze-then-grow *mechanism* is the floor, not
# schedule shape. Uses train_progressive_orchestrator.py directly (one process per
# config, no cold-restart-per-round like the old warmup_progressive/train_progressive
# bash pipeline) -- full test-set render+score happens every round (needed for the
# growth/quality curve, not just the final number).
#
#   bash exp_schedule_ablation.sh <group> <gpu>
#
# Groups (2 GPUs):
#   bicycle     : bicycle linear + bicycle quadratic + bicycle random seed=42       (3 configs)
#   hotdog_ship : hotdog random seed 1-5 (run FIRST) + hotdog linear + hotdog quadratic +
#                 ship linear + ship quadratic + ship random seed=42               (10 configs)
# All configs use --fixed_alpha (canonical, non-random barycentric init) as a control.
# hotdog gets 5 random seeds (not just 1) because a single random draw doesn't prove
# schedule-shape doesn't matter; ship/bicycle get 1 random seed each. The hotdog random
# seeds run first within hotdog_ship since they're the actual point of Q1 -- everything
# else is confirmatory.
#
# Launch both groups in parallel, one per GPU:
#   bash exp_schedule_ablation.sh bicycle 0 &
#   bash exp_schedule_ablation.sh hotdog_ship 1 &
#
# Idempotent: a config whose round_summary.json already has ROUNDS entries is skipped.

set -u
GROUP="${1:?usage: exp_schedule_ablation.sh <group> <gpu>}"
export CUDA_VISIBLE_DEVICES="${2:?usage: exp_schedule_ablation.sh <group> <gpu>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/env.local.sh" ] && source "$SCRIPT_DIR/env.local.sh"
: "${DATASET_BASE_DIR:?Set DATASET_BASE_DIR in env.local.sh}"
: "${MESH_BASE_DIR:?Set MESH_BASE_DIR in env.local.sh}"

ROUNDS=4
ITER_PER=8000
DATE="$(date +%Y-%m-%d)"
PY() { conda run -n lmg python "$@"; }

run_config() {
    local scene=$1 schedule=$2 seed=$3
    local DATA MESHDIR TOTAL_SPLATS IMAGES=""
    case "$scene" in
        hotdog)  DATA=hotdog;  MESHDIR=hotdog;       TOTAL_SPLATS=32000  ;;
        ship)    DATA=ship;    MESHDIR=ship;         TOTAL_SPLATS=32000  ;;
        bicycle) DATA=bicycle; MESHDIR=bicycle-dw50; TOTAL_SPLATS=320000; IMAGES="-i images_4" ;;
        *) echo "unknown scene: $scene"; exit 1 ;;
    esac
    local DATASET_DIR="$DATASET_BASE_DIR/$DATA"
    local MESH_FILE="$MESH_BASE_DIR/$MESHDIR/$MESHDIR.ply"
    local MESH_IMG_DIR="$MESH_BASE_DIR/$MESHDIR"

    local tag="${scene}_${schedule}"
    [ "$schedule" = "random" ] && tag="${tag}_seed${seed}"
    local OUT="output/$DATE/$tag"
    local LOG="log/$DATE/schedule_ablation_${tag}.log"
    mkdir -p "$OUT" "$(dirname "$LOG")"

    local done_rounds=0
    [ -f "$OUT/round_summary.json" ] && \
        done_rounds="$(PY -c "import json; print(len(json.load(open('$OUT/round_summary.json'))))" 2>/dev/null || echo 0)"
    if [ "$done_rounds" = "$ROUNDS" ]; then
        echo "[skip] $tag (round_summary.json already has $ROUNDS rounds)"
        return
    fi

    echo "[run] $tag total_splats=$TOTAL_SPLATS schedule=$schedule seed=$seed"
    PY train_progressive_orchestrator.py --eval -s "$DATASET_DIR" -m "$OUT" $IMAGES \
        --texture_obj_path "$MESH_FILE" --mesh_type milo --gs_type lmg \
        --alloc_policy distortion_progressive --precaptured_mesh_img_path "$MESH_IMG_DIR" \
        --occlusion --mesh_rasterizer_type nvdiffrast --fixed_alpha \
        --rounds "$ROUNDS" --total_splats "$TOTAL_SPLATS" --iters_per_round "$ITER_PER" \
        --schedule "$schedule" --seed "$seed" > "$LOG" 2>&1 \
        || { echo "[FAIL] $tag (see $LOG)"; return; }

    PY plotter/plot_growth_quality.py "$OUT" --metric psnr --out "$OUT/growth_quality.png" >> "$LOG" 2>&1
    echo "[done] $tag"
}

case "$GROUP" in
    bicycle)
        CONFIGS=("bicycle:linear:0" "bicycle:quadratic:0" "bicycle:random:42") ;;
    hotdog_ship)
        CONFIGS=("hotdog:random:1" "hotdog:random:2" "hotdog:random:3" "hotdog:random:4" "hotdog:random:5" \
                 "hotdog:linear:0" "hotdog:quadratic:0" \
                 "ship:linear:0" "ship:quadratic:0" "ship:random:42") ;;
    *) echo "unknown group: $GROUP (expected bicycle|hotdog_ship)"; exit 1 ;;
esac

echo "=============== SCHEDULE ABLATION group=$GROUP on GPU $CUDA_VISIBLE_DEVICES ==============="
for cfg in "${CONFIGS[@]}"; do
    IFS=":" read -r scene schedule seed <<< "$cfg"
    run_config "$scene" "$schedule" "$seed"
done
echo "=============== SCHEDULE ABLATION group=$GROUP DONE ==============="

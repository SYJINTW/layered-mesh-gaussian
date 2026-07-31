#!/bin/bash
# RD-curve sweep: 7 alloc policies x 5 splat budgets, LMG++ progressive orchestrator
# (train_progressive_orchestrator.py, --gs_type lmg, --rounds 4), NOT the legacy
# single-round gs_mesh pipeline exp_policy_sweep.sh used -- user's explicit call
# 2026-07-31: "we DO NEED to use the progressive multiround training, otherwise
# we're doing nothing for the LMG++ exp."
#
# Budget ladder is the old LMG convention (BUDGETS=(40000 80000 160000 320000 640000),
# commented in exp_sample.sh/exp_original_lmg.sh/exp_progressive_lmg.sh/_smoke_bike.sh).
# Bicycle uses a 10x-scaled ladder (its mesh is ~8.8x denser than hotdog/ship) -- if a
# bicycle budget OOMs, falls back to the same (unscaled) ladder value for that index.
#
#   bash exp_rd_sweep.sh <scene> <gpu>      # scene: hotdog | ship | bicycle
#
# Idempotent: a (policy,budget) whose results_lmg.json exists is skipped.

set -u
SCENE_ARG="${1:?usage: exp_rd_sweep.sh <scene> <gpu>}"
export CUDA_VISIBLE_DEVICES="${2:?usage: exp_rd_sweep.sh <scene> <gpu>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/env.local.sh" ] && source "$SCRIPT_DIR/env.local.sh"
: "${DATASET_BASE_DIR:?Set DATASET_BASE_DIR in env.local.sh}"
: "${MESH_BASE_DIR:?Set MESH_BASE_DIR in env.local.sh}"

EXP_NAME="rd_sweep_20260731"
# distortion maps to the round-aware distortion_progressive alloc_policy for orchestrator
# runs; every other policy has no progressive variant (known gap, todo.md) and is passed
# through as-is -- output dirs still use the plain "distortion" name for consistency with
# the single-budget sweep's naming.
DEFAULT_POLICIES=(uniform area planarity2 distortion vertex_color_disp2 mixed_area mixed_colordisp)
if [ -n "${POLICIES_OVERRIDE:-}" ]; then
    read -ra POLICIES <<< "$POLICIES_OVERRIDE"
else
    POLICIES=("${DEFAULT_POLICIES[@]}")
fi

LADDER=(40000 80000 160000 320000 640000)
BICYCLE_LADDER=(400000 800000 1600000 3200000 6400000)

IMAGES=""
case "$SCENE_ARG" in
    hotdog)  DATA=hotdog;  MESHDIR=hotdog; BUDGETS=("${LADDER[@]}") ;;
    ship)    DATA=ship;    MESHDIR=ship;   BUDGETS=("${LADDER[@]}") ;;
    bicycle) DATA=bicycle; MESHDIR=bicycle-dw50; BUDGETS=("${BICYCLE_LADDER[@]}"); IMAGES="-i images_4" ;;
    *) echo "unknown scene: $SCENE_ARG"; exit 1 ;;
esac

DATASET_DIR="$DATASET_BASE_DIR/$DATA"
MESH_FILE="$MESH_BASE_DIR/$MESHDIR/$MESHDIR.ply"
MESH_IMG_DIR="$MESH_BASE_DIR/$MESHDIR"
RAST=nvdiffrast
ROUNDS=4
ITERS_PER_ROUND=8000
PY() { conda run -n lmg python "$@"; }

run_one() {
    local POLICY="$1" BUDGET="$2" SAVE_DIR="$3" ALLOC="$4"
    local LOG="${SAVE_DIR}/log_pipeline.log"
    mkdir -p "$SAVE_DIR"
    echo "[run] policy=$POLICY(alloc=$ALLOC) scene=$SCENE_ARG budget=$BUDGET rounds=$ROUNDS" | tee -a "$LOG"
    PY train_progressive_orchestrator.py --eval -s "$DATASET_DIR" -m "$SAVE_DIR" $IMAGES \
        --texture_obj_path "$MESH_FILE" --mesh_type milo --gs_type lmg --alloc_policy "$ALLOC" \
        --precaptured_mesh_img_path "$MESH_IMG_DIR" --occlusion --mesh_rasterizer_type "$RAST" \
        --rounds "$ROUNDS" --total_splats "$BUDGET" --iters_per_round "$ITERS_PER_ROUND" \
        --schedule linear --fixed_alpha \
        >> "$LOG" 2>&1
}

for POLICY in "${POLICIES[@]}"; do
    ALLOC="$POLICY"
    [ "$POLICY" = "distortion" ] && ALLOC="distortion_progressive"

    for i in "${!BUDGETS[@]}"; do
        BUDGET="${BUDGETS[$i]}"
        SAVE_DIR="output/${EXP_NAME}/${SCENE_ARG}/${POLICY}_${BUDGET}_occlusion"

        if [ -f "$SAVE_DIR/results_lmg.json" ]; then
            echo "[skip] $POLICY/$SCENE_ARG@$BUDGET (results exist)"
            continue
        fi

        run_one "$POLICY" "$BUDGET" "$SAVE_DIR" "$ALLOC"
        RC=$?

        if [ $RC -ne 0 ] && [ "$SCENE_ARG" = "bicycle" ] && grep -qi "out of memory" "$SAVE_DIR/log_pipeline.log"; then
            FALLBACK="${LADDER[$i]}"
            echo "[OOM] $POLICY/bicycle@$BUDGET -- falling back to unscaled ladder value $FALLBACK" | tee -a "$SAVE_DIR/log_pipeline.log"
            FALLBACK_DIR="output/${EXP_NAME}/${SCENE_ARG}/${POLICY}_${FALLBACK}_occlusion"
            if [ -f "$FALLBACK_DIR/results_lmg.json" ]; then
                echo "[skip] $POLICY/$SCENE_ARG@$FALLBACK (fallback results exist)"
            else
                run_one "$POLICY" "$FALLBACK" "$FALLBACK_DIR" "$ALLOC"
                RC=$?
            fi
        fi

        if [ $RC -ne 0 ]; then
            echo "[FAIL] $POLICY/$SCENE_ARG@$BUDGET" | tee -a "$SAVE_DIR/log_pipeline.log"
        else
            echo "[done] $POLICY/$SCENE_ARG@$BUDGET"
        fi
    done
done

echo "=== RD sweep complete for $SCENE_ARG ==="

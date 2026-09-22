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
#   bash experiments/exp_rd_sweep.sh <scene> <gpu>
#     scene: hotdog | ship | lego | ficus | mic | bicycle | drjohnson
#     (the LMG paper's 7-scene set, MMSys'26 camera-ready SS5.2)
#
# Idempotent: a (policy,budget) whose results_lmg.json exists is skipped.

set -u
SCENE_ARG="${1:?usage: experiments/exp_rd_sweep.sh <scene> <gpu>}"
export CUDA_VISIBLE_DEVICES="${2:?usage: experiments/exp_rd_sweep.sh <scene> <gpu>}"

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"; cd "$REPO_ROOT"
[ -f "$REPO_ROOT/env.local.sh" ] && source "$REPO_ROOT/env.local.sh"
: "${DATASET_BASE_DIR:?Set DATASET_BASE_DIR in env.local.sh}"
: "${MESH_BASE_DIR:?Set MESH_BASE_DIR in env.local.sh}"

EXP_NAME="_smoke_rd"
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

# Every scene now uses this one ladder, including the two scene-centric ones. Bicycle
# previously used a 10x-scaled ladder (400k-6.4M); user call 2026-08-11: quality barely
# moves as budget grows on that scene, so the special ladder buys nothing and costs a lot.
# Its old scaled-ladder results stay on disk under the same EXP_NAME (different budget
# values => different dir names, no collision) -- do not delete them.
LADDER=(2000)

IMAGES=""
case "$SCENE_ARG" in
    hotdog)    DATA=hotdog;          MESHDIR=hotdog ;;
    ship)      DATA=ship;            MESHDIR=ship ;;
    lego)      DATA=lego;            MESHDIR=lego ;;
    ficus)     DATA=ficus;           MESHDIR=ficus ;;
    mic)       DATA=mic;             MESHDIR=mic ;;
    bicycle)   DATA=bicycle;         MESHDIR=bicycle-dw50;   IMAGES="-i images_4" ;;
    drjohnson) DATA=drjohnson-dw50;  MESHDIR=drjohnson-dw50 ;;
    *) echo "unknown scene: $SCENE_ARG"; exit 1 ;;
esac
BUDGETS=("${LADDER[@]}")

DATASET_DIR="$DATASET_BASE_DIR/$DATA"
MESH_FILE="$MESH_BASE_DIR/$MESHDIR/$MESHDIR.ply"
MESH_IMG_DIR="$MESH_BASE_DIR/$MESHDIR"
RAST=nvdiffrast
ROUNDS=2
ITERS_PER_ROUND=20
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

        # The bicycle OOM->fallback-to-unscaled-ladder retry was removed 2026-08-11 along
        # with BICYCLE_LADDER: every scene is on the one ladder now, so the fallback budget
        # would equal the budget that just failed and re-run into the identical output dir.
        # A mid-round OOM still writes a plausible-looking results_lmg.json from the last
        # completed round (this is how bicycle's 6.4M column came out truncated but looked
        # finished) -- so check the `iteration` key inside results_lmg.json, not the [done]
        # log line, before trusting any config completed at its nominal budget.

        if [ $RC -ne 0 ]; then
            echo "[FAIL] $POLICY/$SCENE_ARG@$BUDGET" | tee -a "$SAVE_DIR/log_pipeline.log"
        else
            echo "[done] $POLICY/$SCENE_ARG@$BUDGET"
        fi
    done
done

echo "=== RD sweep complete for $SCENE_ARG ==="

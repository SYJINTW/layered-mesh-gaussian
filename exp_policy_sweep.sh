#!/bin/bash
# Standalone alloc_policy comparison, single-round gs_mesh, full budget/iters (not a toy
# smoke scale) -- answers "is vertex_color_disp2 any good ON ITS OWN" before any fusion
# question. Runs ALL policies below for ONE scene on ONE gpu, sequentially.
# Launch one instance per scene to parallelize across GPUs.
#
#   bash exp_policy_sweep.sh <scene> <gpu>      # scene: hotdog | ship | bicycle
#
# Idempotent: a policy whose results_gs_mesh.json exists is skipped (safe to relaunch).

set -u
SCENE_ARG="${1:?usage: exp_policy_sweep.sh <scene> <gpu>}"
export CUDA_VISIBLE_DEVICES="${2:?usage: exp_policy_sweep.sh <scene> <gpu>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/env.local.sh" ] && source "$SCRIPT_DIR/env.local.sh"
: "${DATASET_BASE_DIR:?Set DATASET_BASE_DIR in env.local.sh}"
: "${MESH_BASE_DIR:?Set MESH_BASE_DIR in env.local.sh}"

EXP_NAME="policy_sweep_20260727"
DEFAULT_POLICIES=(uniform area screen_footprint planarity2 distortion vertex_color_disp2 \
    mixed_area mixed_area_v3g1 mixed_area_v1g3 mixed_colordisp mixed_colordisp_v3g1 mixed_colordisp_v1g3)
# POLICIES_OVERRIDE (space-separated string) lets a launcher split the full list across
# multiple GPUs for one scene (e.g. bicycle's 12 policies across 2 GPUs, 6 each).
if [ -n "${POLICIES_OVERRIDE:-}" ]; then
    read -ra POLICIES <<< "$POLICIES_OVERRIDE"
else
    POLICIES=("${DEFAULT_POLICIES[@]}")
fi

# ---------- per-scene params (mirrors exp_ablation.sh's case block) ----------
IMAGES=""
case "$SCENE_ARG" in
    hotdog)  DATA=hotdog;  MESHDIR=hotdog;       FINAL=32000  ;;
    ship)    DATA=ship;    MESHDIR=ship;         FINAL=32000  ;;
    bicycle) DATA=bicycle; MESHDIR=bicycle-dw50; FINAL=320000; IMAGES="-i images_4" ;;
    *) echo "unknown scene: $SCENE_ARG"; exit 1 ;;
esac

ITERATION=32000
SAVE_ITERATIONS=(32000)
DATASET_DIR="$DATASET_BASE_DIR/$DATA"
MESH_FILE="$MESH_BASE_DIR/$MESHDIR/$MESHDIR.ply"
MESH_IMG_DIR="$MESH_BASE_DIR/$MESHDIR"
OCC="--occlusion"; OCCTAG=occlusion
RAST=nvdiffrast
PY() { conda run -n lmg python "$@"; }

for POLICY in "${POLICIES[@]}"; do
    SAVE_DIR="output/${EXP_NAME}/${SCENE_ARG}/${POLICY}_${FINAL}_${OCCTAG}"
    LOG="${SAVE_DIR}/log_pipeline.log"
    POLICY_CACHED="${SAVE_DIR}/${POLICY}_${FINAL}.npy"
    mkdir -p "$SAVE_DIR"

    if [ -f "$SAVE_DIR/results_gs_mesh.json" ]; then
        echo "[skip] $POLICY/$SCENE_ARG (results exist)" | tee -a "$LOG"
        continue
    fi

    echo "[run] policy=$POLICY scene=$SCENE_ARG budget=$FINAL iters=$ITERATION" | tee -a "$LOG"

    PY warmup.py --eval --warmup_only -s "$DATASET_DIR" -m "$SAVE_DIR" $IMAGES \
        --texture_obj_path "$MESH_FILE" --mesh_type milo --debugging --debug_freq 1000 $OCC \
        --total_splats "$FINAL" --alloc_policy "$POLICY" --gs_type gs_mesh \
        --policy_path "$POLICY_CACHED" --precaptured_mesh_img_path "$MESH_IMG_DIR" \
        --iteration 1 --mesh_rasterizer_type "$RAST" \
        >> "$LOG" 2>&1 || { echo "[FAIL warmup] $POLICY/$SCENE_ARG" | tee -a "$LOG"; continue; }

    PY train.py --eval -s "$DATASET_DIR" -m "$SAVE_DIR" $IMAGES \
        --texture_obj_path "$MESH_FILE" --mesh_type milo --debugging --debug_freq 1000 $OCC \
        --total_splats "$FINAL" --alloc_policy "$POLICY" --policy_path "$POLICY_CACHED" \
        --precaptured_mesh_img_path "$MESH_IMG_DIR" --gs_type gs_mesh \
        --iteration "$ITERATION" --save_iterations "${SAVE_ITERATIONS[@]}" \
        --mesh_rasterizer_type "$RAST" \
        >> "$LOG" 2>&1 || { echo "[FAIL train] $POLICY/$SCENE_ARG" | tee -a "$LOG"; continue; }

    for it in "${SAVE_ITERATIONS[@]}"; do
        PY render_mesh_splat.py -m "$SAVE_DIR" -s "$DATASET_DIR" --gs_type gs_mesh --skip_train $OCC $IMAGES \
            --total_splats "$FINAL" --alloc_policy "$POLICY" --texture_obj_path "$MESH_FILE" \
            --mesh_type milo --precaptured_mesh_img_path "$MESH_IMG_DIR" \
            --policy_path "$POLICY_CACHED" --iteration "$it" --mesh_rasterizer_type "$RAST" \
            >> "$LOG" 2>&1 || echo "[FAIL render $it] $POLICY/$SCENE_ARG" | tee -a "$LOG"
    done

    PY metrics.py -m "$SAVE_DIR" --gs_type gs_mesh --skip_lpips >> "$LOG" 2>&1 \
        || echo "[FAIL metrics] $POLICY/$SCENE_ARG" | tee -a "$LOG"

    echo "[done] $POLICY/$SCENE_ARG" | tee -a "$LOG"
done

echo "=== policy sweep complete for $SCENE_ARG ==="

#!/bin/bash
# 2x2 ablation of the two LMG++ mechanisms (Progressive freeze, Fixed alpha) vs LMGModel baseline.
# Runs ALL 4 configs for ONE scene on ONE gpu, sequentially. Launch one instance per scene to parallelize.
#
#   bash exp_ablation.sh <scene> <gpu>      # scene: hotdog | ship | bicycle
#
# Configs (all gs_type=lmg, all end at FINAL splats / 32000 iters):
#   1 baseline    : single round, random alpha            (prog OFF, fixed OFF)
#   2 fixed-only  : single round, fixed alpha             (prog OFF, fixed ON)
#   3 prog-only   : 4 rounds x per-round, random alpha    (prog ON,  fixed OFF)
#   4 full LMG++  : 4 rounds x per-round, fixed alpha     (prog ON,  fixed ON)
# Single-round configs save checkpoints at 8k/16k/24k/32k iters and render each -> iteration curve.
# Idempotent: a config whose results_lmg.json exists is skipped.

set -u
SCENE_ARG="${1:?usage: exp_ablation.sh <scene> <gpu>}"
export CUDA_VISIBLE_DEVICES="${2:?usage: exp_ablation.sh <scene> <gpu>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/env.local.sh" ] && source "$SCRIPT_DIR/env.local.sh"
: "${DATASET_BASE_DIR:?Set DATASET_BASE_DIR in env.local.sh}"
: "${MESH_BASE_DIR:?Set MESH_BASE_DIR in env.local.sh}"

# ---------- per-scene params ----------
IMAGES=""        # extra args for colmap image folder selection
WBG=""           # white background flag
case "$SCENE_ARG" in
    hotdog)  DATA=hotdog;  MESHDIR=hotdog;       PERROUND=8000  ;;
    ship)    DATA=ship;    MESHDIR=ship;         PERROUND=8000  ;;
    bicycle) DATA=bicycle; MESHDIR=bicycle-dw50; PERROUND=80000; IMAGES="-i images_4" ;;
    *) echo "unknown scene: $SCENE_ARG"; exit 1 ;;
esac

ROUNDS=4
ITER_PER=8000
FINAL=$((PERROUND * ROUNDS))        # 32000 (synthetic) / 320000 (bicycle)
TOTAL_ITER=$((ITER_PER * ROUNDS))   # 32000 for all
SAVE_ITERS=(8000 16000 24000 32000) # checkpoint iters (single-round curve)

DATASET_DIR="$DATASET_BASE_DIR/$DATA"
MESH_FILE="$MESH_BASE_DIR/$MESHDIR/$MESHDIR.ply"
MESH_IMG_DIR="$MESH_BASE_DIR/$MESHDIR"
POLICY=distortion_progressive
OCC="--occlusion"; OCCTAG=occlusion
RAST=nvdiffrast
PY() { conda run -n lmg python "$@"; }

# ---------- config 1/2: single round, checkpoint-rendered curve ----------
run_single() {
    local fixed=$1 exp=$2
    local BASE="output/$exp/$SCENE_ARG/${POLICY}_${FINAL}_${OCCTAG}/iteration_0"
    local CACHED="$BASE/load_iter_0/${POLICY}_${FINAL}.npy"
    local LOG="$BASE/log_ablation.log"
    [ -f "$BASE/results_lmg.json" ] && { echo "[skip] $exp/$SCENE_ARG (results exist)"; return; }
    mkdir -p "$BASE"
    local FA=""; [ "$fixed" = true ] && FA="--fixed_alpha"
    echo "[run] $exp/$SCENE_ARG single-round FINAL=$FINAL fixed_alpha=$fixed" | tee -a "$LOG"

    PY warmup_progressive.py --eval $FA --warmup_only -s "$DATASET_DIR" -m "$BASE" $IMAGES \
        --texture_obj_path "$MESH_FILE" --mesh_type milo --debugging --gs_type lmg $OCC \
        --total_splats "$FINAL" --alloc_policy "$POLICY" --policy_path "$CACHED" \
        --precaptured_mesh_img_path "$MESH_IMG_DIR" --load_iter 0 \
        --mesh_rasterizer_type "$RAST" >> "$LOG" 2>&1 || { echo "[FAIL warmup] $exp/$SCENE_ARG"; return; }

    PY train_progressive.py --eval -s "$DATASET_DIR" -m "$BASE" $IMAGES \
        --texture_obj_path "$MESH_FILE" --mesh_type milo --debugging --debug_freq 1000 $OCC \
        --total_splats "$FINAL" --alloc_policy "$POLICY" --policy_path "$CACHED" \
        --precaptured_mesh_img_path "$MESH_IMG_DIR" --gs_type lmg $WBG \
        --iteration "$TOTAL_ITER" --save_iterations "${SAVE_ITERS[@]}" \
        --mesh_rasterizer_type "$RAST" \
        --load_gs_path "$BASE/point_cloud/iteration_0_warmup/point_cloud.ply" \
        --start_iteration 0 >> "$LOG" 2>&1 || { echo "[FAIL train] $exp/$SCENE_ARG"; return; }

    PY generate_dummy_cfg.py -m "$BASE" >> "$LOG" 2>&1
    for it in "${SAVE_ITERS[@]}"; do
        PY render_mesh_splat_progressive.py -m "$BASE" --gs_type lmg --skip_train $OCC \
            --total_splats "$FINAL" --alloc_policy "$POLICY" --texture_obj_path "$MESH_FILE" \
            --mesh_type milo --precaptured_mesh_img_path "$MESH_IMG_DIR" $WBG $IMAGES \
            --policy_path "$CACHED" --iteration "$it" --mesh_rasterizer_type "$RAST" \
            --load_gs_path "$BASE/point_cloud/iteration_$it/point_cloud.ply" >> "$LOG" 2>&1 \
            || echo "[FAIL render $it] $exp/$SCENE_ARG"
    done
    PY metrics.py -m "$BASE" --gs_type lmg --skip_lpips >> "$LOG" 2>&1 || echo "[FAIL metrics] $exp/$SCENE_ARG"
    echo "[done] $exp/$SCENE_ARG" | tee -a "$LOG"
}

# ---------- config 3/4: progressive, render per round ----------
run_progressive() {
    local fixed=$1 exp=$2
    local FA=""; [ "$fixed" = true ] && FA="--fixed_alpha"
    local OUTROOT="output/$exp/$SCENE_ARG/${POLICY}_${PERROUND}_${OCCTAG}"
    [ -f "$OUTROOT/iteration_$((TOTAL_ITER - ITER_PER))/results_lmg.json" ] && \
        { echo "[skip] $exp/$SCENE_ARG (final round results exist)"; return; }
    echo "[run] $exp/$SCENE_ARG progressive PERROUND=$PERROUND rounds=$ROUNDS fixed_alpha=$fixed"
    local CUR=0 PREV=""
    for (( R=1; R<=ROUNDS; R++ )); do
        local BASE="$OUTROOT/iteration_$CUR"
        local CACHED="$BASE/load_iter_$CUR/${POLICY}_${PERROUND}.npy"
        local LOG="$BASE/log_ablation.log"
        mkdir -p "$BASE"
        local WGS="" WALPHA="" FND=""
        if (( R > 1 )); then
            WGS="--gs_path $PREV/point_cloud/iteration_${CUR}/point_cloud.ply"
            WALPHA="--alpha_path $PREV/static_alpha.pt"
            FND="--foundation_pt_path $PREV/point_cloud/iteration_${CUR}/model_params.pt"
        fi

        PY warmup_progressive.py --eval $FA --warmup_only -s "$DATASET_DIR" -m "$BASE" $IMAGES \
            --texture_obj_path "$MESH_FILE" --mesh_type milo --debugging --gs_type lmg $OCC \
            --total_splats "$PERROUND" --alloc_policy "$POLICY" --policy_path "$CACHED" \
            --precaptured_mesh_img_path "$MESH_IMG_DIR" --load_iter "$CUR" $WGS $WALPHA \
            --mesh_rasterizer_type "$RAST" >> "$LOG" 2>&1 || { echo "[FAIL warmup R$R] $exp/$SCENE_ARG"; return; }

        PY train_progressive.py --eval -s "$DATASET_DIR" -m "$BASE" $IMAGES \
            --texture_obj_path "$MESH_FILE" --mesh_type milo --debugging --debug_freq 1000 $OCC \
            --total_splats "$PERROUND" --alloc_policy "$POLICY" --policy_path "$CACHED" \
            --precaptured_mesh_img_path "$MESH_IMG_DIR" --gs_type lmg $WBG \
            --iteration "$ITER_PER" --save_iterations "$ITER_PER" --mesh_rasterizer_type "$RAST" \
            --load_gs_path "$BASE/point_cloud/iteration_${CUR}_warmup/point_cloud.ply" \
            $FND --start_iteration "$CUR" >> "$LOG" 2>&1 || { echo "[FAIL train R$R] $exp/$SCENE_ARG"; return; }

        local RIT=$((CUR + ITER_PER))
        PY generate_dummy_cfg.py -m "$BASE" >> "$LOG" 2>&1
        PY render_mesh_splat_progressive.py -m "$BASE" --gs_type lmg --skip_train $OCC \
            --total_splats "$PERROUND" --alloc_policy "$POLICY" --texture_obj_path "$MESH_FILE" \
            --mesh_type milo --precaptured_mesh_img_path "$MESH_IMG_DIR" $WBG $IMAGES \
            --policy_path "$CACHED" --iteration "$RIT" --mesh_rasterizer_type "$RAST" \
            --load_gs_path "$BASE/point_cloud/iteration_${RIT}/point_cloud.ply" >> "$LOG" 2>&1 \
            || echo "[FAIL render R$R] $exp/$SCENE_ARG"
        PY metrics.py -m "$BASE" --gs_type lmg --skip_lpips >> "$LOG" 2>&1 || echo "[FAIL metrics R$R] $exp/$SCENE_ARG"

        PREV="$BASE"; CUR=$RIT
        echo "[done R$R] $exp/$SCENE_ARG -> next CUR=$CUR"
    done
}

echo "=============== ABLATION $SCENE_ARG on GPU $CUDA_VISIBLE_DEVICES ==============="
run_single      false ablation_single_randalpha   # config 1: baseline
run_single      true  ablation_single_fixedalpha  # config 2: fixed-only
run_progressive false ablation_prog_randalpha     # config 3: prog-only
run_progressive true  ablation_prog_fixedalpha    # config 4: full LMG++
echo "=============== ABLATION $SCENE_ARG DONE ==============="

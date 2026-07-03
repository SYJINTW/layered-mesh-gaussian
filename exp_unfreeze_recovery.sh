#!/bin/bash
# Local-vs-global optimum experiment (Task #7, .claude/todo.md): continue each
# prog_{rand,fixed} config's final round for N more iterations with --unfreeze,
# then compare PSNR/SSIM against the original frozen result at the same iteration
# count. If PSNR recovers toward single_rand's level, that's direct evidence the
# frozen state was a correctable local optimum, not a genuine ceiling.
#
#   bash exp_unfreeze_recovery.sh <gpu>
#
set -u
export CUDA_VISIBLE_DEVICES="${1:?usage: exp_unfreeze_recovery.sh <gpu>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/env.local.sh" ] && source "$SCRIPT_DIR/env.local.sh"
: "${DATASET_BASE_DIR:?Set DATASET_BASE_DIR in env.local.sh}"
: "${MESH_BASE_DIR:?Set MESH_BASE_DIR in env.local.sh}"

EXTRA_ITERS=4000
PY() { conda run -n lmg python "$@"; }

run_one() {
    local scene=$1 alpha=$2  # alpha: rand | fixed
    case "$scene" in
        hotdog)  DATA=hotdog;  MESHDIR=hotdog;       PERROUND=8000;  IMAGES="" ;;
        ship)    DATA=ship;    MESHDIR=ship;         PERROUND=8000;  IMAGES="" ;;
        bicycle) DATA=bicycle; MESHDIR=bicycle-dw50; PERROUND=80000; IMAGES="-i images_4" ;;
        *) echo "unknown scene: $scene"; return 1 ;;
    esac
    # ITER_PER is fixed at 8000/round for every scene regardless of splat budget
    # (exp_ablation.sh: ITER_PER=8000, ROUNDS=4) -- round *directory* boundaries
    # (iteration_0/8000/16000/24000) follow this, NOT the scene-specific PERROUND
    # splat count. FINAL_SPLATS (total_splats arg) does scale per scene.
    local ITER_PER=8000
    local FINAL_ITER=$((ITER_PER * 4))       # 32000 for every scene
    local FINAL_SPLATS=$((PERROUND * 4))     # 32000 hotdog/ship, 320000 bicycle
    local ABL_DIR="output/ablation_prog_${alpha}alpha/${scene}/distortion_progressive_${PERROUND}_occlusion"
    local R4_DIR="${ABL_DIR}/iteration_$((ITER_PER*3))"   # round 4's own dir (iteration_24000)
    local R3_DIR="${ABL_DIR}/iteration_$((ITER_PER*2))"   # round 3's dir (iteration_16000)
    local OUT_DIR="output/unfreeze_recovery/${scene}_${alpha}"
    local LOG="${OUT_DIR}/log.log"

    if [ ! -f "${R4_DIR}/point_cloud/iteration_${FINAL_ITER}/point_cloud.ply" ]; then
        echo "[skip] ${scene}/${alpha}: no final-round checkpoint at ${R4_DIR}/point_cloud/iteration_${FINAL_ITER}"
        return
    fi
    if [ -f "${OUT_DIR}/results_lmg.json" ]; then
        echo "[skip] ${scene}/${alpha}: results already exist"
        return
    fi

    mkdir -p "$OUT_DIR"
    cp -r "${R4_DIR}/load_iter_$((ITER_PER*3))" "${OUT_DIR}/load_iter_$((ITER_PER*3))" 2>/dev/null
    local DATASET_DIR="${DATASET_BASE_DIR}/${DATA}"
    local MESH_FILE="${MESH_BASE_DIR}/${MESHDIR}/${MESHDIR}.ply"
    local MESH_IMG_DIR=$(dirname "$MESH_FILE")
    local NEW_ITER=$((FINAL_ITER + EXTRA_ITERS))
    local POLICY_NPY="${OUT_DIR}/load_iter_$((ITER_PER*3))/distortion_progressive_${FINAL_SPLATS}.npy"

    echo "=== ${scene}/${alpha}: unfreeze-continue ${FINAL_ITER} -> ${NEW_ITER} ===" | tee "$LOG"

    if PY train_progressive.py --eval \
        -s "$DATASET_DIR" -m "$OUT_DIR" $IMAGES \
        --texture_obj_path "$MESH_FILE" --mesh_type milo \
        --gs_type lmg --total_splats "$FINAL_SPLATS" --alloc_policy distortion_progressive \
        --policy_path "$POLICY_NPY" \
        --precaptured_mesh_img_path "$MESH_IMG_DIR" --occlusion \
        --mesh_rasterizer_type nvdiffrast \
        --iteration "$EXTRA_ITERS" --save_iterations "$EXTRA_ITERS" --start_iteration "$FINAL_ITER" \
        --load_gs_path "${R4_DIR}/point_cloud/iteration_${FINAL_ITER}/point_cloud.ply" \
        --foundation_pt_path "${R3_DIR}/point_cloud/iteration_$((ITER_PER*3))/model_params.pt" \
        --unfreeze \
        >> "$LOG" 2>&1; then
        echo "[train ok] ${scene}/${alpha}" | tee -a "$LOG"
    else
        echo "[train FAILED] ${scene}/${alpha}" | tee -a "$LOG"
        return
    fi

    PY generate_dummy_cfg.py -m "$OUT_DIR" -s "$DATASET_DIR" >> "$LOG" 2>&1
    if PY render_mesh_splat_progressive.py \
        -m "$OUT_DIR" --gs_type lmg --skip_train --occlusion \
        --total_splats "$FINAL_SPLATS" --alloc_policy distortion_progressive \
        --texture_obj_path "$MESH_FILE" --mesh_type milo \
        --precaptured_mesh_img_path "$MESH_IMG_DIR" \
        --policy_path "$POLICY_NPY" \
        --iteration "$NEW_ITER" --mesh_rasterizer_type nvdiffrast \
        --load_gs_path "${OUT_DIR}/point_cloud/iteration_${NEW_ITER}/point_cloud.ply" \
        >> "$LOG" 2>&1; then
        echo "[render ok] ${scene}/${alpha}" | tee -a "$LOG"
    else
        echo "[render FAILED] ${scene}/${alpha}" | tee -a "$LOG"
        return
    fi

    PY metrics.py -m "$OUT_DIR" --gs_type lmg --skip_lpips >> "$LOG" 2>&1
    echo "[done] ${scene}/${alpha}" | tee -a "$LOG"
}

for scene in hotdog ship bicycle; do
    for alpha in rand fixed; do
        run_one "$scene" "$alpha"
    done
done

echo "================================================================="
echo "All unfreeze-recovery runs attempted. Results under output/unfreeze_recovery/*/results_lmg.json"
echo "================================================================="

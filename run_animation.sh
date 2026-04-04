#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

# ======= Config ======
EXP_NAME="0404_ship_animation"
SCENE_NAME="ship"
ITERATION="15000"
BUDGET="313547"
POLICY="distortion"

DATASET_DIR="/mnt/data1/samk/NEU/sorted_dataset/${SCENE_NAME}"
MESH_TYPE="milo"
MESH_FILE="/mnt/data1/samk/NEU/sorted_dataset/milo_meshes/${SCENE_NAME}/${SCENE_NAME}.ply"

# Set to true if this checkpoint was trained with occlusion enabled.
USE_OCCLUSION=true

# Animation settings
# choices: ficus_sinus, hotdog_wave_z, hotdog_parabola_z, hotdog_radial_lift, hotdog_fly, 
#          ficus_pot, ship_sinus, make_smaller, no_anim, none
TRANSFORM="hotdog_wave_z"
FPS=30
SKIP_TRAIN=true

# Optional: Load custom camera trajectory from JSON file (leave empty to use dataset cameras)
CAMERA_JSON=""  # example: "custom_cameras.json"

# Optional flags
IS_WHITE_BG=false
RESOLUTION_FACTOR=""  # example: 4

if [ "$USE_OCCLUSION" = true ]; then
    OCCLUSION_TAG="occlusion"
else
    OCCLUSION_TAG="no_occlusion"
fi

MODEL_PATH="output/${EXP_NAME}/${SCENE_NAME}/${POLICY}_${BUDGET}_${OCCLUSION_TAG}"
POLICY_CACHED="${MODEL_PATH}/${POLICY}_${BUDGET}.npy"
LOG_PATH="${MODEL_PATH}/animate_${TRANSFORM}_${ITERATION}.log"

RENDER_DIR="${MODEL_PATH}/test/ours_${ITERATION}/renders_animated_gs_mesh"
VIDEO_PATH="${MODEL_PATH}/test/ours_${ITERATION}/${TRANSFORM}_${OCCLUSION_TAG}_animation.mp4"

# ======= Sanity checks ======
if [ ! -d "$MODEL_PATH" ]; then
    echo "Model path not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "${MODEL_PATH}/point_cloud/iteration_${ITERATION}/model_params.pt" ]; then
    echo "Checkpoint not found: ${MODEL_PATH}/point_cloud/iteration_${ITERATION}/model_params.pt"
    exit 1
fi

if [ ! -f "$MESH_FILE" ]; then
    echo "Mesh file not found: $MESH_FILE"
    exit 1
fi

if [ ! -f "$POLICY_CACHED" ]; then
    echo "Policy cache not found: $POLICY_CACHED"
    exit 1
fi

# ======= Run Animation ======
CMD=(
    python render_mesh_splat_animated.py
    -m "$MODEL_PATH"
    -s "$DATASET_DIR"
    --gs_type gs_mesh
    --total_splats "$BUDGET"
    --alloc_policy "$POLICY"
    --texture_obj_path "$MESH_FILE"
    --mesh_type "$MESH_TYPE"
    --policy_path "$POLICY_CACHED"
    --transform "$TRANSFORM"
    --iteration "$ITERATION"
)

if [ "$USE_OCCLUSION" = true ]; then
    CMD+=(--occlusion)
fi

if [ "$SKIP_TRAIN" = true ]; then
    CMD+=(--skip_train)
fi

if [ "$IS_WHITE_BG" = true ]; then
    CMD+=(--white_background)
fi

if [ -n "$RESOLUTION_FACTOR" ]; then
    CMD+=(--resolution "$RESOLUTION_FACTOR")
fi

if [ -n "$CAMERA_JSON" ] && [ -f "$CAMERA_JSON" ]; then
    CMD+=(--camera_json "$CAMERA_JSON")
fi

echo "Running animation render:"
printf ' %q' "${CMD[@]}"
echo

mkdir -p "$(dirname "$LOG_PATH")"
"${CMD[@]}" 2>&1 | tee "$LOG_PATH"

echo "Animation rendering completed."
echo "Frame output directory: ${RENDER_DIR}"

if [ -d "$RENDER_DIR" ]; then
    echo "Combining frames into video..."
    ffmpeg -y -framerate "$FPS" -i "${RENDER_DIR}/%05d.png" -c:v libx264 -pix_fmt yuv420p "$VIDEO_PATH"
    echo "Video saved to: $VIDEO_PATH"
else
    echo "Render directory not found: $RENDER_DIR"
    exit 1
fi
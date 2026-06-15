#!/bin/bash
# This script runs a pipeline of warmup, training, rendering, and metrics for each experiment.
# It does not exit on the first error, but continues to the next experiment.
# set -e

# [NOTE] copy and modify the config for your own experiments

export CUDA_VISIBLE_DEVICES=1

# ======= Config ======

PROGRESSIVE_ROUND="4"

# in decreasing order
# 1 means only mesh, no splats
# BUDGETS=(40000 80000 160000 320000 640000)
BUDGETS=(8000)

# POLICIES=("uniform" "area" "planarity2" "distortion")
POLICIES=("distortion_progressive")

# "--occlusion" or ""
WHETHER_OCCLUSION=("--occlusion") 

# can do sanity check in the logfile
PROJECT_GLOBAL_PATH="/mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian" #! Change to your project path

DATASET_BASE_DIR="${PROJECT_GLOBAL_PATH}/dataset/images" #! Change to your dataset path
MESH_BASE_DIR="${PROJECT_GLOBAL_PATH}/dataset/meshes" #! Change to your mesh path

ITERATION="8000"
# ITERATION="1000"

# SAVE_ITERATIONS=("7000") 
SAVE_ITERATIONS=("4000") # Need to fill in some iterations or it will failed

EXP_NAME="progressive" #! Change to your experiment name

# SCENE_NAME_LIST=("ficus" "hotdog" "lego" "mic" "ship")
SCENE_NAME_LIST=("hotdog")

MESH_TYPE="milo" # "sugar" or "colmap" or "milo"
MESH_RASTERIZER_TYPE="nvdiffrast" # "pytorch3d" or "nvdiffrast"

RESOLUTION="" # or "--resolution 4" for faster debugging
IS_WHITE_BG="" # set to "--white_background" if the dataset has white background

# DEBUGGING_FREQ="1000"
DEBUGGING_FREQ="1000"

SKIP_LPIPS=true # true or false; set to true to skip lpips computation to save time

for SCENE_NAME in "${SCENE_NAME_LIST[@]}"; do
    DATASET_DIR="${DATASET_BASE_DIR}/${SCENE_NAME}" 
    MESH_FILE="${MESH_BASE_DIR}/${SCENE_NAME}/${SCENE_NAME}.ply"

    MESH_IMG_DIR=$(dirname "$MESH_FILE")

    BASE_OUTPUT_DIR="output/${EXP_NAME}/${SCENE_NAME}"
    BASE_LOG_DIR="log/${EXP_NAME}/${SCENE_NAME}"

    PLOT_DIR="${BASE_OUTPUT_DIR}/for_plot"

    # ======= Helpers ======

    fmt_time() {
        local T=$1
        printf "%02d:%02d:%02d" $((T/3600)) $(((T%3600)/60)) $((T%60))
    }
    total_start=$(date +%s)
    total_exp_seconds=0
    failed_experiments=0

    # ======= Setup Output Dirs and Logs ======

    mkdir -p "$BASE_OUTPUT_DIR"
    mkdir -p "$BASE_LOG_DIR"
    mkdir -p "$PLOT_DIR"

    # Timing summary file
    TIMING_SUMMARY="${BASE_OUTPUT_DIR}/pipeline_timing_summary.tsv"
    echo -e "policy\tbudget\trenderer\twarmup_secs\ttrain_secs\trender_secs\tmetrics_secs\ttotal_secs\tstatus" > "$TIMING_SUMMARY"

    # Failed experiments log
    FAILED_LOG="${BASE_OUTPUT_DIR}/failed_experiments.log"
    > "$FAILED_LOG" # Clear the file

    # ======= Main Loop ======
    # for scene_name in "${SCENE_NAMES[@]}"; do
    CURRENT_ITERATION=0
    for IS_OCCLUSION in "${WHETHER_OCCLUSION[@]}"; do
        for policy in "${POLICIES[@]}"; do
            for budget in "${BUDGETS[@]}"; do
                occlusion_tag="no_occlusion"
                if [ "$IS_OCCLUSION" == "--occlusion" ]; then
                    occlusion_tag="occlusion"
                fi

                PREV_SAVE_DIR=""
                for (( ROUND=1; ROUND<=${PROGRESSIVE_ROUND}; ROUND++ )); do
                    
                    SAVE_DIR="${BASE_OUTPUT_DIR}/${policy}_${budget}_${occlusion_tag}/iteration_${CURRENT_ITERATION}"
                    LOG_FILE="${SAVE_DIR}/log_pipeline_${policy}_${budget}_${occlusion_tag}.log"
                    POLICY_CACHED="${SAVE_DIR}/load_iter_${CURRENT_ITERATION}/${policy}_${budget}.npy"

                    # Ensure the save directory exists
                    mkdir -p "$SAVE_DIR"
                    
                    # For the first round, we start with the original policy without loading from previous iteration
                    if ((ROUND == 1)); then
                        echo "Starting round ${ROUND}/${PROGRESSIVE_ROUND} for policy=${policy}, budget=${budget}, occlusion=${occlusion_tag} from scratch (no loading)..." | tee -a "$LOG_FILE"
                        # Parameters for warmup
                        WARMUP_GS_PATH=""
                        WARMUP_ALPHA_PATH=""
                        # Parameters for training
                        TRAINING_FOUNDATION_PT_PATH="" 
                    else
                        echo "Starting round ${ROUND}/${PROGRESSIVE_ROUND} for policy=${policy}, budget=${budget}, occlusion=${occlusion_tag} loading from iteration ${CURRENT_ITERATION}..." | tee -a "$LOG_FILE"
                        # Parameters for warmup
                        WARMUP_GS_PATH="--gs_path ${PREV_SAVE_DIR}/point_cloud/iteration_${CURRENT_ITERATION}/point_cloud.ply"
                        WARMUP_ALPHA_PATH="--alpha_path ${PREV_SAVE_DIR}/static_alpha.pt"
                        # Parameters for training
                        TRAINING_FOUNDATION_PT_PATH="--foundation_pt_path ${PREV_SAVE_DIR}/point_cloud/iteration_${CURRENT_ITERATION}/model_params.pt"
                    fi
                    
                    # ======= Step 0: Warmup ======
                    echo "Step 0/3: Running warmup..." | tee -a "$LOG_FILE"
                    if python warmup_progressive.py --eval \
                        --warmup_only \
                        -s "$DATASET_DIR" \
                        -m "$SAVE_DIR" \
                        --texture_obj_path "$MESH_FILE" \
                        --mesh_type "$MESH_TYPE" \
                        --debugging \
                        --gs_type lmg \
                        $IS_OCCLUSION \
                        --total_splats "$budget" \
                        --alloc_policy "$policy" \
                        --policy_path "$POLICY_CACHED" \
                        --precaptured_mesh_img_path "$MESH_IMG_DIR" \
                        --load_iter "${CURRENT_ITERATION}" \
                        $WARMUP_GS_PATH \
                        $WARMUP_ALPHA_PATH \
                        --mesh_rasterizer_type "$MESH_RASTERIZER_TYPE" \
                        >> "$LOG_FILE" ; then
                        # this is warmup

                        warmup_end=$(date +%s)
                        warmup_secs=$((warmup_end - warmup_start))
                        echo "Warmup completed in $(fmt_time $warmup_secs) (${warmup_secs}s)." | tee -a "$LOG_FILE"
                        exp_status="WARMUP_SUCCESS"
                    else
                        warmup_end=$(date +%s)
                        warmup_secs=$((warmup_end - warmup_start))
                        exp_status="WARMUP_FAILED"
                        failed_experiments=$((failed_experiments + 1))
                        echo "ERROR: Warmup failed for policy=${policy}, budget=${budget}, occlusion=${occlusion_tag} after ${warmup_secs}s." | tee -a "$LOG_FILE" "$FAILED_LOG"
                    fi
                    
                    # ======= Step 1: Train ======
                    if [ "$exp_status" = "WARMUP_SUCCESS" ]; then
                        echo "Step 1/3: Running training..." | tee -a "$LOG_FILE"
                        GS_PATH="${SAVE_DIR}/point_cloud/iteration_${CURRENT_ITERATION}_warmup/point_cloud.ply"
                        if python train_progressive.py --eval \
                            -s "$DATASET_DIR" \
                            -m "$SAVE_DIR" \
                            --texture_obj_path "$MESH_FILE" \
                            --mesh_type "$MESH_TYPE" \
                            --debugging \
                            --debug_freq "$DEBUGGING_FREQ" \
                            $IS_OCCLUSION \
                            --total_splats "$budget" \
                            --alloc_policy "$policy" \
                            --policy_path "$POLICY_CACHED" \
                            --precaptured_mesh_img_path "$MESH_IMG_DIR" \
                            --gs_type lmg \
                            $IS_WHITE_BG \
                            $RESOLUTION \
                            --iteration "$ITERATION" \
                            --save_iterations "${SAVE_ITERATIONS[@]}" \
                            --mesh_rasterizer_type "$MESH_RASTERIZER_TYPE" \
                            --load_gs_path "$GS_PATH" \
                            $TRAINING_FOUNDATION_PT_PATH \
                            --start_iteration "${CURRENT_ITERATION}" \
                            >> "$LOG_FILE"; then
                            
                            train_end=$(date +%s)
                            train_secs=$((train_end - train_start))
                            echo "Training completed in $(fmt_time $train_secs) (${train_secs}s)." | tee -a "$LOG_FILE"
                            exp_status="TRAIN_SUCCESS"
                        else
                            train_end=$(date +%s)
                            train_secs=$((train_end - train_start))
                            exp_status="TRAIN_FAILED"
                            failed_experiments=$((failed_experiments + 1))
                            echo "ERROR: Training failed for policy=${policy}, budget=${budget}, occlusion=${occlusion_tag} after ${train_secs}s." | tee -a "$LOG_FILE" "$FAILED_LOG"
                        fi
                    fi

                    # ======= Step 2: Render ======
                    exp_status="TRAIN_SUCCESS"
                    if [ "$exp_status" = "TRAIN_SUCCESS" ]; then
                        echo "Step 2/3: Running render (iteration ${iter})..." | tee -a "$LOG_FILE"
                        render_start=$(date +%s)
                        RENDER_ITERATION=$((CURRENT_ITERATION + ITERATION))
                        
                        python generate_dummy_cfg.py \
                            -m "$SAVE_DIR"

                        if python render_mesh_splat_progressive.py \
                            -m "$SAVE_DIR" \
                            --gs_type lmg \
                            --skip_train \
                            $IS_OCCLUSION \
                            --total_splats "$budget" \
                            --alloc_policy "$policy" \
                            --texture_obj_path "$MESH_FILE" \
                            --mesh_type "$MESH_TYPE" \
                            --precaptured_mesh_img_path "$MESH_IMG_DIR" \
                            $RESOLUTION \
                            $IS_WHITE_BG \
                            --policy_path "$POLICY_CACHED" \
                            --iteration $RENDER_ITERATION \
                            --mesh_rasterizer_type "$MESH_RASTERIZER_TYPE" \
                            --load_gs_path "${SAVE_DIR}/point_cloud/iteration_${RENDER_ITERATION}/point_cloud.ply" \
                            >> "$LOG_FILE" ; then

                            render_end=$(date +%s)
                            render_secs=$((render_end - render_start))
                            echo "Render (iter ${iter}) completed in $(fmt_time $render_secs) (${render_secs}s)." | tee -a "$LOG_FILE"
                        else
                            render_end=$(date +%s)
                            render_secs=$((render_end - render_start))
                            render_success=false
                            failed_experiments=$((failed_experiments + 1))
                            echo "ERROR: Render failed for policy=${policy}, budget=${budget}, occlusion=${occlusion_tag}, iter=${iter} after ${render_secs}s." | tee -a "$LOG_FILE" "$FAILED_LOG"
                        fi
                    fi

                    # ======= Step 3: Metrics ======
                    exp_status="RENDER_SUCCESS"
                    if [ "$exp_status" = "RENDER_SUCCESS" ]; then
                        echo "Step 3/3: Running metrics evaluation..." | tee -a "$LOG_FILE"
                        metrics_start=$(date +%s)
                        
                        if [ "$SKIP_LPIPS" = true ]; then
                            SKIP_LPIPS_ARG="--skip_lpips"
                        else
                            SKIP_LPIPS_ARG=""
                        fi
                        
                        if python metrics.py \
                            -m "$SAVE_DIR" \
                            --gs_type lmg \
                            --skip_lpips
                            >> "$LOG_FILE"; then

                            metrics_end=$(date +%s)
                            metrics_secs=$((metrics_end - metrics_start))
                            echo "Metrics completed in $(fmt_time $metrics_secs) (${metrics_secs}s)." | tee -a "$LOG_FILE"
                            exp_status="SUCCESS"
                        else
                            metrics_end=$(date +%s)
                            metrics_secs=$((metrics_end - metrics_start))
                            exp_status="METRICS_FAILED"
                            failed_experiments=$((failed_experiments + 1))
                            echo "ERROR: Metrics failed for policy=${policy}, budget=${budget}, occlusion=${occlusion_tag} after ${metrics_secs}s." | tee -a "$LOG_FILE" "$FAILED_LOG"
                        fi
                    fi

                    PREV_SAVE_DIR="$SAVE_DIR"
                    CURRENT_ITERATION=$((CURRENT_ITERATION + ITERATION))
                    echo "Completed round ${ROUND}/${PROGRESSIVE_ROUND} for policy=${policy}, budget=${budget}, occlusion=${occlusion_tag}. Next iteration starts from ${CURRENT_ITERATION}." | tee -a "$LOG_FILE"
                done
            done
        done
    done

    # total_end=$(date +%s)
    # wall_secs=$((total_end - total_start))
    # echo "================================================================="
    # echo "All pipelines completed."
    # echo "Wall-clock total: $(fmt_time "$wall_secs") (${wall_secs}s)"
    # echo "Sum of experiment durations: $(fmt_time "$total_exp_seconds") (${total_exp_seconds}s)"
    # printf "TOTAL\t\t\t%d\t%d\t%d\t%d\t%d\tTOTAL_SUM\n" "$warmup_secs" "$train_secs" "$render_secs" "$metrics_secs" "$total_exp_seconds" >> "$TIMING_SUMMARY"
    # echo "Failed experiments: ${failed_experiments}"
    # echo "Timing summary saved to: ${TIMING_SUMMARY}"
    # if [ $failed_experiments -gt 0 ]; then
    #     echo "Failed experiments log: ${FAILED_LOG}"
    # fi
    # echo "================================================================="
done

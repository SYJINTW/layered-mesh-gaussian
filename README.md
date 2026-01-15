# Layered Mesh-Gaussian

Official implementation for the paper "LMG: Efficient Streaming of Layered Mesh–Gaussian 3D Scenes"

> **LMG: Efficient Streaming of Layered Mesh–Gaussian 3D Scenes**<br>
[Yuan-Chun Sun](https://syjintw.github.io/) <sup>1</sup>,
[Guodong Chen](#) <sup>2</sup>,
[Sam Ziaie Kondori](#) <sup>1</sup>,
[Mallesham Dasari](#) <sup>2</sup>,
[Cheng-Hsin Hsu](https://aiins.cs.nthu.edu.tw/cheng-hsin-hsu/) <sup>1</sup> <br>
 <sup>1</sup> National Tsing Hua University, <sup>2</sup> Northeastern University<br>
**Accepted by ACM Multimedia Systems Conference 2026 (MMSys'25)** <br>

| [Project](https://aiins-nthu.github.io/LMG/) | [Paper (Coming Soon)](#) |
## Introduction

Layered Mesh-Gaussian (LMG) is a research implementation built upon the official ["GaMeS: Mesh-Based Adapting and Modification of Gaussian Splatting"](https://arxiv.org/abs/2402.01459).  
This project extends the original [official codebase](https://waczjoan.github.io/gaussian-mesh-splatting/) with additional utilities and experimental workflows for mesh-driven Gaussian Splatting and hybrid 3D representation rendering.

## Installation

See [INSTALL.md](doc/INSTALL.md) for instructions environment setup.

## Setup Dataset

See [DATASET.md](doc/DATASET.md) for instructions dataset.

## Getting Started

### Quick Start: Full Pipeline

To run the complete pipeline (Warmup → Training → Rendering → Metrics) with the default scene, policy, and budget, run the following command:

```bash
bash exp_sample.sh
```

> **Note:** You can configure the specific experiment settings (Scene, Policy, Budget) by editing the variables defined at the top of `exp_sample.sh`.

### Output Structure

The pipeline generates artifacts in the `./output` and `./log` directories. Below is the breakdown of where files are stored, using `{EXP_NAME}`, `{SCENE_NAME}`, and `{CONFIG}` (composed of `policy_budget_occlusion`) as placeholders.

#### 1. Trained 3D Gaussians (Enhancement Layer)

The trained model files are saved as `.ply` files.

* **Path Template:**
```text
./output/{EXP_NAME}/{SCENE_NAME}/{CONFIG}/point_cloud/iteration_{ITER}/point_cloud.ply
```

* **Example (`exp_sample.sh`):**
```text
./output/sample_exp/hotdog/distortion_40000_occlusion/point_cloud/iteration_15000/point_cloud.ply
```


#### 2. Rendered Images

The rendered images from the LMG model for specific iterations.

* **Path Template:**
```text
./output/{EXP_NAME}/{SCENE_NAME}/{CONFIG}/test/ours_{ITER}/renders_gs_mesh
```


* **Example (`exp_sample.sh`):**
```text
./output/sample_exp/hotdog/distortion_40000_occlusion/test/ours_7000/renders_gs_mesh
./output/sample_exp/hotdog/distortion_40000_occlusion/test/ours_15000/renders_gs_mesh
```


#### 3. Quantitative Metrics

Visual quality metrics are saved as JSON files containing per-view and aggregated results.

* **Path Template:**
```text
./output/{EXP_NAME}/{SCENE_NAME}/{CONFIG}/per_view_gs_mesh.json
./output/{EXP_NAME}/{SCENE_NAME}/{CONFIG}/results_gs_mesh.json
```

* **Example (`exp_sample.sh`):**
```text
./output/sample_exp/hotdog/distortion_40000_occlusion/per_view_gs_mesh.json
./output/sample_exp/hotdog/distortion_40000_occlusion/results_gs_mesh.json
```

#### 4. Logs

Execution logs for the pipeline are stored in the separate log directory.

* **Path Template:**
```text
./log/{EXP_NAME}/{SCENE_NAME}/log_pipeline_{CONFIG}.log
```

* **Example (`exp_sample.sh`):**
```text
./log/sample_exp/hotdog/log_pipeline_distortion_40000_occlusion.log
```

# ==== NOT YET ====
## Quick Start: Full Pipeline with Debug Script

For a complete pipeline (warmup → training → rendering → metrics) with a specific scene, policy, and budget:

```bash
bash debug_pipeline.sh
```

Configure the script by editing these variables at the top:

```bash
export CUDA_VISIBLE_DEVICES=2

UNIT_BUDGET=1.5                    # Budget proportional to number of triangles
POLICY="planarity"                 # Options: planarity, area, distortion, uniform, random
DATASET_DIR="/path/to/dataset"
SCENE_NAME="bicycle"
MESH_TYPE="colmap"                 # Options: "sugar" or "colmap"
MESH_FILE="/path/to/mesh.ply"      # .ply for colmap, .obj for sugar
RESOLUTION=""                      # Or "--resolution 4" for faster debugging
IS_WHITE_BG="-w"                   # Or empty string for black background
```

## Batch Experiments: Multiple Budgets and Policies

For running experiments with multiple budgets, policies, and occlusion settings:

```bash
bash 1113_pipeline.sh
```

Configure at the top of the script:

```bash
export CUDA_VISIBLE_DEVICES=3

DATASET_DIR="/path/to/dataset"
SAVE_DIR="/path/to/output"

# Splat budgets to test (0 = mesh only, no splats)
BUDGETS=( 1 3000000 2000000 1000000 524288 262144 131072 )

# Allocation policies to test
POLICIES=("area" "distortion" "planarity" "uniform" "random")

# Test with and without occlusion
WHETHER_OCCLUSION=("--occlusion" "")

ITERATION="5000"
EXP_NAME="1113_downsampled"

SCENE_NAME="bicycle"
MESH_TYPE="colmap"
MESH_FILE="/path/to/mesh.ply"
```

This script automatically:

- Runs warmup, training, rendering, and metrics for each combination
- Logs timing for each stage
- Tracks failed experiments
- Generates a timing summary TSV file: `output/EXPERIMENT_NAME/SCENE_NAME/pipeline_timing_summary.tsv`
- Saves results of metrics as JSON files to `output/EXPERIMENT_NAME/SCENE_NAME/for_plot/`

## Individual Pipeline Stages

### Step 0: Warmup (Optional - Pre-render Mesh Backgrounds, Precalculate Allocation Policies)

Warmup pre-renders mesh backgrounds and depth maps for all training cameras. This is optional and can speed up initialization, but is not required.

```bash
CUDA_VISIBLE_DEVICES=2 python train.py --eval \
  --warmup_only \
  -s /path/to/dataset \
  -m output/exp_name \
  --texture_obj_path /path/to/mesh.ply \
  --mesh_type colmap \
  --gs_type gs_mesh \
  --debugging \
  --debug_freq 100 \
  --total_splats 1000000 \
  --alloc_policy planarity \
  --precaptured_mesh_img_path /path/to/mesh/dir \
  -w --iteration 10
```

**What warmup does:**

- Pre-renders mesh backgrounds and depth maps for all training cameras
- Saves to `precaptured_mesh_img_path/mesh_texture/` and `mesh_depth/` directories
- Generates or validates policy allocation file (.npy)
- Exits after completion, does not enter training loop
- Optional for both `--occlusion` and non-occlusion modes

### Step 1: Training

```bash
CUDA_VISIBLE_DEVICES=2 python train.py --eval \
  -s /path/to/dataset \
  -m output/exp_name \
  --texture_obj_path /path/to/mesh.ply \
  --mesh_type colmap \
  --gs_type gs_mesh \
  --debugging \
  --debug_freq 100 \
  --occlusion \
  --total_splats 1000000 \
  --alloc_policy planarity \
  --policy_path output/exp_name/policy.npy \
  --precaptured_mesh_img_path /path/to/mesh/images \
  -w --iteration 5000
```

**Alternative: Use `--budget_per_tri` instead of `--total_splats`:**

```bash
python train.py --eval \
  -s /path/to/dataset \
  -m output/exp_name \
  --texture_obj_path /path/to/mesh.ply \
  --mesh_type colmap \
  --gs_type gs_mesh \
  --budget_per_tri 1.5 \
  --alloc_policy planarity \
  -w --iteration 5000
```

### Step 2: Rendering

```bash
python render_mesh_splat.py \
  -m output/exp_name \
  --gs_type gs_mesh \
  --skip_train \
  --occlusion \
  --total_splats 1000000 \
  --alloc_policy planarity \
  --texture_obj_path /path/to/mesh.ply \
  --mesh_type colmap \
  --policy_path output/exp_name/policy.npy
```

### Step 3: Evaluation Metrics

```bash
python metrics.py \
  -m output/exp_name \
  --gs_type gs_mesh
```

## Key Command-Line Arguments

### Dataset & Mesh

| Argument                 | Description                           | Type           |
| ------------------------ | ------------------------------------- | -------------- |
| `-s, --source_path`      | Path to dataset directory             | str (required) |
| `-m, --model_path`       | Output model directory                | str (required) |
| `--texture_obj_path`     | Path to mesh file (.obj or .ply)      | str            |
| `--mesh_type`            | Mesh source type: `sugar` or `colmap` | str            |
| `-w, --white_background` | Use white background (not black)      | flag           |

### Mesh-Splat Configuration

| Argument           | Description                                                    | Default   |
| ------------------ | -------------------------------------------------------------- | --------- |
| `--gs_type`        | Renderer type: `gs`, `gs_flat`, or `gs_mesh`                   | `gs_mesh` |
| `--total_splats`   | Total number of splats for entire scene, int                   | None      |
| `--budget_per_tri` | Splats per triangle (multiplier), float                        | 1.0       |
| `--alloc_policy`   | Policy: `uniform`, `random`, `area`, `planarity`, `distortion` | `area`    |

| Argument                      | Description                                                      | Default  |
| ----------------------------- | ---------------------------------------------------------------- | -------- |
| `--occlusion`                 | Enable occlusion-aware rendering                                 | Disabled |
| `--policy_path`               | Path to pre-computed policy `.npy` file                          | None     |
| `--precaptured_mesh_img_path` | Dir with `mesh_texture/` and `mesh_depth/` subdirs (from warmup) | None     |

### Training Configuration

| Argument        | Description                       | Default |
| --------------- | --------------------------------- | ------- |
| `--iteration`   | Number of training iterations     | 1000    |
| `--eval`        | Enable evaluation during training | False   |
| `--warmup_only` | Only run warmup stage and exit    | False   |
|                 |                                   |         |

### Debugging

| Argument       | Description                               | Default |
| -------------- | ----------------------------------------- | ------- |
| `--debugging`  | Save debug visualizations during training | False   |
| `--debug_freq` | Frequency of saving debug images          | 1       |

## Comparison: Different Rendering Types

### Original Gaussian Splatting (pure GS)

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --eval \
  -s /path/to/dataset \
  -m output/gs_only \
  --gs_type gs \
  -w --iteration 5000 \
  --debugging --debug_freq 100
```

```bash
python render_gs.py -m output/gs_only --gs_type gs --skip_train
python metrics.py -m output/gs_only --gs_type gs
```

### Mesh-Splat WITH Occlusion

```bash
# Step 1: Training (warmup optional)
CUDA_VISIBLE_DEVICES=1 python train.py --eval \
  -s /path/to/dataset \
  -m output/meshsplat_with_occ \
  --texture_obj_path /path/to/mesh.ply \
  --mesh_type colmap \
  --gs_type gs_mesh \
  --occlusion \
  --budget_per_tri 1.5 \
  --alloc_policy planarity \
  -w --iteration 5000

# Step 2: Rendering
python render_mesh_splat.py \
  -m output/meshsplat_with_occ \
  --gs_type gs_mesh \
  --skip_train \
  --occlusion \
  --texture_obj_path /path/to/mesh.ply \
  --mesh_type colmap
```

### Mesh-Splat WITHOUT Occlusion

```bash
# Step 1: Training
CUDA_VISIBLE_DEVICES=2 python train.py --eval \
  -s /path/to/dataset \
  -m output/meshsplat_no_occ \
  --texture_obj_path /path/to/mesh.ply \
  --mesh_type colmap \
  --gs_type gs_mesh \
  --budget_per_tri 1.5 \
  --alloc_policy planarity \
  -w --iteration 5000

# Step 2: Rendering
python render_mesh_splat.py \
  -m output/meshsplat_no_occ \
  --gs_type gs_mesh \
  --skip_train \
  --texture_obj_path /path/to/mesh.ply \
  --mesh_type colmap
```

## Mesh Format Notes

**SuGaR meshes (.obj):**

```bash
--mesh_type sugar --texture_obj_path /path/to/mesh.obj
```

**Colmap meshes (.ply):**

```bash
--mesh_type colmap --texture_obj_path /path/to/mesh.ply
```

## Output Structure

```
output/
├── EXPERIMENT_NAME/
│   ├── SCENE_NAME/
│   │   ├── policy_1.0_occlusion/
│   │   │   ├── log_pipeline_*.log
│   │   │   ├── policy.npy
│   │   │   ├── results_gs_mesh.json
│   │   │   └── ...
│   │   ├── pipeline_timing_summary.tsv
│   │   ├── failed_experiments.log
│   │   └── for_plot/
│   │       └── *.json (results for plotting)
│   └── log/
│       └── *.log
```

## Notes

- **Warmup is optional:** Pre-renders mesh backgrounds for faster initialization, but not required
- **Budget modes:** Use either `--total_splats` for absolute budget or `--budget_per_tri` for relative budget
- **Policy files:** Generated during training or warmup, can be reused across experiments
- **Precaptured images:** Optional. If not provided, will be computed on-the-fly during training
- **Debugging:** Enable `--debugging` and set `--debug_freq` to inspect intermediate visualizations
- **Occlusion flag:** Works independently - use with or without warmup as needed

## Citation

If you find this repository/work helpful in your research, welcome to cite these papers and give a ⭐.

```
@inproceedings{sun2026lmg,
  title={LMG: Efficient Streaming of Layered Mesh–Gaussian 3D Scenes},
  author={Sun, Yuan-Chun and Chen, Guodong and Kondori, Sam Ziaie and Dasari, Mallesham and Hsu, Cheng-Hsin},
  booktitle={Proceedings of the 17th ACM Multimedia Systems Conference},
  year={2026}
}
```

Last update: Jan 16, 2026
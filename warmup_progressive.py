#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from renderer.mesh_splat_renderer import render, network_gui
import sys
from scene import Scene
from scene import SceneSimple
from games import (
    optimizationParamTypeCallbacks,
    gaussianModel
)

from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torchvision.transforms.functional as TF

import numpy as np
from pathlib import Path


import renderer.mesh_loader.mesh_loader as mesh_loader

# >>>> [YC] add
from games.mesh_splatting.scene.dataset_readers import my_get_num_splats_per_triangle, create_init_point_cloud
# <<<< [YC] add

# [good to have] loss-informed stop criteria
LOSS_CONVG_THRESH = 0.01


def get_tri_avg_colors(mesh_scene):
    vertex_colors = mesh_scene.visual.vertex_colors[:, :3]
    faces = mesh_scene.faces
    tri_vertex_colors = vertex_colors[faces]  # (n_faces, 3, 3)
    tri_avg_colors = tri_vertex_colors.mean(axis=1)
    return tri_avg_colors

def warmup(gs_type, dataset, opt, pipe, 
            testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
            debug_from, save_xyz,
            # >>>> [YC] add
            texture_obj_path, 
            debugging, debug_freq,
            occlusion,
            policy_path,
            precaptured_mesh_img_path,
            mesh_rasterizer_type="pytorch3d",
            load_iter=0,
            gs_path=None,
            alpha_path=None,
            fixed_alpha=False
            # <<<< [YC] add
            ):
    # --------------------------- Warm Up Stage -------------------------- #
    
    # first_iter = 0
    # tb_writer = prepare_output_and_logger(dataset)
    gaussians = gaussianModel[gs_type](dataset.sh_degree) # [YC] note: nothing changing here
    print("[INFO] Training() policy_path:", policy_path)
        
    print("[INFO] Scene initialization...")
    
    scene = SceneSimple(args=dataset, 
                        gaussians=gaussians, 
                        texture_obj_path=texture_obj_path, 
                        gs_path=gs_path
                        )
    
    if gs_path is None:
        num_splats_per_triangle = my_get_num_splats_per_triangle(
                dataset_path=dataset.source_path,
                mesh_scene=scene.mesh_scene,
                mesh_scene_for_render=scene.mesh_scene_for_render, 
                triangles=scene.triangles, faces=scene.faces, vertices=scene.vertices,
                viewpoint_cameras=scene.train_cameras[1], # [NOTE] 1 is resolution of images
                total_splats=dataset.total_splats,
                budgeting_policy_name=dataset.alloc_policy,
                policy_path=policy_path,
                mesh_type=dataset.mesh_type,
                pipe=None, # for using mesh+gs for allocation
                gaussians=None, # for using mesh+gs for allocation
                debugging=False,
                load_iter=load_iter,
                mesh_rasterizer_type=mesh_rasterizer_type,
                mesh_background_color=(1.0, 1.0, 1.0) if dataset.white_background else (0.0, 0.0, 0.0)
            )
    else:
        num_splats_per_triangle = my_get_num_splats_per_triangle(
                dataset_path=dataset.source_path,
                mesh_scene=scene.mesh_scene,
                mesh_scene_for_render=scene.mesh_scene_for_render, 
                triangles=scene.triangles, faces=scene.faces, vertices=scene.vertices,
                viewpoint_cameras=scene.train_cameras[1], # [NOTE] 1 is resolution of images
                total_splats=dataset.total_splats,
                budgeting_policy_name=dataset.alloc_policy,
                policy_path=policy_path,
                mesh_type=dataset.mesh_type,
                pipe=pipe, # for using mesh+gs for allocation
                gaussians=scene.gaussians, # for using mesh+gs for allocation
                debugging=False,
                load_iter=load_iter,
                mesh_rasterizer_type=mesh_rasterizer_type,
                mesh_background_color=(1.0, 1.0, 1.0) if dataset.white_background else (0.0, 0.0, 0.0)
            )
    
    import matplotlib.pyplot as plt
    unique_values, counts = np.unique(num_splats_per_triangle, return_counts=True)
    sort_indices = np.argsort(counts)[::-1]
    sorted_values = unique_values[sort_indices]
    sorted_counts = counts[sort_indices]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(v) for v in sorted_values], sorted_counts, color='skyblue', edgecolor='black')
    ax.set_title('Frequency of Splats per Triangle (Sorted by Count)')
    ax.set_xlabel('Number of Splats')
    ax.set_ylabel('Number of Triangles')
    ax.bar_label(bars)  # Add exact count values on top of each bar
    plt.tight_layout()
    plt.savefig(os.path.join(dataset.model_path, 'splats_frequency_plot.png'), dpi=300)
    plt.close()
    print(f"[INFO] num_splats_per_triangle: {num_splats_per_triangle}")
    print(f"[INFO] num_splats_per_triangle max min: {num_splats_per_triangle.max()} {num_splats_per_triangle.min()}")
    print(f"[INFO] Total number of splats allocated: {num_splats_per_triangle.sum()}")
    
    # if load_iter > 0:
    #     num_splats_per_triangle = num_splats_per_triangle+scene.gaussians.num_splats_per_triangle
    
    tri_avg_colors = get_tri_avg_colors(scene.mesh_scene) 
    
    if gs_path == None:
        # First warmup
        pcd = create_init_point_cloud(
            model_path=dataset.model_path,
            triangles=scene.triangles, faces=scene.faces, vertices=scene.vertices,
            num_splats_per_triangle=num_splats_per_triangle,
            tri_avg_colors=tri_avg_colors,
            fixed_alpha=fixed_alpha,
        )
        gaussians.create_from_pcd(pcd, scene.cameras_extent)
        scene.gaussians = gaussians
    else:
        # Second warmup (progressive warmup)
        # scene.gaussians will not be None
        foundation_gaussians = scene.gaussians
        foundation_num_splats_per_triangle = scene.gaussians.num_splats_per_triangle
        
        total_num_splats_per_triangle = num_splats_per_triangle+scene.gaussians.num_splats_per_triangle
        
        total_gaussians = gaussianModel[gs_type](dataset.sh_degree)
        pcd = create_init_point_cloud(
            model_path=dataset.model_path,
            triangles=scene.triangles, faces=scene.faces, vertices=scene.vertices,
            num_splats_per_triangle=total_num_splats_per_triangle,
            tri_avg_colors=tri_avg_colors,
            fixed_alpha=fixed_alpha,
        )
        total_gaussians.create_from_pcd(pcd, scene.cameras_extent)
        
        # Replace color and opacity
        # foundation:
        # - foundation_gaussians
        # - foundation_num_splats_per_triangle
        # total:
        # - total_gaussians
        # - total_num_splats_per_triangle
        
        # print(total_gaussians._features_dc.shape)
        # print(total_gaussians._features_dc[10:20].shape)
        # print(total_gaussians._features_rest.shape)
        # print(total_gaussians._opacity.shape)
        
        # print(total_num_splats_per_triangle.shape)
        # print(foundation_num_splats_per_triangle.shape)
        current_foundation_gs_idx = 0
        current_total_gs_idx = 0
        for tri_idx, (num_splats_total, num_splats_foundation) in enumerate(zip(total_num_splats_per_triangle, foundation_num_splats_per_triangle)):
            if num_splats_foundation == 0:
                pass
            else:
                with torch.no_grad():
                    # print(tri_idx, num_splats_foundation, current_foundation_gs_idx, current_total_gs_idx)
                    # print(foundation_gaussians._features_dc[current_foundation_gs_idx:current_foundation_gs_idx+num_splats_foundation].shape)
                    # print(total_gaussians._features_dc[current_total_gs_idx:current_total_gs_idx+num_splats_foundation].shape)
                    total_gaussians._features_dc[current_total_gs_idx:current_total_gs_idx+num_splats_foundation] = foundation_gaussians._features_dc[current_foundation_gs_idx:current_foundation_gs_idx+num_splats_foundation]
                    total_gaussians._features_rest[current_total_gs_idx:current_total_gs_idx+num_splats_foundation] = foundation_gaussians._features_rest[current_foundation_gs_idx:current_foundation_gs_idx+num_splats_foundation]
                    total_gaussians._opacity[current_total_gs_idx:current_total_gs_idx+num_splats_foundation] = foundation_gaussians._opacity[current_foundation_gs_idx:current_foundation_gs_idx+num_splats_foundation]
            current_foundation_gs_idx += num_splats_foundation
            current_total_gs_idx += num_splats_total
        
        scene.gaussians = total_gaussians

    scene.save(load_iter, warmup=True) # save the initialized scene as iteration 0

    # ---------------------- Precapture mesh backgrounds ----------------------- #
    # Render textured-mesh RGB + depth per camera once, cached for train/render.
    # Written to the dir names train_progressive / render_mesh_splat_progressive read
    # (no mesh_rasterizer_type subfolder). Skips cameras already present.
    if dataset.warmup_only:
        if not precaptured_mesh_img_path:
            raise ValueError("precaptured_mesh_img_path must be provided for warmup_only mode")

        # scene.mesh_scene is already loaded + transformed (SceneSimple.__init__) —
        # reuse it directly instead of re-reading the file from disk (colmap/milo only;
        # sugar needs pytorch3d's own UV-aware loader, no in-memory trimesh to reuse).
        if dataset.mesh_type in ("colmap", "milo"):
            if mesh_rasterizer_type == "pytorch3d":
                scene.textured_mesh = mesh_loader.to_pytorch3d_meshes(scene.mesh_scene)
            else:
                scene.textured_mesh = scene.mesh_scene
        else:
            scene.textured_mesh = mesh_loader.load_textured_mesh(dataset, texture_obj_path, mesh_rasterizer_type)

        mesh_bg_color = (1, 1, 1) if dataset.white_background else (0, 0, 0)

        def _precapture(cams, tex_dir, depth_dir):
            tex_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            for cam in tqdm(cams, desc=f"Precapturing -> {tex_dir.name}", unit="cam"):
                bg_save = tex_dir / f"{cam.image_name}.png"
                depth_save = depth_dir / f"{cam.image_name}.pt"
                if bg_save.exists() and depth_save.exists():
                    continue
                render_pkg = render(cam, gaussians, pipe,
                                    bg_color=None, bg_depth=None,
                                    textured_mesh=scene.textured_mesh,
                                    mesh_background_color=mesh_bg_color,
                                    mesh_rasterizer_type=mesh_rasterizer_type)
                TF.to_pil_image(render_pkg["bg_color"].detach().clamp(0, 1).cpu()).save(bg_save)
                torch.save(render_pkg["bg_depth"].detach().cpu(), depth_save)

        root = Path(precaptured_mesh_img_path)
        print("[INFO] Warmup: precapturing mesh backgrounds...")
        _precapture(scene.getTrainCameras(), root / "mesh_texture", root / "mesh_depth")
        _precapture(scene.getTestCameras(), root / "test_mesh_texture", root / "test_mesh_depth")
        print("[INFO] Warmup: precaptured mesh backgrounds ready.")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--gs_type', type=str, default="gs_mesh")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--meshes", nargs="+", type=str, default=[])
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000]) # not used
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 20_000, 30_000, 60_000, 90_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--save_xyz", action='store_true')
    
    # >>>> [YC] add
    parser.add_argument('--texture_obj_path', type=str, default="")
    parser.add_argument('--debugging', action='store_true')
    parser.add_argument('--debug_freq', type=int, default=1, help="Iteration of saving debugging images")
    parser.add_argument('--occlusion', action='store_true')
    parser.add_argument('--policy_path', type=str, default="", 
        help="Path to the pre-computed .npy file storing num_gs_per_tri[]. \
        When this is provided, it has higher priority than --alloc_policy; \
        otherwise, will overwrite/recompute")
    
    parser.add_argument('--precaptured_mesh_img_path', type=str, default="",
        help="path to the directory containing precaptured mesh (RGB & D) images for background. \
            should contain mesh_texture/ and mesh_depth/ sub-folders."
        ) # [NOTE] better store alongside mesh file
    # <<<< [YC] add
    
    # use either of the two to set total number of splats (bit budget, or gaussian budget for the whole scene)
    parser.add_argument("--total_splats", type=int, help="Total number of splats to allocate")
    parser.add_argument("--budget_per_tri", type=float, default=1.0, help="set the total number of splats to be this number * number of triangles")
    parser.add_argument("--alloc_policy", type=str, default="area", help="Allocation policy for splats (default: area)")
    parser.add_argument("--warmup_only", action='store_true', help="only run warmup stage and exit, no entering training loop")
    parser.add_argument('--mesh_type', type=str, default="sugar", help="textured mesh type: sugar, colmap, or others")
    
    parser.add_argument("--mesh_rasterizer_type", type=str, default="pytorch3d", 
                        help="which mesh rasterizer to use: pytorch3d or nvdiffrast") 
    
    parser.add_argument("--load_iter", type=int, default=-1, help="-1 means not loading and start from scratch")
    parser.add_argument("--gs_path", type=str, default=None, help="path to the pretrained GS model to load for warmup initialization. If provided, will use this GS for warmup initialization instead of creating from point cloud.")
    parser.add_argument("--alpha_path", type=str, default=None, help="path to the alpha map for warmup initialization.")
    parser.add_argument("--fixed_alpha", action="store_true", default=False,
                        help="Use shared static barycentric pool across all triangles (canonical init)")
    
    lp = ModelParams(parser) # LoadingParams
    args, _ = parser.parse_known_args(sys.argv[1:])
    lp.num_splats = args.num_splats
    lp.meshes = args.meshes
    lp.gs_type = args.gs_type
    
    # >>>> [Sam] add
    lp.total_splats = args.total_splats
    lp.budget_per_tri = args.budget_per_tri
    lp.alloc_policy = args.alloc_policy 
    lp.warmup_only = args.warmup_only
    lp.mesh_type = args.mesh_type.lower()
    # <<<< [Sam] add
    
    op = optimizationParamTypeCallbacks[args.gs_type](parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)

    print("torch cuda: ", torch.cuda.is_available())
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    if len(args.save_iterations) == 0:
        print("[WARN] No save iterations specified, defaulting to saving at the end of training.")
    
    # Policy will be stored under model_path
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    warmup(
        args.gs_type,
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from, args.save_xyz,
        # >>>> [YC] add
        texture_obj_path=args.texture_obj_path,
        debugging=args.debugging, debug_freq=args.debug_freq,
        occlusion=args.occlusion,
        policy_path=args.policy_path,
        precaptured_mesh_img_path=args.precaptured_mesh_img_path,
        mesh_rasterizer_type=args.mesh_rasterizer_type,
        load_iter=args.load_iter,
        gs_path=args.gs_path,
        alpha_path=args.alpha_path,
        fixed_alpha=args.fixed_alpha
        # <<<< [YC] add
    )

    # All done
    print("\n[INFO] Warmup complete.")


#! Initialization
"""
python warmup_progressive.py --eval \
--warmup_only \
-s <DATASET_ROOT>/dataset/images/hotdog \
-m ./output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_0 \
--texture_obj_path <DATASET_ROOT>/dataset/meshes/hotdog/hotdog.ply \
--mesh_type milo \
--debugging \
--gs_type lmg \
--occlusion \
--total_splats 10000 --alloc_policy distortion_progressive \
--policy_path ./output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_0/load_iter_0/distortion_progressive_10000.npy \
--precaptured_mesh_img_path <DATASET_ROOT>/dataset/meshes/hotdog \
--load_iter 0 \
--mesh_rasterizer_type nvdiffrast
"""

#! Warmup from iteration 1000
"""
python warmup_progressive.py --eval \
--warmup_only \
-s <DATASET_ROOT>/dataset/images/hotdog \
-m ./output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_1000 \
--texture_obj_path <DATASET_ROOT>/dataset/meshes/hotdog/hotdog.ply \
--mesh_type milo \
--debugging \
--gs_type lmg \
--occlusion \
--total_splats 10000 --alloc_policy distortion_progressive \
--policy_path ./output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_1000/load_iter_1000/distortion_progressive_10000.npy \
--precaptured_mesh_img_path <DATASET_ROOT>/dataset/meshes/hotdog \
--load_iter 1000 \
--gs_path ./output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_0/point_cloud/iteration_1000/point_cloud.ply \
--mesh_rasterizer_type nvdiffrast
"""

#! Warmup from iteration 2000
"""
python warmup_progressive.py --eval \
--warmup_only \
-s <DATASET_ROOT>/dataset/images/hotdog \
-m ./output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_2000 \
--texture_obj_path <DATASET_ROOT>/dataset/meshes/hotdog/hotdog.ply \
--mesh_type milo \
--debugging \
--gs_type lmg \
--occlusion \
--total_splats 10000 --alloc_policy distortion_progressive \
--policy_path ./output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_2000/load_iter_2000/distortion_progressive_10000.npy \
--precaptured_mesh_img_path <DATASET_ROOT>/dataset/meshes/hotdog \
--load_iter 2000 \
--gs_path ./output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_1000/point_cloud/iteration_2000/point_cloud.ply \
--mesh_rasterizer_type nvdiffrast
"""


#! Baseline
"""
python warmup_progressive.py --eval \
--warmup_only \
-s <DATASET_ROOT>/dataset/images/hotdog \
-m ./output/progressive/hotdog/distortion_progressive_30000_occlusion/iteration_0 \
--texture_obj_path <DATASET_ROOT>/dataset/meshes/hotdog/hotdog.ply \
--mesh_type milo \
--debugging \
--gs_type lmg \
--occlusion \
--total_splats 30000 --alloc_policy distortion_progressive \
--policy_path ./output/progressive/hotdog/distortion_progressive_30000_occlusion/iteration_0/load_iter_0/distortion_progressive_30000.npy \
--precaptured_mesh_img_path <DATASET_ROOT>/dataset/meshes/hotdog \
--load_iter 0 \
--mesh_rasterizer_type nvdiffrast
"""
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

# from pytorch3d.io import load_objs_as_meshes

from itertools import count
import json
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
# from renderer.gaussian_renderer import render, network_gui
from renderer.mesh_splat_renderer import render, network_gui
import sys
from scene import Scene
from scene import SceneSimple
from games import (
    optimizationParamTypeCallbacks,
    gaussianModel
)

from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import numpy as np
from pathlib import Path

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io import load_ply
from pytorch3d.renderer import (
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    )

import open3d as o3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)
from scene.cameras import convert_camera_from_gs_to_pytorch3d
from pytorch3d.renderer.blending import BlendParams
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

import matplotlib.pyplot as plt
import matplotlib.cm as cm


# [good to have] loss-informed stop criteria
LOSS_CONVG_THRESH = 0.01

def frozen_mask(foundation_num_splats_per_triangle, total_num_splats_per_triangle):
    idx_list = []
    current_foundation_gs_idx = 0
    current_total_gs_idx = 0
    for tri_idx, (num_splats_total, num_splats_foundation) in enumerate(zip(total_num_splats_per_triangle, foundation_num_splats_per_triangle)):
        if num_splats_foundation == 0:
            pass
        else:
            with torch.no_grad():
                idx_list.extend([i for i in range(current_total_gs_idx, current_total_gs_idx+num_splats_foundation)])
        current_foundation_gs_idx += num_splats_foundation
        current_total_gs_idx += num_splats_total
    return idx_list

def training(gs_type, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
            debug_from, save_xyz,
            # >>>> [YC] add
            texture_obj_path, 
            debugging, debug_freq,
            occlusion,
            policy_path,
            precaptured_mesh_img_path,
            mesh_rasterizer_type="pytorch3d",
            load_gs_path=None,
            start_iteration=0,
            foundation_pt_path=None
            # <<<< [YC] add
            ):
    
    # --------------------------- Warm Up Stage -------------------------- #
    
    first_iter = 0
    # tb_writer = prepare_output_and_logger(dataset)
    gaussians = gaussianModel[gs_type](dataset.sh_degree) # [YC] note: nothing changing here
    print("[INFO] Training() policy_path:", policy_path)
    
    scene = SceneSimple(args=dataset, 
                        gaussians=gaussians, 
                        texture_obj_path=texture_obj_path, 
                        gs_path=load_gs_path
                        )
    
    # if based_gs_path is not None:
    #     foundation_gs = gaussianModel[gs_type](dataset.sh_degree)
    #     foundation_gs.load_lmg_gs()
    
    #! [YC] note: main changing point is here
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if debugging:
        check_path = Path(scene.model_path) / "debugging" / "training_check"
        check_path.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Finished Warm-Up, Start Training..." )
    #  ------------------------Warm Up Done--------------------------- #
    
    # [BUG] the background fetched in this part is for network GUI debugger only 
    # (not used by us, and not used by training loop)
    # --------------------------- Load background image -------------------------- #
    background_image_path = "/mnt/data1/syjintw/NEU/dataset/hotdog/mesh_texture/r_0.png"
    img = Image.open(background_image_path).convert("RGB")
    # viewpoint_camera_height = 800
    # viewpoint_camera_width = 800
    viewpoint_camera_height = scene.getTrainCameras()[0].image_height
    viewpoint_camera_width = scene.getTrainCameras()[0].image_width
    img = img.resize((viewpoint_camera_width, viewpoint_camera_height), Image.BILINEAR) # fixed issue, should be (W, H)
    transform = T.Compose([
        T.ToTensor(),  # [0, 255] → [0.0, 1.0], shape (3, H, W)
    ])
    background = transform(img).to(torch.float32).cuda()
    
    # ----------------------------- Load depth image ----------------------------- #
    background_depth_pt_path = "/mnt/data1/syjintw/NEU/dataset/hotdog/mesh_depth/r_0.pt"
    background_depth = torch.load(background_depth_pt_path).unsqueeze(0)

    # Trick part
    if foundation_pt_path is not None:
        # foundation_pt = torch.load("/mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/output/progressive/hotdog/exp_gs_110_iteration_200/gs_100_iteration_0/point_cloud/iteration_500/model_params.pt")
        foundation_pt = torch.load(foundation_pt_path)
        # total_pt = torch.load("/mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/output/progressive/hotdog/exp_gs_110_iteration_200/gs_10_iteration_500/point_cloud/iteration_500_warmup/model_params.pt")
        frozen_idx_list = frozen_mask(foundation_pt["num_splats_per_triangle"], gaussians.num_splats_per_triangle)
        frozen_idx_list = torch.tensor(frozen_idx_list, dtype=torch.long, device="cuda")
        # print(frozen_idx_list)
        gaussians.setup_frozen_mask(frozen_idx_list)
    
    # ---------------------------------------------------------------------------- #
    #                              Start Training Loop                             #
    # ---------------------------------------------------------------------------- #
    history_iters = []
    history_loss = []
    history_l1 = []
    history_ssim = []
    
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    if gs_type == "gs_mesh" or gs_type == "lmg":
        if occlusion:
            print("[INFO] DTGS training:: using Depth+Texture+GS rasterizer with occlusion for gs_mesh")
        else:
            print("[INFO] TGS training:: using Texture+GS rasterizer for gs_mesh")
    elif gs_type == "gs":
        print("[INFO] GS training:: using original GS rasterizer for gs")
    else: 
        pass
    
    for iteration in range(first_iter, opt.iterations + 1):
        os.makedirs(f"{scene.model_path}/xyz", exist_ok=True)
        if save_xyz and (iteration % 5000 == 1 or iteration == opt.iterations):
            torch.save(gaussians.get_xyz, f"{scene.model_path}/xyz/{iteration}.pt")
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            print("[INFO] network_gui connected")
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    print("[INFO] Received custom camera for rendering")
                    # net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image = render(custom_cam, gaussians, pipe, 
                                       background, background_depth, 
                                       scaling_modifer)["render"] # [YC] add
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        rand_cam_id = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_cam_id)
        
        # ---------------------------------------------------------------------------- #
        #                                Load Background                               #
        # ---------------------------------------------------------------------------- #
        viewpoint_camera_height = viewpoint_cam.image_height
        viewpoint_camera_width = viewpoint_cam.image_width
        
        transform = T.Compose([
            T.ToTensor(),  # [0, 255] → [0.0, 1.0], shape (3, H, W)
        ])
        
        # ------------------------------ Mesh background ----------------------------- #
        if precaptured_mesh_img_path:
            cached_bg_path = Path(precaptured_mesh_img_path) / mesh_rasterizer_type / "mesh_texture" / f"{viewpoint_cam.image_name}.png"
            if cached_bg_path.exists():
                img = Image.open(cached_bg_path).convert("RGB")
                img = img.resize((viewpoint_camera_width, viewpoint_camera_height), Image.BILINEAR)  # (W, H)
                bg = transform(img).to(torch.float32).cuda()
            
        # ------------------------------ Mesh depth background ----------------------------- #
        # [TODO] perhaps prefetch everything at the start of training?
        if precaptured_mesh_img_path:
            cached_bg_depth_path = Path(precaptured_mesh_img_path) / mesh_rasterizer_type / "mesh_depth" / f"{viewpoint_cam.image_name}.pt"
            if cached_bg_depth_path.exists():
                bg_depth = torch.load(cached_bg_depth_path).unsqueeze(0).to("cuda")

        # ------------------------------ Pure background ----------------------------- #
        pure_bg_template = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        pure_bg = torch.tensor(pure_bg_template, dtype=torch.float32, device="cuda").view(3, 1, 1)
        pure_bg = pure_bg.expand(3, viewpoint_camera_height, viewpoint_camera_width) # (H, W)
        
        # --------------------- Pure depth background (all zeros) -------------------- #
        pure_bg_depth = torch.full((1, viewpoint_camera_height, viewpoint_camera_width), 0, dtype=torch.float32, device="cuda")
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # -------------------------- Rendering for training -------------------------- #
        if gs_type == "gs":
            render_pkg = render(viewpoint_cam, gaussians, pipe, 
                                bg_color=pure_bg, bg_depth=pure_bg_depth)
        elif gs_type == "gs_mesh" or gs_type == "lmg":
            if occlusion: # [YC] use occlusion diff-gaussian-rasterizer for training
                render_pkg = render(viewpoint_cam, gaussians, pipe, 
                                    bg_color=bg, bg_depth=bg_depth, 
                                    textured_mesh=scene.textured_mesh)
                
            else: # [YC] use original diff-gaussian-rasterizer for training
                render_pkg = render(viewpoint_cam, gaussians, pipe, 
                                    bg_color=bg, bg_depth=pure_bg_depth, # [YC] no occlusion handling, use pure_bg_depth
                                    textured_mesh=scene.textured_mesh)
                
                
        image = render_pkg["render"]
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # -------------------------- Load ground truth image ------------------------- #    
        gt_image = viewpoint_cam.original_image.cuda()
         
        # -------------------------- Save debugging visualizations ------------------------- #
        if debugging:
            # ------------------- Change Tensor to PIL.Image for saving ------------------ #
            if iteration % debug_freq == 0:
                print(f"[DEBUG] Training Iteration {iteration}, viewpoint: {viewpoint_cam.image_name}")
                # ---------------------------- Ground truth image ---------------------------- #
                gt_img_to_save = gt_image.detach().clamp(0, 1).cpu()
                gt_img_pil = TF.to_pil_image(gt_img_to_save)
                gt_img_pil.save(check_path/f"{iteration+start_iteration}_gt.png")
                
                # ------------------------ Render image from training ------------------------ #
                img_to_save = image.detach().clamp(0, 1).cpu()
                img_pil = TF.to_pil_image(img_to_save)
                img_pil.save(check_path/f"{iteration+start_iteration}_training.png")
                
                # ----------------------- Background image for training ---------------------- #
                img_to_save = render_pkg["bg_color"].detach().clamp(0, 1).cpu()
                img_pil = TF.to_pil_image(img_to_save)
                img_pil.save(check_path/f"{iteration+start_iteration}_training_mesh_bg.png")
                
                if gs_type == "gs_mesh" or gs_type == "lmg":
                    # ------------- Render mesh background and depth background ------------- #
                    # [1, 1, 1]
                    render_mesh_with_depth = render(viewpoint_cam, gaussians, pipe, 
                                                    bg_color=bg, bg_depth=bg_depth,
                                                    textured_mesh=scene.textured_mesh)
                    _image = render_mesh_with_depth["render"]

                    img_to_save = _image.detach().clamp(0, 1).cpu()
                    img_pil = TF.to_pil_image(img_to_save)
                    img_pil.save(check_path/f"{iteration+start_iteration}_gs_mesh_with_depth.png")
                
                    # ------------- Render mesh background and fake depth background ------------- #
                    # [0, 1, 1]
                    render_mesh_wo_depth = render(viewpoint_cam, gaussians, pipe, 
                                                    bg_color=bg, bg_depth=pure_bg_depth,
                                                    textured_mesh=scene.textured_mesh)
                    _image = render_mesh_wo_depth["render"]

                    img_to_save = _image.detach().clamp(0, 1).cpu()
                    img_pil = TF.to_pil_image(img_to_save)
                    img_pil.save(check_path/f"{iteration+start_iteration}_gs_mesh_wo_depth.png")

                    # ------------- Render pure background and mesh depth background ------------- #
                    # [1, 0, 1]
                    render_pure_with_depth = render(viewpoint_cam, gaussians, pipe, 
                                                    bg_color=pure_bg, bg_depth=bg_depth,
                                                    textured_mesh=scene.textured_mesh)
                    _image = render_pure_with_depth["render"]
                    
                    img_to_save = _image.detach().clamp(0, 1).cpu()
                    img_pil = TF.to_pil_image(img_to_save)
                    img_pil.save(check_path/f"{iteration+start_iteration}_gs_pure_with_depth.png")
                
                    # ------------- Render pure background and fake depth background ------------- #
                    # [1, 1, 1]
                    render_pure_wo_depth = render(viewpoint_cam, gaussians, pipe, 
                                                bg_color=pure_bg, bg_depth=pure_bg_depth,
                                                textured_mesh=None)
                    _image = render_pure_wo_depth["render"]
                    
                    img_to_save = _image.detach().clamp(0, 1).cpu()
                    img_pil = TF.to_pil_image(img_to_save)
                    img_pil.save(check_path/f"{iteration+start_iteration}_gs_pure_wo_depth.png")
            
        # Compute loss and backpropagate
        Ll1 = l1_loss(image, gt_image)
        current_ssim = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - current_ssim)
        
        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            history_iters.append(iteration)
            history_loss.append(loss.item())
            history_l1.append(Ll1.item())
            history_ssim.append(current_ssim.item())
                
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # [good to have] enable training report to observe loss and metrics during training
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
            #                 testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration+start_iteration)

            # Densification
            if (args.gs_type == "gs") or (args.gs_type == "gs_flat"):
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                    size_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (
                            dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
            elif args.gs_type == "gs_mesh" or args.gs_type == "lmg":
                pass
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if hasattr(gaussians, 'update_alpha'):
            gaussians.update_alpha()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()
            
    # ================= 迴圈結束，開始畫圖 ================= #
    print("[INFO] Training complete. Generating metrics plots...")

    plt.figure(figsize=(18, 5))

    # 畫 Loss 圖
    plt.subplot(1, 3, 1)
    plt.plot(history_iters, history_loss, label='Loss', color='red', alpha=0.8)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.ylim(bottom=0.0)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 畫 L1 Loss 圖
    plt.subplot(1, 3, 2)
    plt.plot(history_iters, history_l1, label='L1 Loss', color='blue', alpha=0.8)
    plt.xlabel("Iteration")
    plt.ylabel("L1 Loss")
    plt.title("Training L1 Loss")
    plt.ylim(bottom=0.0)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 畫 SSIM 圖
    plt.subplot(1, 3, 3)
    plt.plot(history_iters, history_ssim, label='SSIM', color='green', alpha=0.8)
    plt.xlabel("Iteration")
    plt.ylabel("SSIM")
    plt.title("Training SSIM")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()

    # 儲存圖表到 model_path 底下
    plot_path = f"{scene.model_path}/training_metrics_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"[INFO] Metrics plot saved to: {plot_path}")
    
    metrics_data = {
        "iteration": history_iters,
        "loss": history_loss,
        "l1": history_l1,
        "ssim": history_ssim
    }

    json_path = f"{scene.model_path}/training_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_data, f)

    print(f"[INFO] Metrics plot saved to: {plot_path}")
    print(f"[INFO] Metrics data saved to: {json_path}")
    
def load_textured_mesh(dataset, texture_obj_path: str) -> Meshes:
    """
    Load a textured 3D mesh from the given path for background rendering.
    
    This function loads mesh of SuGaR (.obj) or Colmap (.ply) format (or others, add if needed)
    and converts it to a PyTorch3D Meshes object on CUDA
    
    Args:
        dataset: Dataset configuration containing mesh_type attribute.
                Should have mesh_type in ['sugar', 'colmap', ...].
        texture_obj_path: Path to the mesh file. If empty string, raises AssertionError.
    Returns:
        Meshes: A PyTorch3D Meshes object on CUDA
    Raises:
        AssertionError: If texture_obj_path is empty or mesh type is unsupported.
        AssertionError: If file extension doesn't match expected format.
    """
    
    assert texture_obj_path != "", "[ERROR] texture_obj_path cannot be empty"
    textured_mesh = None
    mesh_type = dataset.mesh_type
    if texture_obj_path != "":
        print("[INFO] Loading textured mesh for background rendering...")
        
        if mesh_type == "sugar": # From SuGaR
            assert texture_obj_path.lower().endswith(".obj"), "[ERROR] SuGaR mesh should be .obj file!"
            textured_mesh = load_objs_as_meshes([texture_obj_path]).to("cuda")
             
        elif mesh_type == "colmap" or mesh_type == "milo": 
            # From Colmap, download from https://nerfbaselines.github.io/
            assert texture_obj_path.lower().endswith(".ply"), "[ERROR] Colmap mesh should be .ply file!"
            mesh_tm = trimesh.load(texture_obj_path, force='mesh', process=False)
            verts = torch.tensor(mesh_tm.vertices, dtype=torch.float32)
            faces = torch.tensor(mesh_tm.faces, dtype=torch.int64)
            colors = torch.tensor(mesh_tm.visual.vertex_colors[:, :3], dtype=torch.float32) / 255.0
            
            # Combine into a textured mesh
            textured_mesh = Meshes(
                verts=[verts],
                faces=[faces],
                textures=TexturesVertex(verts_features=[colors])
            ).to("cuda")
        else:
            print("[ERROR] Unknown/Unsupported mesh type!")        
            
    assert textured_mesh is not None, "[ERROR] Textured mesh is not loaded properly!"
    
    return textured_mesh

def load_textured_mesh_for_nvdiffrast(dataset, texture_obj_path: str) -> Meshes:
    return trimesh.load(texture_obj_path, force='mesh')

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
    
    parser.add_argument("--based_gs_path", type=str, default=None, help="")
    parser.add_argument("--load_gs_path", type=str, default=None, help="path to the pretrained GS model to load for training initialization")
    parser.add_argument("--start_iteration", type=int, default=0, help="iteration number to start training from, used together with --load_gs_path")
    parser.add_argument("--foundation_pt_path", type=str, help="path to the foundation model's .pt file, used for computing frozen mask for progressive training")
    
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
    
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
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
        load_gs_path=args.load_gs_path,
        start_iteration=args.start_iteration,
        foundation_pt_path=args.foundation_pt_path
        # <<<< [YC] add
    )

    # All done
    print("\n[INFO] Training complete.")


#! First training
"""
python train_progressive.py --eval \
-s /mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/dataset/images/hotdog \
-m ./output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_0 \
--texture_obj_path /mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/dataset/meshes/hotdog/hotdog.ply \
--mesh_type milo \
--debugging --debug_freq 100 \
--occlusion \
--total_splats 10000 \
--alloc_policy distortion_progressive \
--policy_path ./output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_0/load_iter_0/distortion_progressive_10000.npy \
--precaptured_mesh_img_path /mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/dataset/meshes/hotdog \
--gs_type lmg \
--iteration 1000 --save_iterations 500 \
--mesh_rasterizer_type nvdiffrast \
--load_gs_path ./output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_0/point_cloud/iteration_0_warmup/point_cloud.ply
"""

# 1000 - 2000
"""
python train_progressive.py --eval \
-s /mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/dataset/images/hotdog \
-m ./output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_1000 \
--texture_obj_path /mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/dataset/meshes/hotdog/hotdog.ply \
--mesh_type milo \
--debugging --debug_freq 100 \
--occlusion \
--total_splats 10000 \
--alloc_policy distortion_progressive \
--policy_path ./output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_1000/load_iter_1000/distortion_progressive_10000.npy \
--precaptured_mesh_img_path /mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/dataset/meshes/hotdog \
--gs_type lmg \
--iteration 1000 --save_iterations 500 \
--mesh_rasterizer_type nvdiffrast \
--load_gs_path ./output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_1000/point_cloud/iteration_1000_warmup/point_cloud.ply \
--start_iteration 1000
"""

# 2000 - 3000
"""
python train_progressive.py --eval \
-s /mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/dataset/images/hotdog \
-m ./output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_2000 \
--texture_obj_path /mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/dataset/meshes/hotdog/hotdog.ply \
--mesh_type milo \
--debugging --debug_freq 100 \
--occlusion \
--total_splats 10000 \
--alloc_policy distortion_progressive \
--policy_path ./output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_2000/load_iter_2000/distortion_progressive_10000.npy \
--precaptured_mesh_img_path /mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/dataset/meshes/hotdog \
--gs_type lmg \
--iteration 1000 --save_iterations 500 \
--mesh_rasterizer_type nvdiffrast \
--load_gs_path ./output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_2000/point_cloud/iteration_2000_warmup/point_cloud.ply \
--start_iteration 2000
"""


#! Baseline
"""
python train_progressive.py --eval \
-s /mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/dataset/images/hotdog \
-m ./output/progressive/hotdog/distortion_progressive_30000_occlusion/iteration_0 \
--texture_obj_path /mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/dataset/meshes/hotdog/hotdog.ply \
--mesh_type milo \
--debugging --debug_freq 100 \
--occlusion \
--total_splats 30000 \
--alloc_policy distortion_progressive \
--policy_path ./output/progressive/hotdog/distortion_progressive_30000_occlusion/iteration_0/load_iter_0/distortion_progressive_30000.npy \
--precaptured_mesh_img_path /mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/dataset/meshes/hotdog \
--gs_type lmg \
--iteration 3000 --save_iterations 1000 2000 \
--mesh_rasterizer_type nvdiffrast \
--load_gs_path ./output/progressive/hotdog/distortion_progressive_30000_occlusion/iteration_0/point_cloud/iteration_0_warmup/point_cloud.ply
"""
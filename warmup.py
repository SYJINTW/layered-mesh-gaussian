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

import torchvision.transforms.functional as TF

import numpy as np
from pathlib import Path


import renderer.mesh_loader.mesh_loader_pytorch3d as mesh_loader_pytorch3d
import renderer.mesh_loader.mesh_loader_nvdiffrast as mesh_loader_nvdiffrast

# [good to have] loss-informed stop criteria
LOSS_CONVG_THRESH = 0.01


def warmup(gs_type, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
            debug_from, save_xyz,
            # >>>> [YC] add
            texture_obj_path, 
            debugging, debug_freq,
            occlusion,
            policy_path,
            precaptured_mesh_img_path,
            mesh_rasterizer_type="pytorch3d"
            # <<<< [YC] add
            ):
    
    # --------------------------- Warm Up Stage -------------------------- #
    
    # first_iter = 0
    # tb_writer = prepare_output_and_logger(dataset)
    gaussians = gaussianModel[gs_type](dataset.sh_degree) # [YC] note: nothing changing here
    print("[INFO] Training() policy_path:", policy_path)
        
    # >>>> [YC] add: if there is textured mesh, load it here (before training loop)
    if gs_type == "gs_mesh":
        # [TODO] Tricky part, but it is correct
        textured_mesh = mesh_loader_pytorch3d.load_textured_mesh_for_pytorch3d(dataset, texture_obj_path)
        # if mesh_rasterizer_type == "pytorch3d":
        #     textured_mesh = mesh_loader_pytorch3d.load_textured_mesh_for_pytorch3d(dataset, texture_obj_path)
        # elif mesh_rasterizer_type == "nvdiffrast":
        #     textured_mesh = mesh_loader_nvdiffrast.load_textured_mesh_for_nvdiffrast(dataset, texture_obj_path)
    else:
        textured_mesh = None
    # [DONE] pass the textured mesh, to Scene, Policy, renderer and such.
    # because, why pass the path when its already loaded right here?
    # <<<< [YC] add
    
    #! [YC] note: main changing point is here
    
    print("[DEBUG] going into Scene initialization...")
    
    # [TODO] need to update the efficient of type of textured_mesh while using different rasterizers
    scene = Scene(dataset, gaussians, policy_path=policy_path, texture_obj_path=texture_obj_path, textured_mesh=textured_mesh)
    gaussians.training_setup(opt)
    
    # [TODO] Tricky part, but it is correct
    if gs_type == "gs_mesh":
        if mesh_rasterizer_type == "pytorch3d":
            scene.textured_mesh = mesh_loader_pytorch3d.load_textured_mesh_for_pytorch3d(dataset, texture_obj_path)
        elif mesh_rasterizer_type == "nvdiffrast":
            scene.textured_mesh = mesh_loader_nvdiffrast.load_textured_mesh_for_nvdiffrast(dataset, texture_obj_path)
    else:
        scene.textured_mesh = None
        
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if debugging:
        print("[DEBUG] [INFO] Debugging mode is on.")
        check_path = Path(scene.model_path)/"debugging"/"training_check"
        check_path.mkdir(parents=True, exist_ok=True)
    
    if dataset.warmup_only:
        if not precaptured_mesh_img_path:
            raise ValueError("precaptured_mesh_img_path must be provided for warmup_only mode")
        
        # ------------------------------ Training Camera ----------------------------- #
        # Precapture mesh_bg and mesh_bg_depth in warmup stage
        precaptured_bg_dir = Path(precaptured_mesh_img_path) / mesh_rasterizer_type /f"mesh_texture"
        precaptured_depth_dir = Path(precaptured_mesh_img_path) / mesh_rasterizer_type / f"mesh_depth"
        
        # Ensure directories exist
        precaptured_bg_dir.mkdir(parents=True, exist_ok=True)
        precaptured_depth_dir.mkdir(parents=True, exist_ok=True)
        
        print("[INFO] Warmup stage: Generating precaptured mesh background and depth images...")
        
        for cam in tqdm(scene.getTrainCameras(), desc="Precapturing training backgrounds", unit="camera"):
            # Generate file paths
            bg_save_path = precaptured_bg_dir / f"{cam.image_name}.png"
            depth_save_path = precaptured_depth_dir / f"{cam.image_name}.pt"
            
            # Skip if already exists
            if bg_save_path.exists() and depth_save_path.exists():
                print(f"\t[INFO] Skipping {cam.image_name}, already exists.")
                continue
            
            # Render background and depth
            bg_color = (1,1,1) if dataset.white_background else (0,0,0)
            render_pkg = render(cam, gaussians, pipe, 
                                bg_color=None, bg_depth=None, 
                                textured_mesh=scene.textured_mesh,
                                mesh_background_color=bg_color,
                                mesh_rasterizer_type=mesh_rasterizer_type
                                )
            
            # Save background image
            bg_image = render_pkg["bg_color"].detach().clamp(0, 1).cpu()
            bg_image_pil = TF.to_pil_image(bg_image)
            bg_image_pil.save(bg_save_path)
            
            # Save depth image
            bg_depth = render_pkg["bg_depth"].detach().cpu()
            torch.save(bg_depth, depth_save_path)
            
            # print(f"[INFO] Saved precaptured results for [training] {cam.image_name}")
        
        # ------------------------------- Testing Camera ------------------------------ #
        precaptured_test_bg_dir = Path(precaptured_mesh_img_path) / mesh_rasterizer_type / "test_mesh_texture"
        precaptured_test_depth_dir = Path(precaptured_mesh_img_path) / mesh_rasterizer_type / "test_mesh_depth"
        
        precaptured_test_bg_dir.mkdir(parents=True, exist_ok=True)
        precaptured_test_depth_dir.mkdir(parents=True, exist_ok=True)
        
        for cam in tqdm(scene.getTestCameras(), desc="Precapturing test backgrounds", unit="camera"):
            bg_save_path = precaptured_test_bg_dir / f"{cam.image_name}.png"
            depth_save_path = precaptured_test_depth_dir / f"{cam.image_name}.pt"
            
            # Skip if already exists
            if bg_save_path.exists() and depth_save_path.exists():
                print(f"\t[INFO] Skipping {cam.image_name}, already exists.")
                continue
            
            # [DONE] fix black background issue in precapture stage
            # didn't pass bg=[0,0,0] into the mesh_renderer_pytorch3d()
            # Render background and depth
            
            bg_color = (1,1,1) if dataset.white_background else (0,0,0)
            render_pkg = render(cam, gaussians, pipe, 
                                bg_color=None, bg_depth=None, 
                                textured_mesh=scene.textured_mesh,
                                mesh_background_color=bg_color,
                                mesh_rasterizer_type=mesh_rasterizer_type
                                )
            
            # Save background image
            bg_image = render_pkg["bg_color"].detach().clamp(0, 1).cpu()
            bg_image_pil = TF.to_pil_image(bg_image)
            bg_image_pil.save(bg_save_path)
            
            # Save depth image
            bg_depth = render_pkg["bg_depth"].detach().cpu()
            torch.save(bg_depth, depth_save_path)
            
            # print(f"[INFO] Saved precaptured results for [testing] {cam.image_name}")
          
    
def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("[INFO] Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("[INFO] Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

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
        mesh_rasterizer_type=args.mesh_rasterizer_type
        # <<<< [YC] add
    )

    # All done
    print("\n[INFO] Warmup complete.")

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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from pathlib import Path
from PIL import Image
import torchvision
import torchvision.transforms as T
from renderer.mesh_splat_renderer import render_animated
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from games import gaussianModelRender

from pytorch3d.io import load_objs_as_meshes
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from train import load_textured_mesh

import json
import time


def transform_ficus_sinus(vertices, t, idxs):
    vertices[idxs, 2] += 0.005 * torch.sin(vertices[idxs, 0] * 2 * torch.pi + t)  # sinus
    vertices[idxs, 2] += 0.005 * torch.sin(vertices[idxs, 1] * 5 * torch.pi + t)  # sinus
    return vertices


def transform_hotdog_fly(vertices, t, idxs):
    vertices_new = vertices.clone()
    f = torch.sin(t) * 0.5
    # vertices_new[:, 2] += f * vertices[:, 0] ** 2 # parabola
    vertices_new[:, 2] += 0.3 * torch.sin(vertices[:, 0] * torch.pi + t)
    # vertices_new[:, 2] += t * (vertices[:, 0] ** 2 + vertices[:, 1] ** 2) ** (1 / 2) * 0.01
    return vertices_new


def transform_ficus_pot(vertices, t, idxs):
    if t > 8 * torch.pi:
        vertices[idxs, 2] += 0.005 * torch.sin(vertices[idxs, 1] * 5 * torch.pi + t)
    else:
        vertices[idxs, 2] -= (0.005 + t) * (vertices[idxs, 0] / 10) ** 2
    return vertices


def transform_ship_sinus(vertices, t, idxs=None):
    f = torch.sin(t) * 0.5
    vertices[:, 2] += 0.05 * torch.sin(vertices[:, 0] * torch.pi + f)  # sinus
    return vertices


def make_smaller(vertices, t, idxs=None):
    vertices_new = vertices.clone()
    f = torch.sin(t) + 1
    vertices_new = f * vertices_new
    return vertices_new


def do_not_transform(vertices, t):
    return vertices


def render_set(gs_type, model_path, name, iteration, views, gaussians, pipeline, background,
               texture_obj_path: str = None,
               occlusion: bool = False,
               policy_path: str = None,
               mesh_type: str = "colmap",
               textured_mesh=None,
               precaptured_mesh_img_path: str = None,
               transform_func=None):
    """
    Render animated sequence with mesh deformation.
    """
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"renders_animated_{gs_type}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_animated")
    debug_path = os.path.join(model_path, name, "ours_{}".format(iteration), "debug_animated")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(debug_path, exist_ok=True)

    # Time sequence for animation
    t = torch.linspace(0, 10 * torch.pi, len(views))
    
    # Get initial vertices
    vertices = gaussians.vertices.clone()
    # Capture initial triangle indices to restore later
    initial_triangle_indices = gaussians.triangle_indices.clone()

    mesh_vert = textured_mesh.verts_packed().clone() if textured_mesh is not None else None
    
    # Choose indexes if you want to change only part of the mesh
    idxs = None

    # [Timing] Initialize counters
    total_anim_time = 0.0
    total_render_time = 0.0
    loop_start_time = time.time()

    for idx, view in enumerate(tqdm(views, desc="Rendering animated progress")):
        # Debug flag for saving intermediate images
        debug_flag = (idx % 10 == 0)
        
        # [Timing] Start animation timer
        t_anim_start = time.time()

        # Apply transformation to vertices
        if transform_func is not None:
            new_gs_vertices = transform_func(vertices.clone(), t[idx], idxs)
            new_mesh_vert = transform_func(mesh_vert.clone(), t[idx], idxs) if mesh_vert is not None else None
        else:
            new_gs_vertices = vertices
            new_mesh_vert = mesh_vert
            
        
        # Update textured mesh with new vertices
        
        print("[DEBUG] Updating textured mesh with new vertices...")
        current_textured_mesh = None
        if textured_mesh is not None:
            current_textured_mesh = Meshes(
                verts=[new_mesh_vert],
                faces=textured_mesh.faces_list(),
                textures=textured_mesh.textures
            )
        
        # CRITICAL FIX: Perform indexing on CPU to avoid CUDA device-side assertions
        # Move data to CPU
        new_vertices_cpu = new_gs_vertices.detach().cpu()
        faces_cpu = gaussians.faces.detach().cpu().long()
        triangle_indices_cpu = gaussians.triangle_indices.detach().cpu().long()

        # 1. Create all mesh triangles (CPU)
        # Shape: (num_mesh_faces, 3, 3)
        all_triangles_cpu = new_vertices_cpu[faces_cpu]
        
        # 2. Select only the triangles bound to Gaussians (CPU)
        # Shape: (num_gaussians, 3, 3)
        # Clamp indices to ensure safety against any mismatch
        max_face_idx = all_triangles_cpu.shape[0] - 1
        triangle_indices_cpu = torch.clamp(triangle_indices_cpu, 0, max_face_idx)
        
        # [Fix] Update the model's indices on GPU with the safe clamped values
        gaussians.triangle_indices = triangle_indices_cpu.cuda()

        selected_triangles_cpu = all_triangles_cpu[triangle_indices_cpu]
        
        # 3. Move result back to GPU
        triangles = selected_triangles_cpu.float().cuda()
        
        # Update the gaussians' vertices reference (needed for prepare_scaling_rot)
        gaussians.vertices = new_gs_vertices

        # [Timing] End animation timer
        torch.cuda.synchronize()
        t_anim_end = time.time()
        total_anim_time += (t_anim_end - t_anim_start)

        # Render
        # rendering = render_animated(idxs, triangles, view, gaussians, pipeline,
        #                           background,
        #                           textured_mesh=textured_mesh)

        if idx == 0:
            print(f"[DEBUG] all triangles shape: {all_triangles_cpu.shape}")
            print(f"[DEBUG] selected triangles shape: {triangles.shape}")
            print(f"[DEBUG] GS vertices range: {vertices.min(dim=0).values} to {vertices.max(dim=0).values}")
            print(f"[DEBUG] Mesh vertices range: {mesh_vert.min(dim=0).values} to {mesh_vert.max(dim=0).values}")
        
        # Load precaptured mesh background and depth if available
        bg = None
        bg_depth = None
        
        
        # [NOTE] don't use precaptured, need to calculate rigged positions in place
        # if precaptured_mesh_img_path:
        #     cached_bg_path = Path(precaptured_mesh_img_path) / "test_mesh_texture" / f"{view.image_name}.png"
        #     cached_bg_depth_path = Path(precaptured_mesh_img_path) / "test_mesh_depth" / f"{view.image_name}.pt"
            
        #     if cached_bg_path.exists():
        #         img = Image.open(cached_bg_path).convert("RGB")
        #         img = img.resize((view.image_width, view.image_height), Image.BILINEAR)
        #         transform = T.Compose([T.ToTensor()])
        #         bg = transform(img).to(torch.float32).cuda()
        #         if debug_flag:
        #             torchvision.utils.save_image(bg, os.path.join(debug_path, '{0:05d}_bg'.format(idx) + ".png"))
            
        #     if cached_bg_depth_path.exists():
        #         bg_depth = torch.load(cached_bg_depth_path).unsqueeze(0).to("cuda")
        
        
        pure_bg_template = background
        pure_bg = pure_bg_template.clone().detach().view(3, 1, 1)
        pure_bg = pure_bg.expand(3, view.image_height, view.image_width)
        pure_bg_depth = torch.full((1, view.image_height, view.image_width), 0, dtype=torch.float32, device="cuda")
        
        
        # [Timing] Start render timer
        t_render_start = time.time()

        # Render based on gs_type
        if gs_type == "gs":
            # Pure GS rendering without textured mesh
            
            rendering = render_animated(idxs, triangles, view, gaussians, pipeline,
                                       bg_color=pure_bg, bg_depth=pure_bg_depth)["render"]
            # print("\033[94m [INFO] AnimatedRender::GS using pure GS rasterizer\033[0m")
            
        elif gs_type == "gs_mesh":
            
            
            # [NOTE] to render base layer only
            # gaussians._opacity = torch.full_like(gaussians._opacity, -1e10) # effectively transparent gaussians
            
            if occlusion:
                rendering = render_animated(idxs, triangles, view, gaussians, pipeline,
                                           bg_color=bg, bg_depth=bg_depth,
                                           textured_mesh=current_textured_mesh,
                                           mesh_background_color=background)["render"]
                # print("\033[92m [INFO] AnimatedRender::DTGS using Depth+Texture+GS rasterizer\033[0m")
            else:
                pure_bg_depth = torch.full((1, view.image_height, view.image_width), 0, dtype=torch.float32, device="cuda")
                rendering = render_animated(idxs, triangles, view, gaussians, pipeline,
                                           bg_color=bg, bg_depth=pure_bg_depth,
                                           textured_mesh=current_textured_mesh,
                                           mesh_background_color=background)["render"]
                # print("\033[96m [INFO] AnimatedRender::TGS using Texture+GS rasterizer\033[0m")
            
            # Debug: save pure gaussian
            if debug_flag:
                _pure_bg_template = background
                _pure_bg = _pure_bg_template.clone().detach().view(3, 1, 1)
                _pure_bg = _pure_bg.expand(3, view.image_height, view.image_width)
                _pure_bg_depth = torch.full((1, view.image_height, view.image_width), 0, dtype=torch.float32, device="cuda")
                _rendering = render_animated(idxs, triangles, view, gaussians, pipeline,
                                            bg_color=_pure_bg, bg_depth=_pure_bg_depth)["render"]
                torchvision.utils.save_image(_rendering, os.path.join(debug_path, '{0:05d}_pure_gs'.format(idx) + ".png"))
        
        # [Timing] End render timer
        torch.cuda.synchronize()
        t_render_end = time.time()
        total_render_time += (t_render_end - t_render_start)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        # Save debug images
        if debug_flag:
            torchvision.utils.save_image(rendering, os.path.join(debug_path, '{0:05d}_rendering'.format(idx) + ".png"))
            # torchvision.utils.save_image(gt, os.path.join(debug_path, '{0:05d}_gt'.format(idx) + ".png"))

    # Restore the gaussians to their original state
    # This prevents the deformation from persisting to the next render_set call (e.g. train set)
    gaussians.vertices = vertices
    gaussians.triangle_indices = initial_triangle_indices

    # [Timing] Calculate and print stats
    loop_end_time = time.time()
    total_loop_time = loop_end_time - loop_start_time
    num_frames = len(views)
    fps = num_frames / total_loop_time if total_loop_time > 0 else 0
    
    print(f"\n[INFO] Timing Statistics for {name}:")
    print(f"  - Total Animation Calculation Time: {total_anim_time:.4f}s (Avg: {total_anim_time/num_frames:.4f}s/frame)")
    print(f"  - Total Rasterization Time: {total_render_time:.4f}s (Avg: {total_render_time/num_frames:.4f}s/frame)")
    print(f"  - Total Loop Time (incl. I/O): {total_loop_time:.4f}s")
    print(f"  - FPS: {fps:.2f}")


def render_sets(gs_type: str, dataset: ModelParams, iteration: int, pipeline: PipelineParams, 
                skip_train: bool, skip_test: bool,
                texture_obj_path: str = None,
                occlusion: bool = False,
                policy_path: str = None,
                precaptured_mesh_img_path: str = None,
                transform_name: str = "hotdog_fly"):
    """
    Render animated sets for train/test views.
    """
    render_timer_start = time.time()
    
    # Select transformation function
    transform_funcs = {
        "ficus_sinus": transform_ficus_sinus,
        "hotdog_fly": transform_hotdog_fly,
        "ficus_pot": transform_ficus_pot,
        "ship_sinus": transform_ship_sinus,
        "make_smaller": make_smaller,
        "none": do_not_transform
    }
    transform_func = transform_funcs.get(transform_name, transform_hotdog_fly)
    print(f"[INFO] Using transformation: {transform_name}")
    
    with torch.no_grad():
        gaussians = gaussianModelRender[gs_type](dataset.sh_degree)
        
        # Load scene once to get cameras
        textured_mesh = load_textured_mesh(dataset=dataset, texture_obj_path=texture_obj_path)
        scene = Scene(dataset, gaussians,
                     load_iteration=iteration, shuffle=False,
                     policy_path=policy_path,
                     texture_obj_path=texture_obj_path,
                     textured_mesh=textured_mesh)
        
        if hasattr(gaussians, 'update_alpha'):
            gaussians.update_alpha()
        if hasattr(gaussians, 'prepare_vertices'):
            gaussians.prepare_vertices()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()

        mesh_type = dataset.mesh_type if hasattr(dataset, 'mesh_type') else "sugar"
        print(f"[INFO] AnimatedRender:: Using mesh type: {mesh_type}")
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_kwargs = {
            'mesh_type': mesh_type,
            'texture_obj_path': texture_obj_path,
            'occlusion': occlusion,
            'policy_path': policy_path,
            'precaptured_mesh_img_path': precaptured_mesh_img_path,
            'transform_func': transform_func
        }

        if not skip_test:
            # Reload textured mesh for test to ensure independence
            test_textured_mesh = load_textured_mesh(dataset=dataset, texture_obj_path=texture_obj_path)
            render_set(gs_type, dataset.model_path, "test", scene.loaded_iter,
                      scene.getTestCameras(), gaussians, pipeline, background,
                      textured_mesh=test_textured_mesh,
                      **render_kwargs)

        if not skip_train:
            # Reload textured mesh for train to ensure independence
            train_textured_mesh = load_textured_mesh(dataset=dataset, texture_obj_path=texture_obj_path)
            render_set(gs_type, dataset.model_path, "train", scene.loaded_iter,
                      scene.getTrainCameras(), gaussians, pipeline, background,
                      textured_mesh=train_textured_mesh,
                      **render_kwargs)
    
    render_time = time.time() - render_timer_start
    print(f"[INFO] Total animated rendering time: {render_time:.2f} seconds")


if __name__ == "__main__":
    script_start_time = time.time()
    parser = ArgumentParser(description="Animated rendering script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--gs_type', type=str, default="gs_mesh")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # Mesh and rendering options
    parser.add_argument("--texture_obj_path", type=str, default=None)
    parser.add_argument("--occlusion", action="store_true")
    parser.add_argument("--policy_path", type=str, default="")
    parser.add_argument("--precaptured_mesh_img_path", type=str, default="")
    
    # Animation options
    parser.add_argument("--transform", type=str, default="hotdog_fly",
                       choices=["ficus_sinus", "hotdog_fly", "ficus_pot", "ship_sinus", "make_smaller", "none"],
                       help="Transformation to apply to mesh vertices")
    
    # Budget and allocation
    parser.add_argument("--total_splats", type=int)
    parser.add_argument("--alloc_policy", type=str, default="area")
    parser.add_argument("--budget_per_tri", type=float, default=1.0)
    parser.add_argument('--mesh_type', type=str, default="sugar")
    
    args = get_combined_args(parser)
    model.gs_type = args.gs_type
    model.num_splats = args.num_splats
    model.total_splats = args.total_splats
    model.alloc_policy = args.alloc_policy
    model.budget_per_tri = args.budget_per_tri
    model.mesh_type = args.mesh_type
    
    print("Rendering animated " + args.model_path)

    safe_state(args.quiet)

    render_sets(args.gs_type, model.extract(args), args.iteration, pipeline.extract(args),
               args.skip_train, args.skip_test,
               texture_obj_path=args.texture_obj_path,
               occlusion=args.occlusion,
               policy_path=args.policy_path,
               precaptured_mesh_img_path=args.precaptured_mesh_img_path,
               transform_name=args.transform)
    
    print(f"[INFO] Total script execution time: {time.time() - script_start_time:.2f} seconds")
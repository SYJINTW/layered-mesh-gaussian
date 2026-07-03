#
# In-memory multi-round progressive training orchestrator (Task #2, .claude/todo.md).
#
# Replaces exp_progressive_lmg.sh's bash loop (4 cold `python` process spawns per
# round: warmup_progressive.py -> train_progressive.py -> render_mesh_splat_progressive.py
# -> metrics.py) with ONE persistent process. Mesh/cameras load once; precaptured
# backgrounds prefetch once; the Gaussian model is grown and trained in place; test-set
# metrics are computed directly from render tensors (PNGs are still written for visual
# record, but nothing is read back from disk to score a round). PLY/model_params.pt are
# still written at each round boundary for checkpoint compatibility with existing tools
# (metrics.py, plot_ablation.py, probe_frozen_gradients.py all still work unmodified
# against these directories).
#
# Per-splat `round_id` (added to LMGModel by this change) replaces disk-based
# frozen_mask() comparison: frozen = round_id < current_round, no foundation_pt_path
# file needed since the foundation IS the in-memory model from the previous round.
#
# Usage mirrors exp_progressive_lmg.sh's config block, but as CLI args:
#   conda run -n lmg python train_progressive_orchestrator.py --eval \
#     -s <dataset_dir> -m <output_dir> \
#     --texture_obj_path <mesh.ply> --mesh_type milo \
#     --gs_type lmg --alloc_policy distortion_progressive \
#     --precaptured_mesh_img_path <mesh_img_dir> --occlusion \
#     --mesh_rasterizer_type nvdiffrast \
#     --rounds 4 --splats_per_round 8000 --iters_per_round 8000 \
#     --fixed_alpha --skip_lpips

import json
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from random import randint

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from games import gaussianModel, optimizationParamTypeCallbacks
from games.mesh_splatting.scene.dataset_readers import create_init_point_cloud, my_get_num_splats_per_triangle
from renderer.mesh_splat_renderer import render
from scene import SceneSimple
from utils.image_utils import psnr as psnr_fn
from utils.loss_utils import l1_loss, ssim as ssim_fn


def get_tri_avg_colors(mesh_scene):
    vertex_colors = mesh_scene.visual.vertex_colors[:, :3]
    tri_vertex_colors = vertex_colors[mesh_scene.faces]  # (n_faces, 3, 3)
    return tri_vertex_colors.mean(axis=1)


def build_bg_cache(cams, precaptured_dir, tex_subdir, depth_subdir):
    """Load every camera's precaptured mesh bg texture+depth into CPU RAM once.
    Same values as train_progressive.py's per-cam disk read; hoisted out of the
    round loop entirely (round 1..N all reuse this, not just iterations within
    one round -- the bash pipeline reloads this fresh every round's `train.py`
    invocation, we load it exactly once for the whole experiment)."""
    cache = {}
    if not precaptured_dir:
        return cache
    tf = T.Compose([T.ToTensor()])
    for cam in cams:
        tex_path = Path(precaptured_dir) / tex_subdir / f"{cam.image_name}.png"
        depth_path = Path(precaptured_dir) / depth_subdir / f"{cam.image_name}.pt"
        if tex_path.exists() and depth_path.exists():
            img = Image.open(tex_path).convert("RGB").resize(
                (cam.image_width, cam.image_height), Image.BILINEAR)
            cache[cam.image_name] = (tf(img).to(torch.float32), torch.load(depth_path))
    return cache


def grow_splats(gaussians, scene, dataset, opt, pipe, round_idx, policy_path,
                 mesh_rasterizer_type, fixed_alpha, splats_this_round):
    """In-memory equivalent of warmup_progressive.py's warmup(): compute the
    allocation policy for this round's new splats, splice them onto the existing
    (round-1..N-1) splats, and copy appearance forward for the carried-over prefix.
    No PLY/model_params.pt round-trip -- `scene.gaussians` (or `gaussians`, same
    object for round 1) already holds everything needed in memory."""
    is_first_round = (round_idx == 1)
    new_num_splats_per_triangle = my_get_num_splats_per_triangle(
        dataset_path=dataset.source_path,
        mesh_scene=scene.mesh_scene,
        mesh_scene_for_render=scene.mesh_scene_for_render,
        triangles=scene.triangles, faces=scene.faces, vertices=scene.vertices,
        viewpoint_cameras=scene.train_cameras[1],
        total_splats=splats_this_round,
        budgeting_policy_name=dataset.alloc_policy,
        policy_path=policy_path,
        mesh_type=dataset.mesh_type,
        pipe=None if is_first_round else pipe,
        gaussians=None if is_first_round else scene.gaussians,
        debugging=False,
        load_iter=0 if is_first_round else round_idx,
        mesh_rasterizer_type=mesh_rasterizer_type,
        mesh_background_color=(1.0, 1.0, 1.0) if dataset.white_background else (0.0, 0.0, 0.0),
    )
    tri_avg_colors = get_tri_avg_colors(scene.mesh_scene)

    if is_first_round:
        pcd = create_init_point_cloud(
            model_path=dataset.model_path,
            triangles=scene.triangles, faces=scene.faces, vertices=scene.vertices,
            num_splats_per_triangle=new_num_splats_per_triangle,
            tri_avg_colors=tri_avg_colors, fixed_alpha=fixed_alpha,
        )
        gaussians.create_from_pcd(pcd, scene.cameras_extent)
        gaussians.round_id = torch.ones(gaussians._xyz.shape[0], dtype=torch.long, device="cuda")
        scene.gaussians = gaussians
        return gaussians, new_num_splats_per_triangle

    foundation = scene.gaussians
    foundation_num_splats_per_triangle = foundation.num_splats_per_triangle
    total_num_splats_per_triangle = new_num_splats_per_triangle + foundation_num_splats_per_triangle

    total_gaussians = gaussianModel[dataset.gs_type](dataset.sh_degree)
    pcd = create_init_point_cloud(
        model_path=dataset.model_path,
        triangles=scene.triangles, faces=scene.faces, vertices=scene.vertices,
        num_splats_per_triangle=total_num_splats_per_triangle,
        tri_avg_colors=tri_avg_colors, fixed_alpha=fixed_alpha,
    )
    total_gaussians.create_from_pcd(pcd, scene.cameras_extent)
    total_gaussians.round_id = torch.full((total_gaussians._xyz.shape[0],), round_idx,
                                           dtype=torch.long, device="cuda")

    # Copy appearance (+ round_id provenance) forward for the carried-over prefix of
    # each triangle's slot -- same per-triangle prefix-copy contract frozen_mask()
    # (train_progressive.py) relies on, just done in-memory instead of via disk PLY.
    cur_foundation_idx = 0
    cur_total_idx = 0
    with torch.no_grad():
        for num_total, num_foundation in zip(total_num_splats_per_triangle, foundation_num_splats_per_triangle):
            if num_foundation > 0:
                f_sl = slice(cur_foundation_idx, cur_foundation_idx + num_foundation)
                t_sl = slice(cur_total_idx, cur_total_idx + num_foundation)
                total_gaussians._features_dc[t_sl] = foundation._features_dc[f_sl]
                total_gaussians._features_rest[t_sl] = foundation._features_rest[f_sl]
                total_gaussians._opacity[t_sl] = foundation._opacity[f_sl]
                total_gaussians.round_id[t_sl] = foundation.round_id[f_sl]
            cur_foundation_idx += num_foundation
            cur_total_idx += num_total

    scene.gaussians = total_gaussians
    return total_gaussians, total_num_splats_per_triangle


def train_round(gaussians, scene, opt, pipe, gs_type, iterations, start_iteration,
                 occlusion, bg_cache, round_idx, debugging, debug_freq, model_path):
    """In-memory equivalent of train_progressive.py's training loop for one round.
    Frozen mask comes from round_id (no foundation_pt_path file needed): everything
    created in a strictly earlier round is frozen for this round."""
    gaussians.training_setup(opt)  # fresh optimizer each round -- see note in module docstring
    frozen_idx = (gaussians.round_id < round_idx).nonzero(as_tuple=True)[0]
    if len(frozen_idx) > 0:
        gaussians.setup_frozen_mask(frozen_idx)

    if debugging:
        check_path = Path(model_path) / "debugging" / "training_check"
        check_path.mkdir(parents=True, exist_ok=True)

    history = {"iteration": [], "loss": [], "l1": [], "ssim": []}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    viewpoint_stack = None
    cams = scene.getTrainCameras()
    progress_bar = tqdm(range(1, iterations + 1), desc=f"Round {round_idx} training")
    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = cams.copy()
        cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        bg, bg_depth = None, None
        if cam.image_name in bg_cache:
            cbg, cbg_depth = bg_cache[cam.image_name]
            bg = cbg.to("cuda", non_blocking=True)
            bg_depth = cbg_depth.to("cuda", non_blocking=True)
        pure_bg_depth = torch.zeros((1, cam.image_height, cam.image_width),
                                     dtype=torch.float32, device="cuda")

        render_pkg = render(cam, gaussians, pipe, bg_color=bg,
                             bg_depth=bg_depth if occlusion else pure_bg_depth,
                             textured_mesh=scene.textured_mesh)
        image = render_pkg["render"]
        gt_image = cam.original_image.cuda()

        if debugging and iteration % debug_freq == 0:
            TF.to_pil_image(gt_image.detach().clamp(0, 1).cpu()).save(
                check_path / f"{iteration + start_iteration}_gt.png")
            TF.to_pil_image(image.detach().clamp(0, 1).cpu()).save(
                check_path / f"{iteration + start_iteration}_training.png")

        Ll1 = l1_loss(image, gt_image)
        cur_ssim = ssim_fn(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - cur_ssim)
        loss.backward()

        with torch.no_grad():
            history["iteration"].append(iteration)
            history["loss"].append(loss.item())
            history["l1"].append(Ll1.item())
            history["ssim"].append(cur_ssim.item())
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.5f}"})
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        gaussians.update_alpha()
        gaussians.prepare_scaling_rot()

    train_secs = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    with open(Path(model_path) / f"train_timing_round{round_idx}.json", "w") as f:
        json.dump({"train_secs": train_secs, "peak_vram_mb": peak_vram_mb, "iterations": iterations}, f)
    with open(Path(model_path) / f"training_metrics_round{round_idx}.json", "w") as f:
        json.dump(history, f)
    print(f"[INFO] Round {round_idx} train_secs={train_secs:.1f} peak_vram_mb={peak_vram_mb:.0f} "
          f"frozen={len(frozen_idx)}/{gaussians._xyz.shape[0]}")


def render_and_score_test_set(gaussians, scene, pipe, gs_type, occlusion, mesh_rasterizer_type,
                               test_bg_cache, model_path, iteration, skip_lpips=True):
    """In-memory equivalent of render_mesh_splat_progressive.py + metrics.py for the test
    set: renders directly from the in-memory `gaussians` (no PLY reload) and scores PSNR/
    SSIM straight off the render tensor (no PNG round-trip). PNGs are still written to the
    usual test/ours_N/{renders_lmg,gt}/ layout so existing tooling (metrics.py rerun,
    visual inspection) keeps working unmodified."""
    render_dir = Path(model_path) / "test" / f"ours_{iteration}" / f"renders_{gs_type}"
    gt_dir = Path(model_path) / "test" / f"ours_{iteration}" / "gt"
    render_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    cams = scene.getTestCameras()
    ssims, psnrs = [], []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for idx, cam in enumerate(tqdm(cams, desc=f"Round render+score @ iter {iteration}")):
            bg, bg_depth = None, None
            if cam.image_name in test_bg_cache:
                cbg, cbg_depth = test_bg_cache[cam.image_name]
                bg = cbg.to("cuda", non_blocking=True)
                bg_depth = cbg_depth.to("cuda", non_blocking=True)
            pure_bg_depth = torch.zeros((1, cam.image_height, cam.image_width),
                                         dtype=torch.float32, device="cuda")
            render_pkg = render(cam, gaussians, pipe, bg_color=bg,
                                 bg_depth=bg_depth if occlusion else pure_bg_depth,
                                 textured_mesh=scene.textured_mesh,
                                 mesh_rasterizer_type=mesh_rasterizer_type)
            image = render_pkg["render"]
            gt_image = cam.original_image[0:3].cuda()

            ssims.append(ssim_fn(image.unsqueeze(0), gt_image.unsqueeze(0)).item())
            psnrs.append(psnr_fn(image.unsqueeze(0), gt_image.unsqueeze(0)).item())

            TF.to_pil_image(image.detach().clamp(0, 1).cpu()).save(render_dir / f"{idx:05d}.png")
            TF.to_pil_image(gt_image.detach().clamp(0, 1).cpu()).save(gt_dir / f"{idx:05d}.png")

    render_secs = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    results = {
        f"ours_{iteration}": {
            "SSIM": float(np.mean(ssims)), "PSNR": float(np.mean(psnrs)), "LPIPS": -1.0,
            "n_views": len(cams), "iteration": iteration,
            "render_secs": render_secs, "render_peak_vram_mb": peak_vram_mb,
        }
    }
    with open(Path(model_path) / "results_lmg.json", "w") as f:
        json.dump(results, f, indent=True)
    print(f"[INFO] Round @ iter {iteration}: PSNR={np.mean(psnrs):.3f} SSIM={np.mean(ssims):.4f}")
    return results


def save_checkpoint(gaussians, model_path, iteration):
    point_cloud_path = Path(model_path) / "point_cloud" / f"iteration_{iteration}"
    point_cloud_path.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(point_cloud_path / "point_cloud.ply"))


def orchestrate(dataset, opt, pipe, texture_obj_path, occlusion, precaptured_mesh_img_path,
                 mesh_rasterizer_type, fixed_alpha, rounds, splats_per_round, iters_per_round,
                 debugging, debug_freq):
    model_path = dataset.model_path
    gs_type = dataset.gs_type
    gaussians = gaussianModel[gs_type](dataset.sh_degree)

    print("[INFO] Loading scene (mesh + cameras) once for the whole experiment...")
    scene = SceneSimple(args=dataset, gaussians=gaussians,
                         texture_obj_path=texture_obj_path, gs_path=None)

    print("[INFO] Prefetching precaptured mesh backgrounds once for the whole experiment "
          "(bash pipeline reloads these from disk every round's train.py invocation)...")
    train_bg_cache = build_bg_cache(scene.getTrainCameras(), precaptured_mesh_img_path,
                                     "mesh_texture", "mesh_depth")
    test_bg_cache = build_bg_cache(scene.getTestCameras(), precaptured_mesh_img_path,
                                    "test_mesh_texture", "test_mesh_depth")

    wall_t0 = time.time()
    current_iteration = 0
    for round_idx in range(1, rounds + 1):
        round_dir = Path(model_path) / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        policy_path = str(round_dir / f"distortion_progressive_{splats_per_round}.npy")

        print(f"\n[INFO] ===== Round {round_idx}/{rounds}: grow =====")
        gaussians, num_splats_per_triangle = grow_splats(
            gaussians, scene, dataset, opt, pipe, round_idx, policy_path,
            mesh_rasterizer_type, fixed_alpha, splats_per_round)
        print(f"[INFO] Round {round_idx}: {gaussians._xyz.shape[0]} total splats")

        print(f"[INFO] ===== Round {round_idx}/{rounds}: train =====")
        train_round(gaussians, scene, opt, pipe, gs_type, iters_per_round, current_iteration,
                    occlusion, train_bg_cache, round_idx, debugging, debug_freq, model_path)
        current_iteration += iters_per_round

        print(f"[INFO] ===== Round {round_idx}/{rounds}: checkpoint =====")
        save_checkpoint(gaussians, model_path, current_iteration)

        print(f"[INFO] ===== Round {round_idx}/{rounds}: render + score test set =====")
        render_and_score_test_set(gaussians, scene, pipe, gs_type, occlusion, mesh_rasterizer_type,
                                   test_bg_cache, model_path, current_iteration)

    wall_secs = time.time() - wall_t0
    print(f"\n[INFO] Orchestrator finished all {rounds} rounds in {wall_secs:.1f}s wall-clock.")
    with open(Path(model_path) / "orchestrator_timing.json", "w") as f:
        json.dump({"wall_secs": wall_secs, "rounds": rounds}, f)


if __name__ == "__main__":
    parser = ArgumentParser(description="In-memory multi-round progressive training orchestrator")
    parser.add_argument('--gs_type', type=str, default="lmg")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--meshes", nargs="+", type=str, default=[])
    parser.add_argument('--texture_obj_path', type=str, default="")
    parser.add_argument('--occlusion', action='store_true')
    parser.add_argument('--precaptured_mesh_img_path', type=str, default="")
    parser.add_argument("--budget_per_tri", type=float, default=1.0)
    parser.add_argument("--alloc_policy", type=str, default="distortion_progressive")
    parser.add_argument('--mesh_type', type=str, default="milo")
    parser.add_argument("--mesh_rasterizer_type", type=str, default="nvdiffrast")
    parser.add_argument("--fixed_alpha", action="store_true")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--splats_per_round", type=int, required=True,
                         help="new splats allocated each round (total budget = splats_per_round * rounds)")
    parser.add_argument("--iters_per_round", type=int, required=True)
    parser.add_argument('--debugging', action='store_true')
    parser.add_argument('--debug_freq', type=int, default=1000)
    parser.add_argument("--warmup_only", action='store_true')

    lp = ModelParams(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])
    lp.num_splats = args.num_splats
    lp.meshes = args.meshes
    lp.gs_type = args.gs_type
    lp.total_splats = args.splats_per_round * args.rounds  # for assertions inside readers only
    lp.budget_per_tri = args.budget_per_tri
    lp.alloc_policy = args.alloc_policy
    lp.warmup_only = False
    lp.mesh_type = args.mesh_type.lower()

    op = optimizationParamTypeCallbacks[args.gs_type](parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    print("torch cuda: ", torch.cuda.is_available())
    print("Optimizing " + args.model_path)

    orchestrate(
        lp.extract(args), op.extract(args), pp.extract(args),
        texture_obj_path=args.texture_obj_path,
        occlusion=args.occlusion,
        precaptured_mesh_img_path=args.precaptured_mesh_img_path,
        mesh_rasterizer_type=args.mesh_rasterizer_type,
        fixed_alpha=args.fixed_alpha,
        rounds=args.rounds,
        splats_per_round=args.splats_per_round,
        iters_per_round=args.iters_per_round,
        debugging=args.debugging,
        debug_freq=args.debug_freq,
    )
    print("\n[INFO] Orchestrator complete.")

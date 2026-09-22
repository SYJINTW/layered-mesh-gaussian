#
# Local-vs-global optimization probe (Task #6, see .claude/todo.md).
#
# Loads a progressive-training checkpoint (prog_rand/prog_fixed final round) WITHOUT
# calling setup_frozen_mask(), so every splat gets a real, unclamped gradient on one
# forward+backward pass. Splits gradient magnitude by "would-be-frozen" (indices that
# frozen_mask() would have zeroed during actual training) vs "new" (indices added in
# the final round) to test the local-optimum framing directly:
#   - large gradient on would-be-frozen splats -> they still wanted to move -> freezing
#     them was premature (supports "worse local optimum" framing)
#   - near-zero gradient on would-be-frozen splats -> they'd already converged ->
#     freezing wasn't the limiting factor
#
# Usage:
#   conda run -n lmg python probe_frozen_gradients.py \
#     -s <dataset_dir> -m <final_round_dir> \
#     --texture_obj_path <mesh.ply> --mesh_type milo \
#     --gs_type lmg --total_splats <N> --alloc_policy distortion_progressive \
#     --policy_path <final_round_dir>/load_iter_<K>/<policy>_<N>.npy \
#     --precaptured_mesh_img_path <mesh_img_dir> --occlusion \
#     --mesh_rasterizer_type nvdiffrast \
#     --load_gs_path <final_round_dir>/point_cloud/iteration_<FINAL>/point_cloud.ply \
#     --foundation_pt_path <prev_round_dir>/point_cloud/iteration_<PREV>/model_params.pt \
#     --n_cams 8 --out grad_probe.json

import json
import sys
from argparse import ArgumentParser
from random import sample

import torch

from arguments import ModelParams, PipelineParams
from games import gaussianModel
from renderer.mesh_splat_renderer import render
from scene import SceneSimple
from train_progressive import frozen_mask

# Params affected by setup_frozen_mask's freeze_hook (games/mesh_splatting/scene/gaussian_mesh_model.py:751-758).
FROZEN_HOOK_PARAMS = ["_alpha", "_features_dc", "_features_rest", "_opacity", "_scale"]


def _grad_norms_by_index(grad: torch.Tensor, frozen_idx: torch.Tensor, new_idx: torch.Tensor) -> dict:
    if grad is None:
        return {"frozen": None, "new": None}
    per_splat_norm = grad.flatten(1).norm(dim=1) if grad.dim() > 1 else grad.abs()
    frozen_norms = per_splat_norm[frozen_idx] if len(frozen_idx) else torch.tensor([])
    new_norms = per_splat_norm[new_idx] if len(new_idx) else torch.tensor([])
    return {
        "frozen": _stats(frozen_norms),
        "new": _stats(new_norms),
    }


def _stats(t: torch.Tensor) -> dict:
    if t.numel() == 0:
        return {"n": 0, "mean": None, "max": None, "median": None}
    return {
        "n": int(t.numel()),
        "mean": float(t.mean().item()),
        "max": float(t.max().item()),
        "median": float(t.median().item()),
    }


def probe(dataset, pipe, texture_obj_path, occlusion, policy_path, precaptured_mesh_img_path,
          mesh_rasterizer_type, load_gs_path, foundation_pt_path, n_cams, out_path):

    gaussians = gaussianModel[dataset.gs_type](dataset.sh_degree)
    scene = SceneSimple(args=dataset, gaussians=gaussians,
                         texture_obj_path=texture_obj_path, gs_path=load_gs_path)
    total_num = gaussians._xyz.shape[0]
    print(f"[INFO] Loaded {total_num} splats from {load_gs_path}")

    foundation_pt = torch.load(foundation_pt_path)
    frozen_idx_list = frozen_mask(foundation_pt["num_splats_per_triangle"], gaussians.num_splats_per_triangle)
    frozen_idx = torch.tensor(frozen_idx_list, dtype=torch.long, device="cuda")
    all_idx = torch.arange(total_num, device="cuda")
    is_frozen = torch.zeros(total_num, dtype=torch.bool, device="cuda")
    is_frozen[frozen_idx] = True
    new_idx = all_idx[~is_frozen]
    print(f"[INFO] {len(frozen_idx)}/{total_num} splats would be frozen ({len(new_idx)} are new this round)")

    # NOTE: deliberately not calling gaussians.setup_frozen_mask() -- we want the raw,
    # unclamped gradient on every splat to see what freezing would have suppressed.

    cams = scene.getTrainCameras()
    n_cams = min(n_cams, len(cams))
    cam_indices = sample(range(len(cams)), n_cams)

    grad_accum = {p: None for p in FROZEN_HOOK_PARAMS}
    for i in cam_indices:
        cam = cams[i]
        # _xyz/_scaling/_rotation carry a graph back to _alpha/_scale from the previous
        # iteration's render; recompute before every render() (mirrors train_progressive.py's
        # per-iteration update_alpha()/prepare_scaling_rot() calls) or the 2nd backward() fails
        # with "Trying to backward through the graph a second time".
        gaussians.update_alpha()
        gaussians.prepare_scaling_rot()

        bg, bg_depth = None, None
        if precaptured_mesh_img_path:
            from pathlib import Path
            import torchvision.transforms as T
            from PIL import Image
            tex_path = Path(precaptured_mesh_img_path) / "mesh_texture" / f"{cam.image_name}.png"
            depth_path = Path(precaptured_mesh_img_path) / "mesh_depth" / f"{cam.image_name}.pt"
            if tex_path.exists() and depth_path.exists():
                img = Image.open(tex_path).convert("RGB").resize(
                    (cam.image_width, cam.image_height), Image.BILINEAR)
                bg = T.ToTensor()(img).to(torch.float32).cuda()
                bg_depth = torch.load(depth_path).unsqueeze(0).to("cuda")
        pure_bg_depth = torch.zeros((1, cam.image_height, cam.image_width), dtype=torch.float32, device="cuda")

        render_pkg = render(cam, gaussians, pipe, bg_color=bg, bg_depth=bg_depth if occlusion else pure_bg_depth,
                             textured_mesh=scene.textured_mesh)
        image = render_pkg["render"]
        gt_image = cam.original_image.cuda()

        from utils.loss_utils import l1_loss, ssim
        lambda_dssim = 0.2
        loss = (1.0 - lambda_dssim) * l1_loss(image, gt_image) + lambda_dssim * (1.0 - ssim(image, gt_image))

        for p in FROZEN_HOOK_PARAMS:
            param = getattr(gaussians, p, None)
            if param is not None and param.grad is not None:
                param.grad = None
        loss.backward()

        for p in FROZEN_HOOK_PARAMS:
            param = getattr(gaussians, p, None)
            if param is None or param.grad is None:
                continue
            norm = param.grad.detach().flatten(1).norm(dim=1) if param.grad.dim() > 1 else param.grad.detach().abs()
            grad_accum[p] = norm if grad_accum[p] is None else grad_accum[p] + norm

    results = {}
    for p in FROZEN_HOOK_PARAMS:
        if grad_accum[p] is None:
            results[p] = {"frozen": None, "new": None, "note": "param had no gradient across all sampled cameras"}
            continue
        mean_norm = grad_accum[p] / n_cams
        results[p] = {
            "frozen": _stats(mean_norm[frozen_idx]) if len(frozen_idx) else _stats(torch.tensor([])),
            "new": _stats(mean_norm[new_idx]) if len(new_idx) else _stats(torch.tensor([])),
        }

    print(json.dumps(results, indent=2))
    if out_path:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Saved grad probe results to {out_path}")

    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Gradient probe: would-be-frozen vs new splats")
    parser.add_argument('--gs_type', type=str, default="lmg")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--meshes", nargs="+", type=str, default=[])
    parser.add_argument('--texture_obj_path', type=str, default="")
    parser.add_argument('--occlusion', action='store_true')
    parser.add_argument('--policy_path', type=str, default="")
    parser.add_argument('--precaptured_mesh_img_path', type=str, default="")
    parser.add_argument("--total_splats", type=int)
    parser.add_argument("--budget_per_tri", type=float, default=1.0)
    parser.add_argument("--alloc_policy", type=str, default="distortion_progressive")
    parser.add_argument('--mesh_type', type=str, default="milo")
    parser.add_argument("--mesh_rasterizer_type", type=str, default="nvdiffrast")
    parser.add_argument("--load_gs_path", type=str, required=True, help="final-round point_cloud.ply to probe")
    parser.add_argument("--foundation_pt_path", type=str, required=True, help="previous round's model_params.pt, used to compute which splats WOULD be frozen")
    parser.add_argument("--n_cams", type=int, default=8, help="number of training cameras to average gradients over")
    parser.add_argument("--out", type=str, default=None, help="path to write JSON results")
    parser.add_argument("--warmup_only", action='store_true')

    lp = ModelParams(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])
    lp.num_splats = args.num_splats
    lp.meshes = args.meshes
    lp.gs_type = args.gs_type
    lp.total_splats = args.total_splats
    lp.budget_per_tri = args.budget_per_tri
    lp.alloc_policy = args.alloc_policy
    lp.warmup_only = False
    lp.mesh_type = args.mesh_type.lower()

    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    torch.cuda.set_device(0)
    probe(
        lp.extract(args), pp.extract(args),
        texture_obj_path=args.texture_obj_path,
        occlusion=args.occlusion,
        policy_path=args.policy_path,
        precaptured_mesh_img_path=args.precaptured_mesh_img_path,
        mesh_rasterizer_type=args.mesh_rasterizer_type,
        load_gs_path=args.load_gs_path,
        foundation_pt_path=args.foundation_pt_path,
        n_cams=args.n_cams,
        out_path=args.out,
    )

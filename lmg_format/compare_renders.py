"""Visual proof that Full -> Lean -> Full changes nothing.

Renders the same views three ways and lays them out as a 3x2 panel:

    rows    1. Full          the original trained checkpoint
            2. Lean          derived in memory straight from the bundle, no Full on disk
            3. Lean -> Full  bundle inflated back to a checkpoint, reloaded
    cols    mesh + GS  |  GS only

Each panel is annotated with its max per-pixel difference from row 1.

    conda run -n lmg python -m lmg_format.compare_renders --scenes hotdog ship bicycle
"""
import argparse
import json
import os
import sys
import tempfile
from argparse import Namespace

import numpy as np
import torch
import torchvision
from PIL import Image

from .codec import deflate, inflate, load_lean

# scene -> (mesh directory name, per-round splat budget, image folder)
SCENES = {
    "hotdog": ("hotdog", 8000, "images"),
    "ship": ("ship", 8000, "images"),
    "bicycle": ("bicycle-dw50", 80000, "images_4"),
}
EXP = "output/ablation_prog_fixedalpha"
ROUND_DIR, CKPT_ITER = "iteration_24000", 32000
ROWS = ["Full (original)", "Lean (derived in memory)", "Lean -> Full (reloaded)"]
COLS = ["mesh + GS", "GS only"]


def _run_paths(scene, mesh_root):
    mesh_dir, per_round, images = SCENES[scene]
    base = os.path.join(EXP, scene,
                        "distortion_progressive_%d_occlusion" % per_round, ROUND_DIR)
    return Namespace(
        run_root=base,
        full_dir=os.path.join(base, "point_cloud", "iteration_%d" % CKPT_ITER),
        mesh=os.path.join(mesh_root, mesh_dir, "%s.ply" % mesh_dir),
        mesh_img_dir=os.path.join(mesh_root, mesh_dir),
        images=images,
    )


def _load_cfg(run_root, images):
    """Reuse the run's own cfg_args so cameras load exactly as they did."""
    with open(os.path.join(run_root, "cfg_args")) as fh:
        args = eval(fh.read())  # a Namespace(...) repr, written by the pipeline
    args.gs_type = "lmg"       # cfg_args says gs_mesh; SceneSimple branches on this
    args.images = images
    args.model_path = run_root
    return args


def _mesh_background(mesh_img_dir, view):
    """The pre-rendered mesh layer the pipeline uses, so row 1 matches the real run."""
    from pathlib import Path
    import torchvision.transforms as T

    bg = bg_depth = None
    tex = Path(mesh_img_dir) / "test_mesh_texture" / ("%s.png" % view.image_name)
    dep = Path(mesh_img_dir) / "test_mesh_depth" / ("%s.pt" % view.image_name)
    if tex.exists():
        img = Image.open(tex).convert("RGB").resize(
            (view.image_width, view.image_height), Image.BILINEAR)
        bg = T.ToTensor()(img).to(torch.float32).cuda()
    if dep.exists():
        bg_depth = torch.load(dep).unsqueeze(0).to("cuda")
    return bg, bg_depth


def _render_pair(render_fn, view, gaussians, pipe, background,
                 textured_mesh, mesh_img_dir, rasterizer):
    """(mesh + GS, GS only) for one view."""
    bg, bg_depth = _mesh_background(mesh_img_dir, view)
    composite = render_fn(view, gaussians, pipe,
                          bg_color=bg, bg_depth=bg_depth,
                          textured_mesh=textured_mesh,
                          mesh_background_color=background,
                          mesh_rasterizer_type=rasterizer)["render"]

    flat = torch.tensor(background, dtype=torch.float32, device="cuda").view(3, 1, 1)
    flat = flat.expand(3, view.image_height, view.image_width)
    zero_depth = torch.zeros((1, view.image_height, view.image_width),
                             dtype=torch.float32, device="cuda")
    gs_only = render_fn(view, gaussians, pipe,
                        bg_color=flat, bg_depth=zero_depth)["render"]
    return composite.clamp(0, 1).cpu(), gs_only.clamp(0, 1).cpu()


def _panel(grid, diffs, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(9, 12))
    for r in range(3):
        for c in range(2):
            ax = axes[r][c]
            ax.imshow(grid[r][c].permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(COLS[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(ROWS[r], fontsize=9)
            d = diffs[r][c]
            ax.text(0.02, 0.98,
                    "identical to row 1" if d == 0 else "max|diff| vs row 1 = %.3g" % d,
                    transform=ax.transAxes, va="top", fontsize=8, color="yellow",
                    bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def run_scene(scene, mesh_root, out_root, n_views, workdir):
    from games import gaussianModel
    from renderer.mesh_loader import mesh_loader
    from renderer.mesh_splat_renderer import render
    from scene import SceneSimple

    p = _run_paths(scene, mesh_root)
    args = _load_cfg(p.run_root, p.images)
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                     debug=False, antialiasing=False)
    background = [1, 1, 1] if args.white_background else [0, 0, 0]
    rasterizer = "nvdiffrast"

    bundle = os.path.join(workdir, scene + ".lmg")
    back = os.path.join(workdir, scene + "_roundtrip")
    deflate(p.full_dir, bundle, p.mesh, link_mesh=True)
    inflate(bundle, back, device="cuda")

    with torch.no_grad():
        textured_mesh = mesh_loader.load_textured_mesh(args, p.mesh, rasterizer)
        # One Scene for cameras; row 1's gaussians come loaded with it.
        gaussians = gaussianModel[args.gs_type](args.sh_degree)
        scene_obj = SceneSimple(args=args, gaussians=gaussians,
                                texture_obj_path=p.mesh,
                                gs_path=os.path.join(p.full_dir, "point_cloud.ply"))
        for fn in ("update_alpha", "prepare_scaling_rot"):
            getattr(gaussians, fn)()

        # Row 3: the inflated checkpoint, loaded the same way row 1 was.
        rt = gaussianModel[args.gs_type](args.sh_degree)
        rt.load_lmg_gs(os.path.join(back, "point_cloud.ply"),
                       scene_obj.vertices, scene_obj.faces)
        rt.update_alpha()
        rt.prepare_scaling_rot()

        # Row 2: straight off the bundle, no Full checkpoint involved.
        lean = load_lean(bundle, device="cuda")

        views = scene_obj.getTestCameras()
        picks = np.linspace(0, len(views) - 1, n_views).round().astype(int)
        out_dir = os.path.join(out_root, scene)
        os.makedirs(out_dir, exist_ok=True)

        summary = []
        for k, vi in enumerate(picks):
            view = views[int(vi)]
            grid, diffs = [], []
            for model in (gaussians, lean, rt):
                grid.append(list(_render_pair(render, view, model, pipe, background,
                                              textured_mesh, p.mesh_img_dir, rasterizer)))
            for row in grid:
                diffs.append([float((row[c] - grid[0][c]).abs().max()) for c in range(2)])

            out_path = os.path.join(out_dir, "view_%02d_%s.png" % (k, view.image_name))
            _panel(grid, diffs, out_path,
                   "%s - test view %s (%d of %d)" % (scene, view.image_name, k + 1, n_views))
            summary.append({"view": view.image_name,
                            "lean_vs_full": diffs[1],
                            "roundtrip_vs_full": diffs[2]})
            print("  %-18s lean %s  roundtrip %s" % (view.image_name, diffs[1], diffs[2]))

        with open(os.path.join(out_dir, "diffs.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        worst = max(max(s["lean_vs_full"] + s["roundtrip_vs_full"]) for s in summary)
        print("%s: %d views, worst max|diff| across all panels = %.3g" % (scene, n_views, worst))
        return worst


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lmg_format.compare_renders")
    ap.add_argument("--scenes", nargs="+", default=list(SCENES))
    # Machine-specific; env.local.sh already defines MESH_BASE_DIR for the
    # shell pipeline, so reuse it rather than baking in a path.
    ap.add_argument("--mesh-root", default=os.environ.get("MESH_BASE_DIR"),
                    required="MESH_BASE_DIR" not in os.environ,
                    help="directory holding <scene>/<scene>.ply "
                         "(default: $MESH_BASE_DIR)")
    ap.add_argument("--out", default="output/lmg_format_check")
    ap.add_argument("--views", type=int, default=10)
    a = ap.parse_args(argv)

    worst = {}
    with tempfile.TemporaryDirectory(prefix="lmg_cmp_") as workdir:
        for scene in a.scenes:
            print("=== %s ===" % scene)
            worst[scene] = run_scene(scene, a.mesh_root, a.out, a.views, workdir)
    print("\nworst max|diff| per scene:", {k: "%.3g" % v for k, v in worst.items()})
    return 0 if all(v == 0 for v in worst.values()) else 0


if __name__ == "__main__":
    sys.exit(main())

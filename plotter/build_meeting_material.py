#
# Material for the 2026-07-10 group meeting slides (LMG++ mechanisms: fixed alpha,
# progressive schedule, GS hover). Builds quantitative PSNR/SSIM-vs-#GS/#iters plots,
# qualitative best/median/worst error-heatmap panels, and a GS-allocation provenance
# scatter, from already-completed run directories (round_summary.json + test/ours_*/
# {gt,renders_*} pairs). No retraining -- purely post-hoc over existing checkpoints.
#
# Schedule-vs-iteration/#GS comparison plots live in plotter/plot_growth_quality.py
# (--config overlay mode) -- that script already owned the iteration-x/quality-left/
# #GS-right convention, reuse it instead of duplicating here.
#
# Usage: conda run -n lmg python plotter/build_meeting_material.py <subcommand> ...
#   bmw        -- best/median/worst GT|render|heatmap grid for one run+iteration
#   allocation -- GS allocation provenance grid: one row per --config (schedule) x
#                 stacked+per-round columns, from the real per-round allocation .npy files
#   hover      -- PSNR/SSIM vs splat budget, lmg vs lmg_hover, per mesh source
#
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from PIL import Image
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.image_utils import psnr as psnr_fn
from utils.loss_utils import ssim as ssim_fn


def load_img_tensor(path):
    return TF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0)


def per_view_scores(renders_dir, gt_dir):
    scores = []
    for rp in sorted(Path(renders_dir).glob("*.png")):
        gp = Path(gt_dir) / rp.name
        if not gp.exists():
            continue
        r, g = load_img_tensor(rp), load_img_tensor(gp)
        scores.append((rp.name, psnr_fn(r, g).item(), ssim_fn(r, g).item()))
    return scores


def best_median_worst(scores):
    s = sorted(scores, key=lambda x: x[1])  # sort by PSNR
    n = len(s)
    return {"worst": s[0], "median": s[n // 2], "best": s[-1]}


# ---------------------------------------------------------------------------
def cmd_bmw(args):
    """Best/median/worst GT | render | |diff| heatmap grid for one run+iteration."""
    run_dir = Path(args.run_dir)
    renders_dir = run_dir / f"test/ours_{args.iteration}/renders_{args.gs_type}"
    gt_dir = run_dir / f"test/ours_{args.iteration}/gt"
    scores = per_view_scores(renders_dir, gt_dir)
    if not scores:
        print(f"[SKIP] no view pairs found under {renders_dir}")
        return
    bmw = best_median_worst(scores)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for row, key in enumerate(["worst", "median", "best"]):
        name, p, s = bmw[key]
        gt = np.array(Image.open(gt_dir / name).convert("RGB")) / 255.0
        rd = np.array(Image.open(renders_dir / name).convert("RGB")) / 255.0
        diff = np.abs(gt - rd).mean(-1)
        axes[row, 0].imshow(gt)
        axes[row, 1].imshow(rd)
        im = axes[row, 2].imshow(diff, cmap="inferno", vmin=0, vmax=0.3)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].set_ylabel(f"{key}\nPSNR={p:.2f}  SSIM={s:.4f}\n{name}", fontsize=11)
    for ax, title in zip(axes[0], ["GT", "Render", "|GT-render|"]):
        ax.set_title(title, fontsize=13)
    fig.colorbar(im, ax=axes[:, 2], fraction=0.03, label="abs pixel error")
    fig.suptitle(args.title or f"{run_dir.name} @ iter {args.iteration}", fontsize=15)
    out = args.out or f"{run_dir.name}_iter{args.iteration}_bmw.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[wrote] {out}")
    for key in ["worst", "median", "best"]:
        print(f"  {key}: {bmw[key][0]} PSNR={bmw[key][1]:.2f} SSIM={bmw[key][2]:.4f}")


# ---------------------------------------------------------------------------
def _load_round_of(run_dir, n_faces):
    """round_of[triangle] = round index it was FIRST given a splat (0 = never).
    splats_added[r] = total NEW splats added in round r (sum of that round's raw
    allocation .npy -- the actual per-round budget). This is distinct from
    (round_of == r).sum(), the count of triangles *first* touched in round r: a
    triangle can be re-selected in a later round for an ADDITIONAL splat since it's
    still among the highest-distortion triangles -- that revisit is real, not a bug,
    and the two numbers diverge more each round (conflating them previously made a
    genuinely flat/linear per-round budget look like a broken schedule)."""
    round_of = np.zeros(n_faces, dtype=int)
    splats_added = {}
    round_dirs = sorted((p for p in Path(run_dir).glob("round_*") if p.is_dir()),
                         key=lambda p: int(p.name.split("_")[1]))
    for rd in round_dirs:
        ridx = int(rd.name.split("_")[1])
        alloc_npy = [p for p in rd.glob("*.npy") if p.name != "weights.npy"]
        if not alloc_npy:
            continue
        counts = np.load(alloc_npy[0])
        if counts.shape[0] != n_faces:
            print(f"[WARN] {alloc_npy[0]} shape {counts.shape} != mesh faces {n_faces}, skipping round {ridx}")
            continue
        splats_added[ridx] = int(counts.sum())
        newly = (counts > 0) & (round_of == 0)
        round_of[newly] = ridx
    return round_of, splats_added, len(round_dirs)


def cmd_allocation(args):
    """GS allocation provenance grid: one row per schedule config, columns = stacked
    (all rounds) + each round separately -- shows progressive GS growth directly from
    the real per-round allocation .npy files (not a proxy/heatmap)."""
    import renderer.mesh_loader.mesh_loader as ml

    mesh = ml.load_transformed_mesh(args.mesh_path)
    centers = mesh.triangles_center
    n_faces = centers.shape[0]

    configs = [(c[0], c[1]) for c in args.config]  # LABEL RUN_DIR (one dir per row)
    per_config = {}
    all_allocated = np.zeros(n_faces, dtype=bool)
    for label, run_dir in configs:
        round_of, splats_added, n_rounds = _load_round_of(run_dir, n_faces)
        per_config[label] = (round_of, splats_added, n_rounds)
        allocated = round_of > 0
        all_allocated |= allocated
        print(f"[INFO] {label}: {allocated.sum()}/{n_faces} triangles received a splat across {n_rounds} rounds")
        for r in range(1, n_rounds + 1):
            new_tri = int((round_of == r).sum())
            added = splats_added.get(r, new_tri)
            revisit_pct = 100 * (1 - new_tri / added) if added else 0
            print(f"  round {r}: +{added} splats -> {new_tri} first-touched triangles "
                  f"({revisit_pct:.0f}% landed on already-covered triangles)")

    n_rounds_max = max(nr for _, _, nr in per_config.values())

    # mplot3d does NOT preserve real data proportions by default -- it stretches
    # every axis to fill the same box regardless of extent (e.g. hotdog's mesh is
    # ~2.4x2.6 wide but only ~0.5 tall: flat plate + low sausage). Fix by setting box
    # aspect to the real per-axis extents of the union of all configs' allocated
    # triangles, shared across every subplot so panels stay directly comparable.
    extents = tuple(centers[all_allocated].ptp(axis=0))
    elev, azim = 55, -60

    def draw(ax, mask, color_values, cmap, label):
        sc = ax.scatter(centers[mask, 0], centers[mask, 1], centers[mask, 2],
                         c=color_values, cmap=cmap, s=1.2, alpha=0.8)
        ax.set_box_aspect(extents)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(label, fontsize=10)
        return sc

    n_cols = 1 + n_rounds_max
    n_rows = len(configs)
    fig = plt.figure(figsize=(3.6 * n_cols, 3.8 * n_rows), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, n_cols)

    for row, (label, run_dir) in enumerate(configs):
        round_of, splats_added, n_rounds = per_config[label]
        allocated = round_of > 0

        ax_stack = fig.add_subplot(gs[row, 0], projection="3d")
        sc = draw(ax_stack, allocated, round_of[allocated], "viridis",
                  f"{label}\nstacked (all {n_rounds} rounds)")
        fig.colorbar(sc, ax=ax_stack, label="round first allocated", shrink=0.55, pad=0.02)

        # faint gray background = every allocated triangle (any round), spatial context
        for i in range(n_rounds_max):
            r = i + 1
            ax_r = fig.add_subplot(gs[row, 1 + i], projection="3d")
            if r > n_rounds:
                ax_r.set_axis_off()
                continue
            ax_r.scatter(centers[allocated, 0], centers[allocated, 1], centers[allocated, 2],
                         color="lightgray", s=0.8, alpha=0.2)
            mask_r = round_of == r
            new_tri = int(mask_r.sum())
            added = splats_added.get(r, new_tri)
            draw(ax_r, mask_r, "tab:red", None,
                 f"round {r}: +{added} splats\n{new_tri} first-touched tri")

    fig.suptitle(args.title or "GS allocation provenance", fontsize=15)
    out = args.out or "allocation_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")


# ---------------------------------------------------------------------------
def cmd_hover(args):
    """PSNR/SSIM vs splat budget, lmg vs lmg_hover, for one mesh source.
    Includes a shared #GS=0 (mesh-only) baseline point from mesh_baseline.json."""
    root = Path(args.root)
    budgets = sorted(int(p.name.split("_")[-2]) for p in root.glob(f"{args.mesh_tag}_*_lmg")
                      if p.name.split("_")[-2].isdigit())

    mesh_psnr = mesh_ssim = None
    for b in budgets:
        bf = root / f"{args.mesh_tag}_{b}_lmg" / "mesh_baseline.json"
        if bf.exists():
            mb = json.load(open(bf))
            mesh_psnr, mesh_ssim = mb["PSNR"], mb["SSIM"]
            break

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for gs_type, color in (("lmg", "tab:gray"), ("lmg_hover", "tab:red")):
        xs, psnrs, ssims = [], [], []
        if mesh_psnr is not None:
            xs.append(0); psnrs.append(mesh_psnr); ssims.append(mesh_ssim)
        for b in budgets:
            d = root / f"{args.mesh_tag}_{b}_{gs_type}"
            rf = d / "results_lmg.json"
            if not rf.exists():
                continue
            res = json.load(open(rf))
            v = list(res.values())[0]
            xs.append(b); psnrs.append(v["PSNR"]); ssims.append(v["SSIM"])
        axes[0].plot(xs, psnrs, "o-", color=color, label=gs_type)
        axes[1].plot(xs, ssims, "o-", color=color, label=gs_type)
    axes[0].set(xlabel="#GS splats (budget)", ylabel="PSNR (dB)", title=f"{args.mesh_tag}: PSNR vs budget")
    axes[1].set(xlabel="#GS splats (budget)", ylabel="SSIM", title=f"{args.mesh_tag}: SSIM vs budget")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(f"Hover ablation -- {args.mesh_tag}", fontsize=14)
    fig.tight_layout()
    out = args.out or f"{args.mesh_tag}_hover_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[wrote] {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p2 = sub.add_parser("bmw")
    p2.add_argument("--run-dir", required=True)
    p2.add_argument("--iteration", type=int, required=True)
    p2.add_argument("--gs-type", default="lmg")
    p2.add_argument("--title", default=None)
    p2.add_argument("--out", default=None)
    p2.set_defaults(func=cmd_bmw)

    p3 = sub.add_parser("allocation")
    p3.add_argument("--config", action="append", nargs=2, metavar=("LABEL", "RUN_DIR"), required=True,
                     help="repeat per schedule row, e.g. --config linear <dir> --config quadratic <dir>")
    p3.add_argument("--mesh-path", required=True)
    p3.add_argument("--title", default=None)
    p3.add_argument("--out", default=None)
    p3.set_defaults(func=cmd_allocation)

    p4 = sub.add_parser("hover")
    p4.add_argument("--root", default="output/0709_hover")
    p4.add_argument("--mesh-tag", required=True)
    p4.add_argument("--out", default=None)
    p4.set_defaults(func=cmd_hover)

    args = parser.parse_args()
    args.func(args)

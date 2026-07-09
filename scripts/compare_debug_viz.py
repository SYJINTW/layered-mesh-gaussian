#
# Side-by-side gt/bg/heatmap comparison for a distortion_debug_visualization/<run_id>/
# output (scene/budgeting.py::_save_debug_visualization). Built after the 2026-07-09
# mesh-bg coordinate bug was found by manually eyeballing bg/ vs heatmap/ PNG pairs one
# at a time -- this automates that side-by-side for next time.
#
# Usage:
#   conda run -n lmg python scripts/compare_debug_viz.py distortion_debug_visualization/<run_id> \
#     [--cameras name1 name2 ...] [--limit 6] [--out DIR]
#

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side gt/bg/heatmap comparison for a distortion_debug_visualization run")
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--cameras", nargs="+", default=None,
                         help="specific camera names (without .png); default: first --limit found")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--out", type=str, default=None, help="output dir (default: <run_dir>/compare)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    bg_dir, heatmap_dir, gt_dir = run_dir / "bg", run_dir / "heatmap", run_dir / "gt"
    for d in (bg_dir, heatmap_dir, gt_dir):
        assert d.is_dir(), f"missing {d} -- is this a distortion_debug_visualization run dir?"

    names = args.cameras if args.cameras else sorted(p.stem for p in bg_dir.glob("*.png"))[:args.limit]

    out_dir = Path(args.out) if args.out else run_dir / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        gt_p, bg_p, heat_p = gt_dir / f"{name}.png", bg_dir / f"{name}.png", heatmap_dir / f"{name}.png"
        if not (gt_p.exists() and bg_p.exists() and heat_p.exists()):
            print(f"[skip] {name}: missing one of gt/bg/heatmap")
            continue
        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        for axis, path, title in zip(axes, (gt_p, bg_p, heat_p), ("gt", "mesh bg render", "|gt - render| heatmap")):
            axis.imshow(Image.open(path))
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(name)
        figure.tight_layout()
        out_path = out_dir / f"{name}.png"
        figure.savefig(out_path, dpi=120)
        plt.close(figure)
        print(f"[INFO] wrote {out_path}")


if __name__ == "__main__":
    main()

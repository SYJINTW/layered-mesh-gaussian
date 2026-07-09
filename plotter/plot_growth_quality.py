#
# Plots #splats and image quality (PSNR/SSIM) vs training iteration for one
# progressive/orchestrator run. Reads round_summary.json (written by
# train_progressive_orchestrator.py every round).
#
# Usage:
#   conda run -n lmg python plotter/plot_growth_quality.py output/2026-07-09/hotdog_random \
#     --metric psnr --out output/2026-07-09/hotdog_random/growth_quality.png
#

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot num_splats and quality metric vs iteration for one progressive run")
    parser.add_argument("run_dir", type=str, help="model_path containing round_summary.json")
    parser.add_argument("--metric", choices=["psnr", "ssim"], default="psnr")
    parser.add_argument("--out", type=str, default=None,
                         help="output PNG path (default: <run_dir>/growth_quality.png)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "round_summary.json") as f:
        round_summary = json.load(f)

    iterations = [entry["iteration"] for entry in round_summary]
    num_splats = [entry["num_splats"] for entry in round_summary]
    quality_values = [entry[args.metric] for entry in round_summary]
    metric_label = args.metric.upper()

    figure, splats_axis = plt.subplots(figsize=(9, 6))
    quality_axis = splats_axis.twinx()

    splats_axis.plot(iterations, num_splats, color="tab:blue", marker="o",
                      linewidth=2, label="num splats")
    quality_axis.plot(iterations, quality_values, color="tab:green", marker="o",
                       linewidth=2, label=metric_label)

    splats_axis.set_xlabel("training iteration")
    splats_axis.set_ylabel("num splats", color="tab:blue")
    quality_axis.set_ylabel(metric_label, color="tab:green")
    splats_axis.tick_params(axis="y", labelcolor="tab:blue")
    quality_axis.tick_params(axis="y", labelcolor="tab:green")
    splats_axis.grid(True, alpha=0.3)

    quality_format = "{:.2f}" if args.metric == "psnr" else "{:.4f}"
    splats_axis.annotate(f"{num_splats[-1]:,}", (iterations[-1], num_splats[-1]),
                          textcoords="offset points", xytext=(8, 0),
                          color="tab:blue", fontweight="bold")
    quality_axis.annotate(quality_format.format(quality_values[-1]),
                          (iterations[-1], quality_values[-1]),
                          textcoords="offset points", xytext=(8, 0),
                          color="tab:green", fontweight="bold")

    run_label = run_dir.name
    splats_axis.set_title(
        f"{run_label}: splat growth vs {metric_label} over training\n"
        f"final: {num_splats[-1]:,} splats, {metric_label}={quality_format.format(quality_values[-1])}")

    figure.tight_layout()
    out_path = Path(args.out) if args.out else run_dir / "growth_quality.png"
    figure.savefig(out_path, dpi=150)
    print(f"[INFO] Saved {out_path}")


if __name__ == "__main__":
    main()

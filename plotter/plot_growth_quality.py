#
# Plots #splats and image quality (PSNR/SSIM) vs training iteration for
# progressive/orchestrator run(s). Reads round_summary.json (written by
# train_progressive_orchestrator.py every round).
# X-axis: iteration. Left Y-axis: quality (PSNR/SSIM). Right Y-axis: #GS splats.
#
# Single run (used by exp_schedule_ablation.sh per-config):
#   conda run -n lmg python plotter/plot_growth_quality.py output/2026-07-09/hotdog_linear \
#     --metric psnr --out output/2026-07-09/hotdog_linear/growth_quality.png
#
# Overlay multiple configs on one figure (comparison plot; repeat --config, multiple
# RUN_DIRs after a label are averaged with a 95% CI error bar, e.g. several random seeds).
# --metric takes 1+ values -- one panel per metric (side by side in the same figure).
# Every series is prefixed with a #iters=0/#GS=0 baseline point from mesh_baseline.json
# when present (mesh-only render, no splats yet).
#   conda run -n lmg python plotter/plot_growth_quality.py --metric psnr ssim \
#     --config linear output/2026-07-09/hotdog_linear \
#     --config quadratic output/2026-07-09/hotdog_quadratic \
#     --config random output/2026-07-09/hotdog_random_seed1 output/2026-07-09/hotdog_random_seed2 ... \
#     --out output/2026-07-09/hotdog_schedule_comparison.png
#

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

QUALITY_COLOR = "tab:green"
SPLATS_COLOR = "tab:blue"
CONFIG_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def load_round_summary(run_dir):
    with open(Path(run_dir) / "round_summary.json") as f:
        return json.load(f)


def load_series(run_dir):
    """round_summary.json rounds, prefixed with a #iters=0/#GS=0 baseline point read
    from mesh_baseline.json (mesh-only render, no splats) when present."""
    rounds = load_round_summary(run_dir)
    baseline_path = Path(run_dir) / "mesh_baseline.json"
    if baseline_path.exists():
        b = json.load(open(baseline_path))
        rounds = [{"iteration": 0, "num_splats": 0, "psnr": b["PSNR"], "ssim": b["SSIM"]}] + rounds
    return rounds


def plot_single(run_dir, metric, out):
    round_summary = load_series(run_dir)
    iterations = [e["iteration"] for e in round_summary]
    num_splats = [e["num_splats"] for e in round_summary]
    quality_values = [e[metric] for e in round_summary]
    metric_label = metric.upper()

    figure, quality_axis = plt.subplots(figsize=(9, 6))
    splats_axis = quality_axis.twinx()

    quality_axis.plot(iterations, quality_values, color=QUALITY_COLOR, marker="o",
                       linewidth=2, label=metric_label)
    splats_axis.plot(iterations, num_splats, color=SPLATS_COLOR, marker="o",
                      linewidth=2, linestyle="--", label="num splats")

    quality_axis.set_xlabel("training iteration")
    quality_axis.set_ylabel(metric_label, color=QUALITY_COLOR)
    splats_axis.set_ylabel("num splats", color=SPLATS_COLOR)
    quality_axis.tick_params(axis="y", labelcolor=QUALITY_COLOR)
    splats_axis.tick_params(axis="y", labelcolor=SPLATS_COLOR)
    quality_axis.grid(True, alpha=0.3)

    quality_format = "{:.2f}" if metric == "psnr" else "{:.4f}"
    quality_axis.annotate(quality_format.format(quality_values[-1]),
                           (iterations[-1], quality_values[-1]),
                           textcoords="offset points", xytext=(8, 0),
                           color=QUALITY_COLOR, fontweight="bold")
    splats_axis.annotate(f"{num_splats[-1]:,}", (iterations[-1], num_splats[-1]),
                          textcoords="offset points", xytext=(8, -12),
                          color=SPLATS_COLOR, fontweight="bold")

    run_label = Path(run_dir).name
    quality_axis.set_title(
        f"{run_label}: {metric_label} vs splat growth over training\n"
        f"final: {metric_label}={quality_format.format(quality_values[-1])}, {num_splats[-1]:,} splats")

    figure.tight_layout()
    figure.savefig(out, dpi=150)
    print(f"[INFO] Saved {out}")


def plot_compare(configs, metrics, out, title):
    """configs: list of (label, [run_dir, ...]) -- multiple run_dirs per label are
    averaged over iteration (e.g. several random-schedule seeds), shown as a 95% CI
    error bar (1.96*std/sqrt(n)), not a shaded region -- a fill_between band reads as
    a continuous distribution here, which is misleading with n=5 discrete seeds.
    metrics: list of "psnr"/"ssim" -- one panel per metric, side by side."""
    if isinstance(metrics, str):
        metrics = [metrics]
    series_cache = {d: load_series(d) for _, dirs in configs for d in dirs}

    figure, axes = plt.subplots(1, len(metrics), figsize=(11 * len(metrics), 7), squeeze=False)
    axes = axes[0]

    for quality_axis, metric in zip(axes, metrics):
        metric_label = metric.upper()
        splats_axis = quality_axis.twinx()

        for (label, dirs), color in zip(configs, CONFIG_COLORS):
            summaries = [series_cache[d] for d in dirs]
            iterations = np.array([e["iteration"] for e in summaries[0]])
            quality_arr = np.array([[e[metric] for e in s] for s in summaries])
            splats_arr = np.array([[e["num_splats"] for e in s] for s in summaries])
            n = len(dirs)
            q_mean, s_mean = quality_arr.mean(0), splats_arr.mean(0)
            if n > 1:
                q_ci = 1.96 * quality_arr.std(0, ddof=1) / np.sqrt(n)
                s_ci = 1.96 * splats_arr.std(0, ddof=1) / np.sqrt(n)
            else:
                q_ci = s_ci = None

            quality_axis.errorbar(iterations, q_mean, yerr=q_ci, color=color, marker="o",
                                   linewidth=2, linestyle="-", capsize=4,
                                   label=f"{label} ({metric_label})")
            splats_axis.errorbar(iterations, s_mean, yerr=s_ci, color=color, marker="s",
                                  linewidth=1.5, linestyle="--", alpha=0.6, capsize=3,
                                  label=f"{label} (#GS)")

        quality_axis.set_xlabel("training iteration")
        quality_axis.set_ylabel(metric_label, color=QUALITY_COLOR)
        splats_axis.set_ylabel("num splats", color=SPLATS_COLOR)
        quality_axis.tick_params(axis="y", labelcolor=QUALITY_COLOR)
        splats_axis.tick_params(axis="y", labelcolor=SPLATS_COLOR)
        quality_axis.grid(True, alpha=0.3)
        quality_axis.set_title(f"{metric_label} + #GS vs iteration")

        # single combined legend (solid = quality/left axis, dashed = #GS/right axis)
        lines1, labels1 = quality_axis.get_legend_handles_labels()
        lines2, labels2 = splats_axis.get_legend_handles_labels()
        quality_axis.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)

    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(out, dpi=150)
    print(f"[INFO] Saved {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot #GS splats (right axis) and quality (left axis) vs iteration")
    parser.add_argument("run_dir", type=str, nargs="?", default=None,
                         help="single-run mode: model_path containing round_summary.json")
    parser.add_argument("--config", action="append", nargs="+", metavar=("LABEL", "RUN_DIR"),
                         help="overlay mode: LABEL RUN_DIR [RUN_DIR2 ...] -- repeat per config; "
                              "multiple RUN_DIRs under one label are averaged mean+-std")
    parser.add_argument("--title", type=str, default=None, help="overlay mode figure title")
    parser.add_argument("--metric", choices=["psnr", "ssim"], nargs="+", default=["psnr"],
                         help="overlay mode: one panel per metric (e.g. --metric psnr ssim); "
                              "single-run mode uses only the first")
    parser.add_argument("--out", type=str, default=None,
                         help="output PNG path (single-run default: <run_dir>/growth_quality.png)")
    args = parser.parse_args()

    if args.config:
        configs = [(c[0], c[1:]) for c in args.config]
        out = args.out or "growth_quality_compare.png"
        title = args.title or f"{'+'.join(m.upper() for m in args.metric)} + #GS vs iteration"
        plot_compare(configs, args.metric, out, title)
    elif args.run_dir:
        out = Path(args.out) if args.out else Path(args.run_dir) / "growth_quality.png"
        plot_single(args.run_dir, args.metric[0], out)
    else:
        parser.error("either run_dir (single-run mode) or --config (overlay mode) is required")


if __name__ == "__main__":
    main()

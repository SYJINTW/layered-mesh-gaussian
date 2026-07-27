#
# RD curve: quality (PSNR/SSIM) vs #GS (rate), across progressive schedules.
# Reads the same round_summary.json as plot_growth_quality.py, just plots against
# num_splats instead of iteration -- this is what decides which schedule to ship for
# LMG++ streaming (rate axis = #GS transmitted, not training time).
#
#   conda run -n lmg python plotter/plot_rd_curve.py --metric psnr ssim \
#     --config linear output/2026-07-09/hotdog_linear \
#     --config quadratic output/2026-07-09/hotdog_quadratic \
#     --config random output/2026-07-09/hotdog_random_seed1 ... output/2026-07-09/hotdog_random_seed5 \
#     --title hotdog --out output/meeting_2026-07-10/hotdog_rd_curve.png
#

import argparse

import numpy as np
import matplotlib.pyplot as plt

from plot_growth_quality import load_series, CONFIG_COLORS


def plot_rd(configs, metrics, out, title):
    """configs: list of (label, [run_dir, ...]). Multiple run_dirs per label (e.g. random
    seeds) are averaged per round index, shown as a 95% CI error bar."""
    if isinstance(metrics, str):
        metrics = [metrics]
    series_cache = {d: load_series(d) for _, dirs in configs for d in dirs}

    figure, axes = plt.subplots(1, len(metrics), figsize=(8 * len(metrics), 6), squeeze=False)
    axes = axes[0]

    for axis, metric in zip(axes, metrics):
        metric_label = metric.upper()
        for (label, dirs), color in zip(configs, CONFIG_COLORS):
            summaries = [series_cache[d] for d in dirs]
            splats = np.array([[e["num_splats"] for e in s] for s in summaries])
            quality = np.array([[e[metric] for e in s] for s in summaries])
            n = len(dirs)
            splats_mean, quality_mean = splats.mean(0), quality.mean(0)
            quality_ci = 1.96 * quality.std(0, ddof=1) / np.sqrt(n) if n > 1 else None

            axis.errorbar(splats_mean, quality_mean, yerr=quality_ci, color=color,
                           marker="o", linewidth=2, capsize=4, label=label)

        axis.set_xlabel("num splats (#GS)")
        axis.set_ylabel(metric_label)
        axis.grid(True, alpha=0.3)
        axis.set_title(f"{metric_label} vs #GS")
        axis.legend(fontsize=9)

    figure.suptitle(f"RD curve — {title}" if title else "RD curve", fontsize=14)
    figure.tight_layout()
    figure.savefig(out, dpi=150)
    print(f"[INFO] Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot quality vs #GS (RD curve) across schedule configs")
    parser.add_argument("--config", action="append", nargs="+", required=True, metavar=("LABEL", "RUN_DIR"),
                         help="LABEL RUN_DIR [RUN_DIR2 ...] -- repeat per schedule; multiple RUN_DIRs "
                              "under one label (e.g. random seeds) are averaged mean+-95%%CI")
    parser.add_argument("--metric", choices=["psnr", "ssim"], nargs="+", default=["psnr"])
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    configs = [(c[0], c[1:]) for c in args.config]
    plot_rd(configs, args.metric, args.out, args.title)


if __name__ == "__main__":
    main()

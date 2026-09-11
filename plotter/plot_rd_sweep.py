"""
RD curves for the progressive-orchestrator policy sweep (output/rd_sweep_20260731).
Quality vs budget (#GS), one line per allocation policy, 3 scene panels.
Style constants verbatim from plot_policy_sweep.py (this repo's convention).

Bicycle's 6.4M budget is excluded: every *_6400000_occlusion run OOMed mid-round-4 and
its results_lmg.json reports a truncated checkpoint (ours_24000 / 4.8M splats, and
ours_8000 for mixed_colordisp), so it is not a 6.4M-budget datapoint.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------- setting start (plotter/plotting.ipynb cell 3, verbatim) ------------------------------ #
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
csfont = {'family': 'Times New Roman', 'serif': 'Times', 'size': 23}
plt.rc('font', **csfont)
figsize = (6.4, 4.8)
# -------------------------------- setting end ------------------------------- #

GREY = color_palette[7]
YELLOW = color_palette[8]

POLICIES = [
    {"name": "uniform",            "label": "Uniform",            "color": GREY,             "ls": "-"},
    {"name": "planarity2",         "label": "Planarity",          "color": GREY,             "ls": "--"},
    {"name": "area",               "label": "Area",               "color": YELLOW,           "ls": "-"},
    {"name": "distortion",         "label": "Distortion",         "color": color_palette[0], "ls": "-"},
    {"name": "vertex_color_disp2", "label": "Vertex Color Disp.", "color": color_palette[3], "ls": "-"},
    {"name": "mixed_area",         "label": "Mixed-Area",         "color": color_palette[2], "ls": "-"},
    {"name": "mixed_colordisp",    "label": "Mixed-ColorDisp",    "color": color_palette[4], "ls": "-"},
]

SCENE_BUDGETS = {
    "hotdog":  [40000, 80000, 160000, 320000, 640000],
    "ship":    [40000, 80000, 160000, 320000, 640000],
    "bicycle": [400000, 640000, 800000, 1600000, 3200000],
}

INPUT_DIR = Path("output/rd_sweep_20260731")
OUTPUT_DIR = Path("plots/rd_sweep_20260731")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = {"PSNR": "PSNR (dB)", "SSIM": "SSIM"}


def read_metric(scene, policy, budget, metric):
    """Final-checkpoint metric, or None if the config is missing/truncated."""
    f = INPUT_DIR / scene / f"{policy}_{budget}_occlusion" / "results_lmg.json"
    if not f.exists():
        return None
    data = json.load(open(f))
    key = sorted(data, key=lambda k: int(k.split("_")[1]))[-1]
    # 4 rounds x 8000 iters -- anything short of 32000 is an incomplete run, not a datapoint.
    if int(key.split("_")[1]) != 32000:
        return None
    return data[key][metric]


for metric, ylabel in METRICS.items():
    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 2.7, figsize[1] * 1.05))

    for ax, scene in zip(axes, SCENE_BUDGETS):
        for policy in POLICIES:
            xs, ys = [], []
            for budget in SCENE_BUDGETS[scene]:
                v = read_metric(scene, policy["name"], budget, metric)
                if v is not None:
                    xs.append(budget)
                    ys.append(v)
            if xs:
                ax.plot(xs, ys, marker="o", ms=5, lw=2,
                        color=policy["color"], ls=policy["ls"], label=policy["label"], zorder=2)
        ax.set_xscale("log")
        ax.set_xticks(SCENE_BUDGETS[scene])
        ax.set_xticklabels([f"{b // 1000}k" if b < 1000000 else f"{b / 1e6:g}M"
                            for b in SCENE_BUDGETS[scene]], fontsize=14)
        ax.minorticks_off()
        ax.set_xlabel("Budget (\\#GS)" if plt.rcParams["text.usetex"] else "Budget (#GS)", fontsize=17)
        ax.set_title(scene.capitalize(), fontsize=19)
        ax.tick_params(axis="y", labelsize=15)
        ax.grid(alpha=0.3, zorder=1)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(f"Quality in {ylabel}", fontsize=19)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=14,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.10))
    fig.set_constrained_layout(True)

    base = OUTPUT_DIR / f"{metric}_rd_sweep"
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.eps", format="eps", bbox_inches="tight")
    print(f"Wrote {base}.{{png,eps}}")
    plt.close(fig)

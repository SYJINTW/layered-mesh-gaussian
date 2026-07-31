"""
Style constants verbatim from plotter/plotting.ipynb cell 3 (this repo's own convention).
Layout: 3 per-scene panels (the first-round layout), 5 color groups per user's explicit rule:
  grey        = old geometric baselines (uniform, area, planarity2)
  own color   = distortion (2D baseline/reference)
  own color   = vertex_color_disp2 (new 3D cue)
  own color   = mixed_area family (all 3 weight variants share it)
  own color   = mixed_colordisp family (all 3 weight variants share it)
screen_footprint dropped (policy deleted from scene/budgeting.py). No Average group.
"""
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ------------------------------- setting start (plotter/plotting.ipynb cell 3, verbatim) ------------------------------ #
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
errorbar_color = "#3A3A3A"
csfont = {'family': 'Times New Roman', 'serif': 'Times', 'size': 23}
plt.rc('text', usetex=True)
plt.rc('font', **csfont)
bar_width = 0.4
bar_btw_space = 0.04
bar_space = 0.2
err_lw = 1.5
err_capsize = 4
err_capthick = 1.5
figsize = (6.4, 4.8)
# -------------------------------- setting end ------------------------------- #

SCENE_BUDGETS = {"hotdog": 32000, "ship": 32000, "bicycle": 320000}
SCENE_NAME_LIST = ["hotdog", "ship", "bicycle"]

GREY = color_palette[7]
YELLOW = color_palette[8]
C_DISTORTION = color_palette[0]
C_VCD = color_palette[3]
C_MIXED_AREA = color_palette[2]
C_MIXED_CD = color_palette[4]

POLICIES = [
    {"name": "uniform",              "label": "Uniform",           "color": GREY},
    {"name": "planarity2",           "label": "Planarity",         "color": GREY},
    {"name": "area",                 "label": "Area",              "color": YELLOW},
    {"name": "distortion",           "label": "Distortion",        "color": C_DISTORTION},
    {"name": "vertex_color_disp2",   "label": "Vertex Color Disp.", "color": C_VCD},
    {"name": "mixed_area_v1g3",      "label": "Mixed-Area 25/75",  "color": C_MIXED_AREA},
    {"name": "mixed_area",           "label": "Mixed-Area 50/50",  "color": C_MIXED_AREA},
    {"name": "mixed_area_v3g1",      "label": "Mixed-Area 75/25",  "color": C_MIXED_AREA},
    {"name": "mixed_colordisp_v1g3", "label": "Mixed-CD 25/75",    "color": C_MIXED_CD},
    {"name": "mixed_colordisp",      "label": "Mixed-CD 50/50",    "color": C_MIXED_CD},
    {"name": "mixed_colordisp_v3g1", "label": "Mixed-CD 75/25",    "color": C_MIXED_CD},
]

ITERATION = "ours_32000"
INPUT_DIR = Path("output/policy_sweep_20260727")
OUTPUT_DIR = Path("output/policy_sweep_20260727/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = {
    "PSNR": {"ylabel": "PSNR (dB)"},
    "SSIM": {"ylabel": "SSIM"},
}

LEGEND_HANDLES = [
    plt.Rectangle((0, 0), 1, 1, color=GREY),
    plt.Rectangle((0, 0), 1, 1, color=YELLOW),
    plt.Rectangle((0, 0), 1, 1, color=C_DISTORTION),
    plt.Rectangle((0, 0), 1, 1, color=C_VCD),
    plt.Rectangle((0, 0), 1, 1, color=C_MIXED_AREA),
    plt.Rectangle((0, 0), 1, 1, color=C_MIXED_CD),
]
LEGEND_LABELS = ["Old baselines", "Area", "Distortion", "Vertex Color Disp.", "Mixed-Area", "Mixed-ColorDisp"]

for metric_key, metric_info in METRICS.items():
    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 2.7, figsize[1] * 1.15), sharey=False)

    for ax, scene in zip(axes, SCENE_NAME_LIST):
        budget = SCENE_BUDGETS[scene]
        means, stderrs, colors, labels = [], [], [], []
        for policy in POLICIES:
            policy_file = INPUT_DIR / scene / f"{policy['name']}_{budget}_occlusion" / "per_view_gs_mesh.json"
            data = json.load(open(policy_file))
            metric_data = data[ITERATION][metric_key]
            values = [v for v in metric_data.values() if v != -1.0]
            means.append(np.mean(values))
            stderrs.append(np.std(values) / np.sqrt(len(values)))
            colors.append(policy["color"])
            labels.append(policy["label"])

        x = np.arange(len(POLICIES))
        ax.bar(x, means, 0.7, yerr=stderrs, color=colors,
               error_kw={"elinewidth": err_lw, "capthick": err_capthick, "capsize": err_capsize},
               zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=12)
        ax.set_title(scene.capitalize(), fontsize=19)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_axisbelow(True)
        lo = min(m - e for m, e in zip(means, stderrs))
        hi = max(m + e for m, e in zip(means, stderrs))
        pad = (hi - lo) * 0.5
        ax.set_ylim(lo - pad, hi + pad)

    axes[0].set_ylabel(f"Quality in {metric_info['ylabel']}", fontsize=19)

    fig.legend(LEGEND_HANDLES, LEGEND_LABELS, loc="upper center", ncol=6, fontsize=14,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.12))
    fig.set_constrained_layout(True)

    base_name = f"{metric_key}_policy_bar"
    fig.savefig(OUTPUT_DIR / f"{base_name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{base_name}.eps", format="eps", bbox_inches="tight")
    print(f"Wrote {OUTPUT_DIR / base_name}.{{png,eps}}")
    plt.close(fig)

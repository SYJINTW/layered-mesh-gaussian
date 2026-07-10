#
# Training-loss curves for the schedule ablation (linear/quadratic/random splat
# growth, output/2026-07-09/*, orchestrator-based multiround runs) vs the
# single-round baseline (output/ablation_single_randalpha/*, legacy
# train_progressive.py). Same lowpass+PSNR-vs-checkpoint format as
# plot_ablation_loss.py. Random schedule averages across all available seeds
# for that scene (5 for hotdog, 1 for ship/bicycle).
#
#   conda run -n lmg python plotter/plot_schedule_loss.py
#
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

SCHEDULE_DIRS = {
    "hotdog": {
        "linear": ["output/2026-07-09/hotdog_linear"],
        "quadratic": ["output/2026-07-09/hotdog_quadratic"],
        "random": [f"output/2026-07-09/hotdog_random_seed{i}" for i in range(1, 6)],
    },
    "ship": {
        "linear": ["output/2026-07-09/ship_linear"],
        "quadratic": ["output/2026-07-09/ship_quadratic"],
        "random": ["output/2026-07-09/ship_random_seed42"],
    },
    "bicycle": {
        "linear": ["output/2026-07-09/bicycle_linear"],
        "quadratic": ["output/2026-07-09/bicycle_quadratic"],
        "random": ["output/2026-07-09/bicycle_random_seed42"],
    },
}
COLORS = {"single-round baseline": "k", "linear": "tab:blue",
          "quadratic": "tab:orange", "random": "tab:green"}
BASELINE_EXP = "ablation_single_randalpha"
WIN = 200
OUT_DIR = "output/meeting_2026-07-10"


def load_orchestrator_loss(run_dir):
    files = sorted(glob.glob(f"{run_dir}/training_metrics_round*.json"),
                    key=lambda p: int(p.split("_round")[1].split(".json")[0]))
    if not files:
        return None
    gi, gl, offset = [], [], 0
    for f in files:
        d = json.load(open(f))
        gi += [offset + i for i in d["iteration"]]
        gl += d["loss"]
        offset += max(d["iteration"])
    return np.array(gi), np.array(gl)


def load_orchestrator_psnr(run_dir):
    f = f"{run_dir}/round_summary.json"
    if not os.path.exists(f):
        return []
    rs = json.load(open(f))
    return sorted((e["iteration"], e["psnr"]) for e in rs)


def load_baseline_loss(scene):
    base = sorted(glob.glob(f"output/{BASELINE_EXP}/{scene}/*"))
    if not base:
        return None
    files = sorted(glob.glob(f"{base[0]}/iteration_*/training_metrics.json"),
                    key=lambda p: int(p.split("iteration_")[1].split("/")[0]))
    if not files:
        return None
    gi, gl = [], []
    for fpath in files:
        off = int(fpath.split("iteration_")[1].split("/")[0])
        d = json.load(open(fpath))
        gi += [off + i for i in d["iteration"]]
        gl += d["loss"]
    return np.array(gi), np.array(gl)


def load_baseline_psnr(scene):
    pts = []
    for f in glob.glob(f"output/{BASELINE_EXP}/{scene}/*/iteration_*/results_lmg.json"):
        d = json.load(open(f))
        for k, v in d.items():
            pts.append((int(k.split("_")[1]), v["PSNR"]))
    return sorted(set(pts))


for scene, configs in SCHEDULE_DIRS.items():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    any_data = False

    r = load_baseline_loss(scene)
    if r is not None:
        any_data = True
        gi, gl = r
        sm = uniform_filter1d(gl, size=min(WIN, len(gl)))
        ax1.plot(gi, gl, color="k", alpha=0.12, lw=0.5)
        ax1.plot(gi, sm, color="k", lw=1.8, label="single-round baseline")
    p = load_baseline_psnr(scene)
    if p:
        xs, ys = zip(*p)
        ax2.plot(xs, ys, "o-", color="k", label="single-round baseline")

    for label, dirs in configs.items():
        color = COLORS[label]
        losses = [x for x in (load_orchestrator_loss(d) for d in dirs) if x is not None]
        if losses:
            any_data = True
            gi = losses[0][0]
            gl = losses[0][1] if len(losses) == 1 else np.mean([l[1] for l in losses], axis=0)
            sm = uniform_filter1d(gl, size=min(WIN, len(gl)))
            ax1.plot(gi, gl, color=color, alpha=0.12, lw=0.5)
            n_tag = f" (n={len(losses)} seeds)" if len(losses) > 1 else ""
            ax1.plot(gi, sm, color=color, lw=1.8, label=label + n_tag)

        psnrs = [x for x in (load_orchestrator_psnr(d) for d in dirs) if x]
        if psnrs:
            xs = [pt[0] for pt in psnrs[0]]
            ys = np.mean([[pt[1] for pt in p] for p in psnrs], axis=0)
            ax2.plot(xs, ys, "o-", color=color, label=label)

    if not any_data:
        print(f"[SKIP] {scene}: no data found")
        plt.close(fig)
        continue
    ax1.set(xlabel="iteration", ylabel="training loss",
            title=f"{scene}: training loss (lowpass w={WIN})")
    ax1.set_ylim(bottom=0); ax1.grid(True, ls="--", alpha=0.4); ax1.legend()
    ax2.set(xlabel="checkpoint iteration", ylabel="test PSNR (dB)",
            title=f"{scene}: schedule vs single-round baseline")
    ax2.grid(True, ls="--", alpha=0.4); ax2.legend()
    fig.tight_layout()
    out = f"{OUT_DIR}/{scene}_schedule_loss.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[wrote] {out}")

#
# Plot ablation: smoothed training-loss curves + PSNR-vs-checkpoint, 4 configs
# (single_rand baseline, single_fixed, prog_rand, prog_fixed). Restored/adapted
# from the 2026-06-25 version (git e1f5baa, later deleted) -- same format as
# output/old_0625_plots/ablation_*.png, pointed at the current exp_ablation.sh
# rerun's output dirs. Missing configs/scenes (still running or not started)
# are skipped, not errored on -- rerun once more data lands.
#
#   conda run -n lmg python plotter/plot_ablation_loss.py
#
import json
import glob

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

SCENES = ["hotdog", "ship", "bicycle"]
CONFIGS = {  # label -> output exp dir
    "single_rand (baseline)": "ablation_single_randalpha",
    "single_fixed":           "ablation_single_fixedalpha",
    "prog_rand":              "ablation_prog_randalpha",
    "prog_fixed":             "ablation_prog_fixedalpha",
}
COLORS = {"single_rand (baseline)": "k", "single_fixed": "tab:blue",
          "prog_rand": "tab:orange", "prog_fixed": "tab:red"}
WIN = 200  # lowpass moving-average window (iters)
OUT_DIR = "output/meeting_2026-07-10"


def load_loss(exp, scene):
    """Return (global_iter, loss) concatenating prog rounds; None if missing."""
    base = sorted(glob.glob(f"output/{exp}/{scene}/*"))
    if not base:
        return None
    rounds = sorted(glob.glob(f"{base[0]}/iteration_*/training_metrics.json"),
                     key=lambda p: int(p.split("iteration_")[1].split("/")[0]))
    if not rounds:
        return None
    gi, gl = [], []
    for r in rounds:
        off = int(r.split("iteration_")[1].split("/")[0])
        d = json.load(open(r))
        gi += [off + i for i in d["iteration"]]
        gl += d["loss"]
    return np.array(gi), np.array(gl)


def load_psnr(exp, scene):
    """Return sorted [(global_iter, psnr)] from results_lmg.json files."""
    pts = []
    for f in glob.glob(f"output/{exp}/{scene}/*/iteration_*/results_lmg.json"):
        d = json.load(open(f))
        for k, v in d.items():  # k like 'ours_32000'
            pts.append((int(k.split("_")[1]), v["PSNR"]))
    return sorted(set(pts))


for scene in SCENES:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    any_data = False
    for label, exp in CONFIGS.items():
        c = COLORS[label]
        r = load_loss(exp, scene)
        if r is not None:
            any_data = True
            gi, gl = r
            sm = uniform_filter1d(gl, size=min(WIN, len(gl)))
            ax1.plot(gi, gl, color=c, alpha=0.12, lw=0.5)      # raw, faint
            ax1.plot(gi, sm, color=c, lw=1.8, label=label)     # smoothed
        p = load_psnr(exp, scene)
        if p:
            xs, ys = zip(*p)
            ax2.plot(xs, ys, "o-", color=c, label=label)
    if not any_data:
        print(f"[SKIP] {scene}: no configs have training_metrics.json yet")
        plt.close(fig)
        continue
    ax1.set(xlabel="iteration", ylabel="training loss",
            title=f"{scene}: training loss (lowpass w={WIN})")
    ax1.set_ylim(bottom=0); ax1.grid(True, ls="--", alpha=0.4); ax1.legend()
    ax2.set(xlabel="checkpoint iteration", ylabel="test PSNR (dB)",
            title=f"{scene}: PSNR vs checkpoint")
    ax2.grid(True, ls="--", alpha=0.4); ax2.legend()
    fig.tight_layout()
    out = f"{OUT_DIR}/{scene}_training_loss.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[wrote] {out}")

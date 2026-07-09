"""Plot ablation: smoothed training-loss curves + PSNR-vs-checkpoint, 4 configs."""
import json, glob, os
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
    for label, exp in CONFIGS.items():
        c = COLORS[label]
        r = load_loss(exp, scene)
        if r is not None:
            gi, gl = r
            sm = uniform_filter1d(gl, size=min(WIN, len(gl)))
            ax1.plot(gi, gl, color=c, alpha=0.12, lw=0.5)      # raw, faint
            ax1.plot(gi, sm, color=c, lw=1.8, label=label)     # smoothed
        p = load_psnr(exp, scene)
        if p:
            xs, ys = zip(*p)
            ax2.plot(xs, ys, "o-", color=c, label=label)
    ax1.set(xlabel="iteration", ylabel="training loss",
            title=f"{scene}: training loss (lowpass w={WIN})")
    ax1.set_ylim(bottom=0); ax1.grid(True, ls="--", alpha=0.4); ax1.legend()
    ax2.set(xlabel="checkpoint iteration", ylabel="test PSNR (dB)",
            title=f"{scene}: PSNR vs checkpoint")
    ax2.grid(True, ls="--", alpha=0.4); ax2.legend()
    fig.tight_layout()
    out = f"ablation_{scene}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")

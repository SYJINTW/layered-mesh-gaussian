#!/usr/bin/env python3
"""Standalone, read-only analysis of allocation-policy mesh cues across scenes.

Not wired into any production path (scene/budgeting.py, train.py, warmup.py, etc.)
and does not write any policy cache. Purpose: characterize the FULL distribution
(not just top-k overlap) of existing + candidate mesh-native allocation cues,
across multiple scenes, before deciding a fusion operator for the "mixed"
alloc policy (.claude/todo.md).

Cues:
  - area, planarity{1,2,3}: recomputed FRESH here (vectorized, sparse hop-adjacency),
    not loaded from any old policy/*/weights.npy cache -- those caches may have been
    computed against a stale mesh-loading path or different hop/focus params.
  - distortion: the one expensive-to-recompute cue (needs a full multi-view render
    pass); reused from the existing cached policy/mesh_milo/tri_*/distortion/weights.npy,
    trusted fresh as-is.
  - vertex_color_dispersion (NEW, mesh-only): local neighborhood dispersion of each
    triangle's own baked vertex color -- same hop-neighborhood machinery as planarity,
    but on color instead of normals (not the same MRL formula -- colors aren't unit
    vectors on S^2, so this is neighborhood color VARIANCE, not mean-resultant-length).
  - screen_footprint (NEW, mesh+camera, EXPLORATORY/CAVEATED): world-space triangle
    area divided by mean squared distance to training-camera centers. Assumes training
    camera placement is representative of real downstream viewership -- an assumption
    the user explicitly flagged as unverified. Report, don't treat as equal-footing
    evidence versus the mesh-only cues above.
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import trimesh
import scipy.sparse as sp
from scipy.stats import gaussian_kde, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text, qvec2rotmat
from scene.budgeting import AreaBasedBudgetingPolicy

EPS = 1e-8
OUT_DIR = Path("/tmp/claude-1013/-mnt-data1-samk-NEU-LMG-Codebase/01cfdc9c-577d-4eaf-a7e0-cd480e893d88/scratchpad/mesh_cue_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENES = {
    "hotdog": dict(
        mesh_path=str(REPO_ROOT.parent / "sorted_dataset/milo_meshes/hotdog/hotdog.ply"),
        dataset_path=str(REPO_ROOT.parent / "sorted_dataset/hotdog"),
        cam_type="blender",
        distortion_cache=str(REPO_ROOT / "data/weights/hotdog/policy/mesh_milo/tri_1002315/distortion/weights.npy"),
    ),
    "ship": dict(
        mesh_path=str(REPO_ROOT.parent / "sorted_dataset/milo_meshes/ship/ship.ply"),
        dataset_path=str(REPO_ROOT.parent / "sorted_dataset/ship"),
        cam_type="blender",
        distortion_cache=str(REPO_ROOT / "data/weights/ship/policy/mesh_milo/tri_1445109/distortion/weights.npy"),
    ),
    "bicycle": dict(
        mesh_path=str(REPO_ROOT.parent / "sorted_dataset/milo_meshes/bicycle-dw50/bicycle-dw50.ply"),
        dataset_path=str(REPO_ROOT.parent / "sorted_dataset/bicycle"),
        cam_type="colmap",
        distortion_cache=str(REPO_ROOT / "data/weights/bicycle-dw50/policy/mesh_milo/tri_8846590/distortion/weights.npy"),
    ),
}


# --------------------------------------------------------------------------- #
# Vectorized hop-neighborhood (same "within <=hops BFS steps" definition as
# PlanarityBasedBudgetingPolicy._compute_planarity_mrl's neighborhood(), just
# via sparse boolean reachability instead of a per-face Python BFS loop -- the
# per-face loop is too slow to run interactively on an 8.8M-face mesh).
# --------------------------------------------------------------------------- #
def build_face_adjacency(mesh) -> sp.csr_matrix:
    fa = mesh.face_adjacency
    n = mesh.faces.shape[0]
    rows = np.concatenate([fa[:, 0], fa[:, 1]])
    cols = np.concatenate([fa[:, 1], fa[:, 0]])
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def hop_reachability(A: sp.csr_matrix, hops: int) -> sp.csr_matrix:
    n = A.shape[0]
    R = sp.identity(n, format="csr", dtype=np.float32)
    for _ in range(hops):
        R = R + R.dot(A)
        R.data[:] = 1.0
        R.eliminate_zeros()
    return R


def neighborhood_mean(R: sp.csr_matrix, signal: np.ndarray) -> np.ndarray:
    count = np.asarray(R.sum(axis=1)).flatten()
    total = R.dot(signal)
    return total / np.maximum(count, 1.0)[:, None]


# --------------------------------------------------------------------------- #
# Cue computation
# --------------------------------------------------------------------------- #
def compute_area(mesh) -> np.ndarray:
    return AreaBasedBudgetingPolicy(mesh=mesh).weights


def compute_planarity(mesh, A: sp.csr_matrix, hops: int) -> np.ndarray:
    normals = mesh.face_normals.astype(np.float32)
    n_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    n_norm[n_norm == 0] = 1.0
    normals = normals / n_norm
    R = hop_reachability(A, hops)
    mean_n = neighborhood_mean(R, normals)
    mrl = np.clip(np.linalg.norm(mean_n, axis=1), 0.0, 1.0)
    return np.maximum(1.0 - mrl, EPS).astype(np.float32)  # focus='nonplanar'


def compute_vertex_color_dispersion(mesh, A: sp.csr_matrix, hops: int) -> np.ndarray:
    vertex_colors = np.asarray(mesh.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
    face_colors = vertex_colors[mesh.faces].mean(axis=1)  # (F, 3)
    R = hop_reachability(A, hops)
    neighbor_mean = neighborhood_mean(R, face_colors)
    dispersion = np.linalg.norm(face_colors - neighbor_mean, axis=1)
    return np.maximum(dispersion, EPS).astype(np.float32)


def load_colmap_camera_centers(dataset_path: str) -> np.ndarray:
    sparse_dir = os.path.join(dataset_path, "sparse", "0")
    bin_path = os.path.join(sparse_dir, "images.bin")
    txt_path = os.path.join(sparse_dir, "images.txt")
    cam_extrinsics = read_extrinsics_binary(bin_path) if os.path.exists(bin_path) else read_extrinsics_text(txt_path)
    centers = []
    for key in cam_extrinsics:
        extr = cam_extrinsics[key]
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        # camera center in world space: C = -R @ T  (matches getWorld2View2 inverse's translation row)
        C = -R @ T
        centers.append(C)
    return np.stack(centers, axis=0)


def load_blender_camera_centers(dataset_path: str, transformsfile: str = "transforms_train.json") -> np.ndarray:
    with open(os.path.join(dataset_path, transformsfile)) as f:
        contents = json.load(f)
    centers = []
    for frame in contents["frames"]:
        c2w = np.array(frame["transform_matrix"])
        centers.append(c2w[:3, 3])  # camera-to-world translation column IS the camera center
    return np.stack(centers, axis=0)


def compute_screen_footprint(mesh, cam_centers: np.ndarray) -> np.ndarray:
    tri_centroids = mesh.triangles_center  # (F, 3)
    areas = mesh.area_faces
    # mean_c(||t-c||^2) = ||t||^2 - 2 t.mean(c) + mean(||c||^2) -- exact, avoids
    # materializing the (F, C) dense distance matrix (14GB+ for bicycle's 8.8M
    # faces x 194 cams if done the naive broadcast way).
    mean_c = cam_centers.mean(axis=0)
    mean_c2 = (cam_centers ** 2).sum(axis=1).mean()
    mean_d2 = (tri_centroids ** 2).sum(axis=1) - 2.0 * tri_centroids.dot(mean_c) + mean_c2
    footprint = areas / np.maximum(mean_d2, EPS)
    return np.maximum(footprint, EPS).astype(np.float32)


# --------------------------------------------------------------------------- #
# Stats / plotting
# --------------------------------------------------------------------------- #
def print_stats(scene: str, cue: str, w: np.ndarray):
    cv = w.std() / w.mean() if w.mean() > 0 else float("nan")
    print(f"[{scene:8s}] {cue:22s} min={w.min():.6g} max={w.max():.6g} "
          f"mean={w.mean():.6g} std={w.std():.6g} cv={cv:.4f}")


def plot_distributions(scene: str, cues: dict):
    n = len(cues)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (name, w) in zip(axes, cues.items()):
        wl = np.log10(w + EPS)
        ax.hist(wl, bins=80, density=True, alpha=0.45, color="steelblue")
        try:
            kde = gaussian_kde(wl)
            xs = np.linspace(wl.min(), wl.max(), 300)
            ax.plot(xs, kde(xs), color="darkred", lw=2.5)
        except Exception as e:
            ax.text(0.5, 0.5, f"KDE failed: {e}", transform=ax.transAxes, ha="center", fontsize=11)
        ax.set_title(name, fontsize=20, fontweight="bold", pad=10)
        ax.set_xlabel("log10(weight)", fontsize=15)
        ax.set_ylabel("density", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(alpha=0.25)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"{scene}: per-cue weight distribution (log10 scale)", fontsize=24, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / f"{scene}_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_correlation_heatmaps(scene: str, cues: dict):
    names = list(cues.keys())
    n = len(names)
    mat_pearson = np.eye(n)
    mat_spearman = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            wi, wj = cues[names[i]], cues[names[j]]
            r = np.corrcoef(wi, wj)[0, 1]
            rho, _ = spearmanr(wi, wj)
            mat_pearson[i, j] = mat_pearson[j, i] = r
            mat_spearman[i, j] = mat_spearman[j, i] = rho

    cell = 1.15  # inches per matrix cell -- scales the figure so text never gets crammed
    fig, axes = plt.subplots(1, 2, figsize=(2 * (n * cell + 2.5), n * cell + 2.0))
    for ax, mat, title in [(axes[0], mat_pearson, "Pearson r"), (axes[1], mat_spearman, "Spearman rho")]:
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=15)
        ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=15)
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                weight = "bold" if abs(val) > 0.3 and i != j else "normal"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=16,
                        color=color, fontweight=weight)
        ax.set_title(title, fontsize=22, pad=14)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046)
        cbar.ax.tick_params(labelsize=13)
    fig.suptitle(f"{scene}: full-population cue correlation (not top-k truncated)", fontsize=22, y=1.03)
    fig.tight_layout()
    out = OUT_DIR / f"{scene}_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


CACHE_DIR = OUT_DIR / "_cue_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def compute_cues(scene: str, cfg: dict) -> dict:
    """All the expensive per-scene work (mesh load, sparse hop-adjacency, BFS-equivalent
    reachability -- the part that takes ~9min on bicycle's 8.8M faces). Cached to disk
    keyed by scene name; re-run only wipes plots, never silently re-triggers this."""
    mesh = trimesh.load(cfg["mesh_path"], process=False)
    A = build_face_adjacency(mesh)

    cues = {}
    cues["area"] = compute_area(mesh)
    for hops in (1, 2, 3):
        cues[f"planarity{hops}"] = compute_planarity(mesh, A, hops)
    cues["vertex_color_disp2"] = compute_vertex_color_dispersion(mesh, A, hops=2)

    dist_path = cfg["distortion_cache"]
    if os.path.exists(dist_path):
        cues["distortion"] = np.load(dist_path).astype(np.float32)
        if len(cues["distortion"]) != mesh.faces.shape[0]:
            print(f"  [WARNING] distortion cache length {len(cues['distortion'])} != "
                  f"mesh face count {mesh.faces.shape[0]}, dropping cue for this scene")
            del cues["distortion"]
    else:
        print(f"  [WARNING] no distortion cache at {dist_path}, skipping distortion for {scene}")

    try:
        if cfg["cam_type"] == "blender":
            centers = load_blender_camera_centers(cfg["dataset_path"])
        else:
            centers = load_colmap_camera_centers(cfg["dataset_path"])
        cues["screen_footprint*"] = compute_screen_footprint(mesh, centers)
    except Exception as e:
        print(f"  [WARNING] screen_footprint failed ({e}), skipping (exploratory cue only)")

    return cues


def load_or_compute_cues(scene: str, cfg: dict, recompute: bool) -> dict:
    cache_path = CACHE_DIR / f"{scene}.npz"
    if cache_path.exists() and not recompute:
        print(f"  [cache] loading cues from {cache_path} (pass --recompute to force)")
        npz = np.load(cache_path)
        return {name: npz[name] for name in npz.files}

    cues = compute_cues(scene, cfg)
    np.savez(cache_path, **cues)
    print(f"  [cache] wrote {cache_path}")
    return cues


def main():
    recompute = "--recompute" in sys.argv
    for scene, cfg in SCENES.items():
        print(f"\n=== {scene} ===")
        cues = load_or_compute_cues(scene, cfg, recompute)

        for name, w in cues.items():
            print_stats(scene, name, w)

        plot_distributions(scene, cues)
        plot_correlation_heatmaps(scene, cues)

    print(f"\n(* = train-camera-conditioned, exploratory only -- generalization to real "
          f"downstream viewership NOT verified)")
    print(f"All figures written under {OUT_DIR}")


if __name__ == "__main__":
    main()

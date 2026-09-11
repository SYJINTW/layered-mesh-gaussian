"""Generate conformance vectors for the Rust decoder.

Builds tiny synthetic scenes through the real LMGModel / LMGModelHover code,
writes them out as LMG Full, deflates them to Lean, and records the derived
fields (xyz / scaling / rotation) that Python produced. The Rust test rebuilds
those from the Lean bundle alone and must match.

Synthetic rather than a real checkpoint because the vectors are committed: the
smallest real scene carries a 20 MB mesh, and no real checkpoint exercises the
hover variant at all.

    python -m lmg_format.make_vectors rust/lmg-format/tests/vectors
"""
import os
import sys

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from .codec import deflate

SH_DEGREE = 3


def _write_mesh(path, verts, faces):
    """Binary little-endian PLY, the shape trimesh emits."""
    v = np.array([tuple(p) for p in verts],
                 dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    f = np.array([(list(map(int, t)),) for t in faces],
                 dtype=[("vertex_indices", "i4", (3,))])
    PlyData([PlyElement.describe(v, "vertex"),
             PlyElement.describe(f, "face")], text=False).write(path)


def _build_full(out_dir, mesh_path, budgets, pool, variant, seed=0):
    """Write a Full checkpoint (point_cloud.ply + model_params.pt) via the model."""
    from games.mesh_splatting.scene.gaussian_mesh_model import LMGModel, LMGModelHover
    from renderer.mesh_loader import mesh_loader

    rng = np.random.RandomState(seed)
    budgets = np.asarray(budgets, dtype=np.int64)
    n = int(budgets.sum())
    tri_idx = np.repeat(np.arange(budgets.shape[0]), budgets)
    starts = np.cumsum(budgets) - budgets
    alpha_idx = np.arange(n) - np.repeat(starts, budgets)

    mesh = mesh_loader.load_transformed_mesh(mesh_path)
    verts = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)
    faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.int64)

    model = (LMGModelHover if variant == "hover" else LMGModel)(SH_DEGREE)
    model.vertices = torch.nn.Parameter(verts)
    model.faces = faces
    model.triangles = verts[faces]
    model.triangle_indices = torch.from_numpy(tri_idx)
    model.alpha_indices = torch.from_numpy(alpha_idx)
    model.num_splats_per_triangle = budgets.astype(np.int32)
    model._alpha = torch.nn.Parameter(torch.from_numpy(pool[alpha_idx]).float())
    model._scale = torch.nn.Parameter(torch.ones((n, 1), dtype=torch.float32))
    model._opacity = torch.nn.Parameter(
        torch.from_numpy(rng.randn(n, 1).astype(np.float32)))
    model._features_dc = torch.nn.Parameter(
        torch.from_numpy(rng.randn(n, 1, 3).astype(np.float32)))
    model._features_rest = torch.nn.Parameter(
        torch.from_numpy(rng.randn(n, (SH_DEGREE + 1) ** 2 - 1, 3).astype(np.float32)))
    if variant == "hover":
        # Spans both branches of the asymmetric tanh activation.
        model._hover = torch.nn.Parameter(
            torch.from_numpy(rng.uniform(-0.5, 0.5, (n, 1)).astype(np.float32)))

    os.makedirs(out_dir, exist_ok=True)
    model.save_ply(os.path.join(out_dir, "point_cloud.ply"))
    return n


def _expected(full_dir, out_path):
    """Pull the derived columns Python wrote, as a safetensors the Rust test reads."""
    from safetensors.numpy import save_file
    v = PlyData.read(os.path.join(full_dir, "point_cloud.ply")).elements[0]
    cols = lambda names: np.stack([np.asarray(v[c]) for c in names], axis=1).astype(np.float32)
    save_file({
        "xyz": cols(["x", "y", "z"]),
        "scaling": cols(["scale_0", "scale_1", "scale_2"]),
        "rotation": cols(["rot_0", "rot_1", "rot_2", "rot_3"]),
    }, out_path)


def main(out_root):
    verts = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.3], [0.2, -0.8, 0.5], [-0.6, 0.4, -0.7],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 3, 2], [0, 4, 1], [2, 5, 0]], dtype=np.int64)
    budgets = [2, 1, 3, 0]  # includes an empty face; N = 6
    pool = np.array([[0.5, 0.3, 0.2],
                     [0.1, 0.8, 0.1],
                     [0.25, 0.25, 0.5]], dtype=np.float32)

    for variant in ("binding", "hover"):
        case = os.path.join(out_root, variant)
        os.makedirs(case, exist_ok=True)
        mesh_path = os.path.join(case, "src_mesh.ply")
        _write_mesh(mesh_path, verts, faces)

        full = os.path.join(case, "full")
        n = _build_full(full, mesh_path, budgets, pool, variant)
        # static_alpha.pt at the run root is how deflate detects pool mode.
        torch.save(torch.from_numpy(pool), os.path.join(case, "static_alpha.pt"))

        bundle = os.path.join(case, "bundle.lmg")
        meta = deflate(full, bundle, mesh_path)
        _expected(full, os.path.join(case, "expected.safetensors"))
        os.remove(mesh_path)
        print("%-8s N=%d barycentric=%s scale=%s -> %s"
              % (variant, n, meta["barycentric_mode"], meta["scale_mode"], bundle))



def from_full(full_dir, mesh_path, out_case, sh_degree=SH_DEGREE):
    """Real-data vector: deflate an existing Full checkpoint and record its derived fields.

    Not committed (the meshes are tens of MB); point the Rust test at it with
    LMG_REAL_VECTOR=<out_case>.
    """
    import shutil
    os.makedirs(out_case, exist_ok=True)
    bundle = os.path.join(out_case, "bundle.lmg")
    meta = deflate(full_dir, bundle, mesh_path)
    _expected(full_dir, os.path.join(out_case, "expected.safetensors"))
    shutil.copy2(os.path.join(full_dir, "model_params.pt"),
                 os.path.join(out_case, "source_model_params.pt"))
    print("%s -> %s (N=%s, %s/%s)" % (full_dir, bundle, meta["num_splats"],
                                      meta["barycentric_mode"], meta["scale_mode"]))
    return bundle

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rust/lmg-format/tests/vectors")

"""LMG Full <-> LMG Lean transcoder.

Full  = mesh.ply + point_cloud.ply (62 f32/splat) + model_params.pt (pickle)
Lean  = <name>.lmg/{mesh.ply, scene.safetensors}

Lean drops everything derivable from the mesh: xyz, normals, scaling, rotation,
triangle_indices, alpha_indices, and -- when the encoder verifies they are
unchanged from init -- _alpha and _scale. num_splats_per_triangle is stored
sparsely; it is dense over every face but ~1% of faces carry a splat.

Lossless: every stored array round-trips bit-exact. Derived fields are
reproduced by the formulas in gaussian_mesh_model.py, not byte-compared.
"""
import hashlib
import json
import os

import numpy as np
import torch
from plyfile import PlyData
from safetensors.numpy import load_file, save_file

FORMAT_VERSION = "1"
PARAMS_NAME = "scene.safetensors"
MESH_NAME = "mesh.ply"

# Keys that only GaussianMeshModel (gs_mesh) writes; LMGModel never does.
GS_MESH_KEYS = {"point_cloud", "triangles", "vertices", "faces"}


class UnsupportedVariant(Exception):
    pass


class NotDeduplicable(Exception):
    """A Full checkpoint violates a structural assumption Lean relies on."""


# --------------------------------------------------------------------------- #
#                                   helpers                                    #
# --------------------------------------------------------------------------- #

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _read_appearance(ply_path, sh_degree=3):
    """Pull f_dc/f_rest/opacity out of a Full point_cloud.ply, in PLY field order.

    Keeping the flat PLY ordering (channel-major for f_rest) means inflate can
    write the bytes straight back out without a second transpose convention.
    """
    ply = PlyData.read(ply_path)
    v = ply.elements[0]
    n = v.count
    n_rest = 3 * ((sh_degree + 1) ** 2 - 1)

    f_dc = np.stack([v["f_dc_%d" % i] for i in range(3)], axis=1)
    f_rest = np.stack([v["f_rest_%d" % i] for i in range(n_rest)], axis=1)
    opacity = np.asarray(v["opacity"]).reshape(n, 1)
    return (f_dc.astype(np.float32),
            f_rest.astype(np.float32),
            opacity.astype(np.float32))


def _sparsify(budgets):
    """Dense [F] budget array -> (face_id, count) pairs for occupied faces."""
    face_id = np.flatnonzero(budgets).astype(np.uint32)
    counts = budgets[face_id]
    dtype = np.uint8 if counts.max(initial=0) <= 255 else np.uint16
    return face_id, counts.astype(dtype)


def _densify(face_id, counts, num_faces):
    dense = np.zeros(num_faces, dtype=np.int32)
    dense[face_id.astype(np.int64)] = counts
    return dense


def _derive_indices(budgets):
    """triangle_indices and alpha_indices, both implied by the budget array."""
    tri = np.repeat(np.arange(budgets.shape[0], dtype=np.int64), budgets)
    starts = np.cumsum(budgets, dtype=np.int64) - budgets  # exclusive prefix sum
    alpha = np.arange(tri.shape[0], dtype=np.int64) - np.repeat(starts, budgets)
    return tri, alpha


def _static_alpha_path(full_dir):
    """static_alpha.pt sits at the run root, above point_cloud/iteration_N/."""
    d = os.path.abspath(full_dir)
    while os.path.basename(d):
        cand = os.path.join(d, "static_alpha.pt")
        if os.path.isfile(cand):
            return cand
        d = os.path.dirname(d)
    return None


# --------------------------------------------------------------------------- #
#                                   deflate                                    #
# --------------------------------------------------------------------------- #

def _write_normalized_mesh(mesh_path, dst):
    """Write the mesh as the model sees it: trimesh's processed vertices/faces.

    trimesh merges near-duplicate vertices on load, so the raw file and the
    loaded mesh differ by ~1e-5 on some coordinates. That is invisible in
    linear space but `prepare_scaling_rot` takes a log, and this mesh has
    degenerate triangles (min s2 == 0.0) where log(x + 1e-8) amplifies it to
    ~4e-2. Storing the processed mesh means a decoder that reads the PLY
    plainly -- the Rust one does -- derives exactly what Python derives,
    instead of being held to a tolerance that degenerate faces cannot meet.
    """
    from renderer.mesh_loader import mesh_loader
    from plyfile import PlyElement

    mesh = mesh_loader.load_transformed_mesh(mesh_path)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    v = np.empty(verts.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    v["x"], v["y"], v["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    f = np.empty(faces.shape[0], dtype=[("vertex_indices", "i4", (3,))])
    f["vertex_indices"] = faces
    PlyData([PlyElement.describe(v, "vertex"),
             PlyElement.describe(f, "face")], text=False).write(dst)


def deflate(full_dir, out_bundle, mesh_path, link_mesh=False, sh_degree=3):
    """Full checkpoint dir -> Lean bundle. Returns the metadata dict."""
    pt_path = os.path.join(full_dir, "model_params.pt")
    ply_path = os.path.join(full_dir, "point_cloud.ply")
    params = torch.load(pt_path, map_location="cpu")

    present = GS_MESH_KEYS & set(params)
    if present:
        raise UnsupportedVariant(
            "checkpoint carries gs_mesh keys %s -- GaussianMeshModel is not "
            "supported in format v1, only LMGModel/LMGModelHover" % sorted(present)
        )

    variant = "hover" if "_hover" in params else "binding"
    budgets = _np(params["num_splats_per_triangle"]).astype(np.int64)
    alpha = _np(params["_alpha"]).astype(np.float32)
    scale = _np(params["_scale"]).astype(np.float32)
    n, f = alpha.shape[0], budgets.shape[0]

    if f >= 2 ** 32:
        raise NotDeduplicable("mesh has %d faces, exceeds u32 face ids" % f)

    # Structural assumptions. Refuse rather than silently store a wrong file.
    tri_exp, alpha_idx_exp = _derive_indices(budgets)
    if not np.array_equal(_np(params["triangle_indices"]).astype(np.int64), tri_exp):
        raise NotDeduplicable("triangle_indices is not repeat(arange(F), budgets)")
    if not np.array_equal(_np(params["alpha_indices"]).astype(np.int64), alpha_idx_exp):
        raise NotDeduplicable("alpha_indices is not concat(arange(n) per face)")

    tensors = {}
    tensors["f_dc"], tensors["f_rest"], tensors["opacity"] = _read_appearance(
        ply_path, sh_degree)
    if tensors["f_dc"].shape[0] != n:
        raise NotDeduplicable(
            "ply has %d splats, model_params.pt has %d"
            % (tensors["f_dc"].shape[0], n))

    face_id, counts = _sparsify(budgets)
    tensors["budget_face_id"], tensors["budget_count"] = face_id, counts

    # _alpha: identical to the shared canonical pool under --fixed_alpha.
    pool_path = _static_alpha_path(full_dir)
    barycentric_mode = "raw"
    if pool_path is not None:
        pool = _np(torch.load(pool_path, map_location="cpu")).astype(np.float32)
        if pool.shape[0] > alpha_idx_exp.max(initial=-1) and np.array_equal(
                alpha, pool[alpha_idx_exp]):
            barycentric_mode = "pool"
            tensors["alpha_pool"] = pool
    if barycentric_mode == "raw":
        tensors["alpha"] = alpha

    # _scale: never receives gradient (prepare_scaling_rot runs under no_grad),
    # so it stays at its init of ones -- but verify, don't assume.
    if np.all(scale == 1.0):
        scale_mode = "unit"
    else:
        scale_mode = "raw"
        tensors["scale"] = scale

    if variant == "hover":
        tensors["hover"] = _np(params["_hover"]).astype(np.float32)
    if "round_id" in params:
        rid = _np(params["round_id"]).astype(np.int64)
        if rid.max(initial=0) > 255:
            raise NotDeduplicable("round_id exceeds u8")
        tensors["round_id"] = rid.astype(np.uint8)

    os.makedirs(out_bundle, exist_ok=True)
    dst_mesh = os.path.join(out_bundle, MESH_NAME)
    if os.path.lexists(dst_mesh):
        os.remove(dst_mesh)
    if link_mesh:
        # Local shortcut (verify, big meshes): points at the raw input, so a
        # decoder that does not replicate trimesh's vertex merge sees slightly
        # different triangles. Fine for the Python round-trip, not for shipping.
        os.symlink(os.path.abspath(mesh_path), dst_mesh)
    else:
        _write_normalized_mesh(mesh_path, dst_mesh)

    meta = {
        "format_version": FORMAT_VERSION,
        "model_variant": variant,
        "sh_degree": str(sh_degree),
        "num_splats": str(n),
        "num_faces": str(f),
        "barycentric_mode": barycentric_mode,
        "scale_mode": scale_mode,
        "mesh_sha256": sha256_file(dst_mesh),
        "mesh_normalized": "false" if link_mesh else "true",
        "mesh_file": MESH_NAME,
        "source": json.dumps({
            "full_dir": os.path.abspath(full_dir),
            "mesh_path": os.path.abspath(mesh_path),
        }),
    }
    save_file(tensors, os.path.join(out_bundle, PARAMS_NAME), metadata=meta)
    return meta


# --------------------------------------------------------------------------- #
#                                    inflate                                   #
# --------------------------------------------------------------------------- #

def _rebuild_params(bundle):
    """Lean bundle -> the tensor set LMGModel.save_ply expects, plus metadata."""
    params_path = os.path.join(bundle, PARAMS_NAME)
    with open(params_path, "rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        meta = json.loads(fh.read(header_len)).get("__metadata__", {})
    t = load_file(params_path)

    n, f = int(meta["num_splats"]), int(meta["num_faces"])
    budgets = _densify(t["budget_face_id"], t["budget_count"], f)
    tri_idx, alpha_idx = _derive_indices(budgets.astype(np.int64))

    if meta["barycentric_mode"] == "pool":
        alpha = t["alpha_pool"][alpha_idx]
    else:
        alpha = t["alpha"]
    scale = np.ones((n, 1), np.float32) if meta["scale_mode"] == "unit" else t["scale"]
    return meta, t, budgets, tri_idx, alpha_idx, alpha, scale


def load_lean(bundle, mesh_path=None, device="cpu", check_mesh=True):
    """Lean bundle -> a live LMGModel, geometry derived in memory.

    No Full checkpoint on disk anywhere. The returned model is ready to render;
    `inflate` is just this plus a `save_ply`.
    """
    from games.mesh_splatting.scene.gaussian_mesh_model import LMGModel, LMGModelHover
    from renderer.mesh_loader import mesh_loader

    meta, t, budgets, tri_idx, alpha_idx, alpha, scale = _rebuild_params(bundle)
    mesh_path = mesh_path or os.path.join(bundle, meta["mesh_file"])
    if check_mesh:
        got = sha256_file(mesh_path)
        if got != meta["mesh_sha256"]:
            raise ValueError("mesh sha256 mismatch: bundle %s, file %s"
                             % (meta["mesh_sha256"], got))

    sh_degree = int(meta["sh_degree"])
    cls = LMGModelHover if meta["model_variant"] == "hover" else LMGModel
    model = cls(sh_degree)

    mesh = mesh_loader.load_transformed_mesh(mesh_path)
    verts = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.int64, device=device)

    model.vertices = torch.nn.Parameter(verts)
    model.faces = faces
    model.triangles = verts[faces]
    model.triangle_indices = torch.from_numpy(tri_idx).to(device)
    model.alpha_indices = torch.from_numpy(alpha_idx).to(device)
    model.num_splats_per_triangle = budgets
    model._alpha = torch.nn.Parameter(torch.from_numpy(alpha).to(device))
    model._scale = torch.nn.Parameter(torch.from_numpy(scale).to(device))
    model._opacity = torch.nn.Parameter(torch.from_numpy(t["opacity"]).to(device))

    n = int(meta["num_splats"])
    n_rest = (sh_degree + 1) ** 2 - 1
    # PLY order is channel-major; undo the transpose _save_ply applies.
    model._features_dc = torch.nn.Parameter(
        torch.from_numpy(t["f_dc"].reshape(n, 3, 1)).transpose(1, 2).contiguous().to(device))
    model._features_rest = torch.nn.Parameter(
        torch.from_numpy(t["f_rest"].reshape(n, 3, n_rest)).transpose(1, 2).contiguous().to(device))

    if meta["model_variant"] == "hover":
        model._hover = torch.nn.Parameter(torch.from_numpy(t["hover"]).to(device))
    if "round_id" in t:
        model.round_id = torch.from_numpy(t["round_id"].astype(np.int64)).to(device)

    # Every other loader ends with this. Without it the model still SAVES
    # correctly (save_ply writes all SH bands regardless) but RENDERS with only
    # the DC band, which is a silent ~0.7 error in image space.
    model.active_sh_degree = model.max_sh_degree

    model.update_alpha()
    model.prepare_scaling_rot()
    return model


def inflate(bundle, out_dir, mesh_path=None, device="cpu", check_mesh=True):
    """Lean bundle -> Full checkpoint dir (point_cloud.ply + model_params.pt)."""
    model = load_lean(bundle, mesh_path=mesh_path, device=device, check_mesh=check_mesh)
    os.makedirs(out_dir, exist_ok=True)
    model.save_ply(os.path.join(out_dir, "point_cloud.ply"))
    return out_dir


# --------------------------------------------------------------------------- #
#                                    verify                                    #
# --------------------------------------------------------------------------- #

def verify(full_dir, mesh_path, workdir, device="cpu", tol=1e-6, sh_degree=3):
    """deflate -> inflate -> compare against the original Full checkpoint."""
    bundle = os.path.join(workdir, "bundle.lmg")
    back = os.path.join(workdir, "roundtrip")
    meta = deflate(full_dir, bundle, mesh_path, link_mesh=True, sh_degree=sh_degree)
    inflate(bundle, back, device=device)

    orig = torch.load(os.path.join(full_dir, "model_params.pt"), map_location="cpu")
    new = torch.load(os.path.join(back, "model_params.pt"), map_location="cpu")

    report = {"meta": meta, "exact": {}, "derived": {}, "sizes": {}}
    for key in ("_alpha", "_scale", "triangle_indices", "alpha_indices",
                "num_splats_per_triangle", "round_id", "_hover"):
        if key not in orig:
            continue
        report["exact"][key] = bool(np.array_equal(_np(orig[key]), _np(new[key])))

    a = _read_appearance(os.path.join(full_dir, "point_cloud.ply"), sh_degree)
    b = _read_appearance(os.path.join(back, "point_cloud.ply"), sh_degree)
    for name, x, y in zip(("f_dc", "f_rest", "opacity"), a, b):
        report["exact"][name] = bool(np.array_equal(x, y))

    # Derived fields: reproduced by formula, compared within tolerance.
    pa, pb = (PlyData.read(os.path.join(d, "point_cloud.ply")).elements[0]
              for d in (full_dir, back))
    for group in (("x", "y", "z"),
                  ("scale_0", "scale_1", "scale_2"),
                  ("rot_0", "rot_1", "rot_2", "rot_3")):
        d = max(float(np.abs(np.asarray(pa[f_]) - np.asarray(pb[f_])).max())
                for f_ in group)
        report["derived"][group[0].split("_")[0]] = d

    full_bytes = sum(os.path.getsize(os.path.join(full_dir, f))
                     for f in ("point_cloud.ply", "model_params.pt"))
    lean_bytes = os.path.getsize(os.path.join(bundle, PARAMS_NAME))
    report["sizes"] = {
        "full_bytes": full_bytes,
        "lean_bytes": lean_bytes,
        "ratio": full_bytes / lean_bytes,
    }
    report["ok"] = (all(report["exact"].values())
                    and all(v <= tol for v in report["derived"].values()))
    return report

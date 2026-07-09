import torch
import trimesh
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes


def load_transformed_mesh(texture_obj_path: str) -> trimesh.Trimesh:
    """Single source of truth for mesh loading. No coordinate transform is
    applied: the old rotation_matrix(-pi/2, x) step some callers used, paired
    with a separate axis-swap+negate some callers applied afterward to the
    derived vertex tensor, compose to the identity (verified numerically) --
    so the two "correct" reference copies in this codebase were actually
    correct only because they cancelled out downstream, while any copy that
    applied the rotation alone (or the swap alone) ended up in the wrong
    frame. Every consumer (Gaussian anchoring, background render, distortion
    policy) must use this raw load (or reuse its returned object) instead of
    re-reading the file or re-applying either half of that transform.
    """
    return trimesh.load(texture_obj_path, force='mesh')


def to_pytorch3d_meshes(mesh_scene: trimesh.Trimesh) -> Meshes:
    verts = torch.tensor(mesh_scene.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh_scene.faces, dtype=torch.int64)
    colors = torch.tensor(mesh_scene.visual.vertex_colors[:, :3], dtype=torch.float32) / 255.0
    return Meshes(
        verts=[verts],
        faces=[faces],
        textures=TexturesVertex(verts_features=[colors])
    ).to("cuda")


def load_textured_mesh(dataset, texture_obj_path: str, mesh_rasterizer_type: str):
    """
    Load a textured 3D mesh for background rendering, in whichever in-memory
    format the given rasterizer backend needs: raw trimesh.Trimesh for
    nvdiffrast, PyTorch3D Meshes for pytorch3d.

    Args:
        dataset: Dataset configuration containing mesh_type attribute.
                Should have mesh_type in ['sugar', 'colmap', 'milo'].
        texture_obj_path: Path to the mesh file. If empty string, raises AssertionError.
        mesh_rasterizer_type: 'pytorch3d' or 'nvdiffrast'.
    Returns:
        trimesh.Trimesh (nvdiffrast) or pytorch3d Meshes (pytorch3d), on CUDA where applicable.
    """
    assert texture_obj_path != "", "[ERROR] texture_obj_path cannot be empty"
    mesh_type = dataset.mesh_type

    if mesh_type == "sugar":  # From SuGaR
        assert mesh_rasterizer_type == "pytorch3d", "[ERROR] sugar mesh_type only supports the pytorch3d rasterizer"
        assert texture_obj_path.lower().endswith(".obj"), "[ERROR] SuGaR mesh should be .obj file!"
        return load_objs_as_meshes([texture_obj_path]).to("cuda")

    elif mesh_type == "colmap" or mesh_type == "milo":
        # From Colmap, download from https://nerfbaselines.github.io/
        assert texture_obj_path.lower().endswith(".ply"), "[ERROR] Colmap mesh should be .ply file!"
        mesh_scene = load_transformed_mesh(texture_obj_path)
        if mesh_rasterizer_type == "nvdiffrast":
            return mesh_scene
        elif mesh_rasterizer_type == "pytorch3d":
            return to_pytorch3d_meshes(mesh_scene)
        else:
            raise ValueError(f"[ERROR] Unknown mesh_rasterizer_type: {mesh_rasterizer_type}")
    else:
        raise ValueError("[ERROR] Unknown/Unsupported mesh type!")

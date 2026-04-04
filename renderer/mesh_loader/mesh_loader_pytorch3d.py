import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
import trimesh

def load_textured_mesh_for_pytorch3d(dataset, texture_obj_path: str) -> Meshes:
    """
    Load a textured 3D mesh from the given path for background rendering.
    
    This function loads mesh of SuGaR (.obj) or Colmap (.ply) format (or others, add if needed)
    and converts it to a PyTorch3D Meshes object on CUDA
    
    Args:
        dataset: Dataset configuration containing mesh_type attribute.
                Should have mesh_type in ['sugar', 'colmap', ...].
        texture_obj_path: Path to the mesh file. If empty string, raises AssertionError.
    Returns:
        Meshes: A PyTorch3D Meshes object on CUDA
    Raises:
        AssertionError: If texture_obj_path is empty or mesh type is unsupported.
        AssertionError: If file extension doesn't match expected format.
    """
    
    assert texture_obj_path != "", "[ERROR] texture_obj_path cannot be empty"
    textured_mesh = None
    mesh_type = dataset.mesh_type
    if texture_obj_path != "":
        print("[INFO] Loading textured mesh for background rendering...")
        
        if mesh_type == "sugar": # From SuGaR
            assert texture_obj_path.lower().endswith(".obj"), "[ERROR] SuGaR mesh should be .obj file!"
            textured_mesh = load_objs_as_meshes([texture_obj_path]).to("cuda")
             
        elif mesh_type == "colmap" or mesh_type == "milo": 
            # From Colmap, download from https://nerfbaselines.github.io/
            assert texture_obj_path.lower().endswith(".ply"), "[ERROR] Colmap mesh should be .ply file!"
            mesh_tm = trimesh.load(texture_obj_path, force='mesh', process=False)
            verts = torch.tensor(mesh_tm.vertices, dtype=torch.float32)
            faces = torch.tensor(mesh_tm.faces, dtype=torch.int64)
            colors = torch.tensor(mesh_tm.visual.vertex_colors[:, :3], dtype=torch.float32) / 255.0
            
            # Combine into a textured mesh
            textured_mesh = Meshes(
                verts=[verts],
                faces=[faces],
                textures=TexturesVertex(verts_features=[colors])
            ).to("cuda")
        else:
            print("[ERROR] Unknown/Unsupported mesh type!")        
            
    assert textured_mesh is not None, "[ERROR] Textured mesh is not loaded properly!"
    
    return textured_mesh
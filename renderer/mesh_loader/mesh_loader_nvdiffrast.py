import trimesh

def load_textured_mesh_for_nvdiffrast(dataset, texture_obj_path: str):
    return trimesh.load(texture_obj_path, force='mesh')

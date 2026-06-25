#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#

import os
import numpy as np
import trimesh
import torch
import time

# >>>> [YC] add
from games.mesh_splatting.scene.dataset_readers \
    import get_num_splats_per_triangle, transform_vertices_function
from games.mesh_splatting.utils.graphics_utils import MeshPointCloud
# <<< [YC] add

from games.multi_mesh_splatting.utils.graphics_utils import MultiMeshPointCloud
from scene.dataset_readers import (
    readColmapSceneInfo,
    readNerfSyntheticInfo,
    getNerfppNorm,
    SceneInfo,
    storePly
)
from utils.sh_utils import SH2RGB

from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    read_extrinsics_binary,
    read_intrinsics_binary
)

from scene.dataset_readers import readColmapCameras


def readColmapMeshSceneInfo(path, images, eval, num_splats, meshes, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    pcds = []
    ply_paths = []
    total_pts = 0
    for i, (mesh, num) in enumerate(zip(meshes, num_splats)):
        ply_path = os.path.join(path, f"points3d_{i}.ply")

        mesh_scene = trimesh.load(f'{path}/sparse/0/{mesh}.obj', force='mesh')
        vertices = mesh_scene.vertices
        faces = mesh_scene.faces
        triangles = torch.tensor(mesh_scene.triangles).float()  # equal vertices[faces]

        num_pts_each_triangle = num
        num_pts = num_pts_each_triangle * triangles.shape[0]
        total_pts += num_pts

        # We create random points inside the bounds traingles
        alpha = torch.rand(
            triangles.shape[0],
            num_pts_each_triangle,
            3
        )

        xyz = torch.matmul(
            alpha,
            triangles
        )
        xyz = xyz.reshape(num_pts, 3)

        shs = np.random.random((num_pts, 3)) / 255.0

        pcd = MultiMeshPointCloud(
            alpha=alpha,
            points=xyz,
            colors=SH2RGB(shs),
            normals=np.zeros((num_pts, 3)),
            vertices=vertices,
            faces=faces,
            triangles=triangles.cuda()
        )
        pcds.append(pcd)
        ply_paths.append(ply_path)
        storePly(ply_path, pcd.points, SH2RGB(shs) * 255)
    
    print(
        f"Generating random point cloud ({total_pts})..."
    )

    scene_info = SceneInfo(point_cloud=pcds,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_paths)
    
    return scene_info

# >>>> [YC] add: Only single mesh and follow the Blender (readNerfSyntheticInfo) style
def readColmapSingleMeshSceneInfo(
        path, images, eval, num_splats, 
        # meshes,
        # >>>> [YC] add 
        texture_obj_path: str = None,
        policy_path: str = None,
        total_splats: int = None,  
        budget_per_tri: float = None, 
        budgeting_policy_name: str = "uniform",
        min_splats_per_tri: int = 0,
        max_splats_per_tri: int = 8,
        mesh_type: str = "sugar",
        textured_mesh = None,
        # <<< [YC] add
        llffhold=8):
    
    print("[DEBUG] readColmapSingleMeshSceneInfo called with:")
    
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    
    sort_start = time.time()
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name) 
    sort_end = time.time()
    print(f"[PROFILE] Camera sorting took {sort_end - sort_start:.4f} seconds for {len(cam_infos_unsorted)} cameras")

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        print(f"[INFO] DatasetReader::Colmap scene dataset caminfos: using {len(train_cam_infos)} training views and {len(test_cam_infos)} test views.")
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    pcds = []
    ply_paths = []
    total_pts = 0
    
    if texture_obj_path is None:
        print(f"[INFO] DatasetReader::Reading Mesh object from {path}/mesh.obj")
        mesh_scene = trimesh.load(f'{path}/mesh.obj', force='mesh')
    else:
        print(f"[INFO] Reading Mesh object from {texture_obj_path}")
        mesh_scene = trimesh.load(texture_obj_path, force='mesh')
    
    mesh_scene.apply_transform(trimesh.transformations.rotation_matrix(
        angle=-np.pi/2, direction=[1, 0, 0], point=[0, 0, 0]
    ))
    
    vertices = mesh_scene.vertices
    vertices = transform_vertices_function(
        torch.tensor(vertices),
    )
    faces = mesh_scene.faces
    triangles = vertices[torch.tensor(mesh_scene.faces).long()].float()
    
    has_uv = (mesh_type == "sugar")
    
    print(f"[DEBUG] mesh_type: {mesh_type}, has_uv: {has_uv}")
    
    if has_uv:
        print("[INFO] Mesh has UV coordinates and texture PNG.")
        
        # Access texture info
        print("type(mesh_scene.visual):", type(mesh_scene.visual)) # [YC] debug: should be TextureVisuals
        print("mesh_scene.visual.uv.shape:", mesh_scene.visual.uv.shape) # [YC] debug: UV coordinates per vertex
        print("mesh_scene.visual.material.image:", mesh_scene.visual.material.image) # [YC] debug: PIL image
    
        # Extract texture as numpy
        texture_img = np.array(mesh_scene.visual.material.image)  # (H, W, 3) or (H, W, 4)
        H, W = texture_img.shape[:2]
        
        # Get per-vertex UVs
        uv_coords = mesh_scene.visual.uv  # (n_vertices, 2), values in [0,1]
        
        # Face UVs (map vertex index -> uv)
        face_uvs = uv_coords[faces]  # (n_faces, 3, 2)
    
    ply_path = os.path.join(path, "points3d.ply") # What is this points3d.ply? COLMAP data, generated point cloud, or just placeholder?
    print("ply_path:", ply_path)
    
    if True:
        assert budget_per_tri is not None or total_splats is not None, "Either num_splats or total_splats must be provided for budgeting!"
        
        if total_splats is None:
            total_splats = int(budget_per_tri * triangles.shape[0])
            print(f"[INFO] total_splats not provided, computed from budget_per_tri: {total_splats} splats")
        else:
            print(f"[INFO] total_splats provided: {total_splats} splats")
        
        # >>>> [SAM] Budgeting policy integration
        num_splats_per_triangle = get_num_splats_per_triangle(
            triangles=triangles,
            mesh_scene=mesh_scene,
            train_cam_infos=train_cam_infos,
            path=path,
            num_splats=num_splats,
            policy_path=policy_path,
            total_splats=total_splats,
            budgeting_policy_name=budgeting_policy_name,
            textured_mesh=textured_mesh,
            mesh_type=mesh_type
            
        )
        # <<<< [SAM] Budgeting policy integration
        
        # Since this data set has no colmap data, we start with random points sampled on the mesh surface
        num_pts = num_splats_per_triangle.sum()
        print(f"Generating random point cloud ({num_pts})...")
        print(f"Average points per triangle: {num_pts / triangles.shape[0] if triangles.shape[0] > 0 else 0}...")
        
        # ---------------------------------------------------------------------------- #
        #                 Get initial Gaussian colors from texture map                 #
        # ---------------------------------------------------------------------------- #
        if has_uv:
            face_uvs = mesh_scene.visual.uv[faces]  # (n_faces, 3, 2)
            H, W = texture_img.shape[:2]        
            
            # Convert UVs (3 per face) to pixel coordinates
            px = (face_uvs[..., 0] * (W - 1)).astype(int)
            py = ((1 - face_uvs[..., 1]) * (H - 1)).astype(int)
            
            # Clamp
            px = np.clip(px, 0, W - 1)
            py = np.clip(py, 0, H - 1)
        
            # Sample 3 vertex colors per triangle
            tri_vertex_colors = texture_img[py, px, :3]   # (n_faces, 3, 3)
            
            # Average per-triangle color
            tri_avg_colors = tri_vertex_colors.mean(axis=1)  # (n_faces, 3)
            
            print("tri_avg_colors:", tri_avg_colors.shape) # [YC] debug
            # colors = np.repeat(tri_avg_colors, num_splats, axis=0)  # (num_pts, 3)
        else:
            print("[INFO] Mesh has no UV/texture — using vertex colors instead.")
            if isinstance(mesh_scene.visual, trimesh.visual.color.ColorVisuals):
                vertex_colors = mesh_scene.visual.vertex_colors[:, :3]
                print(f"[INFO] Loaded vertex colors: {vertex_colors.shape}")
                faces = mesh_scene.faces
                tri_vertex_colors = vertex_colors[faces]  # (n_faces, 3, 3)
                tri_avg_colors = tri_vertex_colors.mean(axis=1)
            else:
                raise ValueError("Mesh has neither UV/texture nor vertex colors!")
            
        # We create random points inside the bounds triangles
        xyz_list = []
        alpha_list = []
        alpha_indices_list = []
        color_list = []
        tri_indices_list = []

        # >>>> [SAM] Build point-to-triangle mapping
        for i in range(triangles.shape[0]):
            n = num_splats_per_triangle[i]
            if n == 0:
                continue
                
            alpha = torch.rand(n, 3)
            alpha = alpha / alpha.sum(dim=1, keepdim=True)  # normalize to barycentric coords

            pts = (alpha[:, 0:1] * triangles[i, 0] +
                alpha[:, 1:2] * triangles[i, 1] +
                alpha[:, 2:3] * triangles[i, 2])

            color = np.repeat(tri_avg_colors[i].reshape(1, 3), n, axis=0)  # (num_pts, 3)
            # print(color.shape) # [YC] debug
            
            xyz_list.append(pts)
            alpha_list.append(alpha)
            alpha_indices_list.append(torch.arange(n))
            color_list.append(color)
            tri_indices_list.append(torch.full((n,), i, dtype=torch.long))
        # <<<< [SAM]
        
        # [DEBUG] Check if xyz_list is populated
        print(f"[DEBUG] xyz_list length: {len(xyz_list)}")
        if len(xyz_list) == 0:
            print("[ERROR] xyz_list is empty! No points were generated from triangles.")
            print(f"[DEBUG] triangles shape: {triangles.shape}")
            print(f"[DEBUG] num_pts_each_triangle: {num_splats_per_triangle}")
            raise RuntimeError("Failed to generate random points inside triangles")
        
        
        xyz = torch.cat(xyz_list, dim=0)
        xyz = xyz.reshape(num_pts, 3)
        
        alpha = torch.cat(alpha_list, dim=0)
        alpha_indices = torch.cat(alpha_indices_list, dim=0)

        # shs = np.random.random((num_pts, 3)) / 255.0
        colors = np.concatenate(color_list, axis=0)
        print(colors.shape, xyz.shape, alpha.shape) # [YC] debug
        
        points = trimesh.points.PointCloud(np.array(xyz))

        # Combine into a scene
        scene = trimesh.Scene()
        scene.add_geometry(mesh_scene)
        scene.add_geometry(points)
        
        tri_indices = torch.cat(tri_indices_list, dim=0)
        
        pcd = MeshPointCloud(
            alpha=alpha,
            alpha_indices=alpha_indices,
            points=xyz,
            # colors=SH2RGB(shs),
            colors=colors/255.0,
            normals=np.zeros((num_pts, 3)),
            vertices=vertices,
            faces=faces,
            transform_vertices_function=transform_vertices_function,
            triangles=triangles.cuda(),
            triangle_indices=tri_indices
        )
        print("Created MeshPointCloud with", pcd.points.shape[0], "points.")

        # storePly(ply_path, pcd.points, SH2RGB(shs) * 255)
        storePly(ply_path, pcd.points, colors)
        print("Stored initial point cloud to", ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# <<<< [YC] add
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Colmap_Mesh": readColmapMeshSceneInfo,
    "Colmap_Single_Mesh": readColmapSingleMeshSceneInfo
}

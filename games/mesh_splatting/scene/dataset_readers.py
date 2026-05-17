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

from games.mesh_splatting.utils.graphics_utils import MeshPointCloud
from scene.dataset_readers import (
    readColmapSceneInfo,
    readNerfSyntheticInfo,
    readCamerasFromTransforms,
    getNerfppNorm,
    SceneInfo,
    storePly,
)
from utils.sh_utils import SH2RGB
from scene.budgeting import get_budgeting_policy

from pathlib import Path

softmax = torch.nn.Softmax(dim=2)


def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices


def get_num_splats_per_triangle(
    triangles, # [N,3,3]
    mesh_scene,
    train_cam_infos,
    path,
    num_splats,
    policy_path: str = None,
    total_splats: int = None,
    budgeting_policy_name: str = "uniform",
    # min_splats_per_tri: int = 0, # [TODO] could be adjusted
    # max_splats_per_tri: int = 8, # [TODO] could be adjusted
    textured_mesh = None,
    mesh_type: str = "sugar"
)-> np.ndarray: # [N,], number of splats on each triangle
    
    # define allocation_path only when policy_path provided
    allocation_path = Path(policy_path) if policy_path else None

    # load num_splats allocation policy from pre-computed file
    if allocation_path is not None and allocation_path.exists():
        print(f"[INFO] Loading splat allocation from: {allocation_path}")
        num_splats_per_triangle = np.load(allocation_path)
        print("[INFO] loaded distribution, max and min:", num_splats_per_triangle.max(), num_splats_per_triangle.min())
    
    # [DONE] load weights here
            
    # Use budgeting policy, computing on-the-fly
    elif total_splats is not None:
        print(f"[INFO] no pre-computed policy found")
        print(f"[INFO] Recalculating splat allocation on the fly")
        print(f"[INFO] Scene::Reader() Using budgeting policy: {budgeting_policy_name}")

        budgeting_policy = get_budgeting_policy(
            budgeting_policy_name,
            mesh=mesh_scene,
            viewpoint_camera_infos=train_cam_infos, # access camera objects from cam_infos in the allocator somehow
            dataset_path=path,
            mesh_type=mesh_type,
            p3d_mesh=textured_mesh,
        )
        num_splats_per_triangle = budgeting_policy.allocate(
            total_splats=total_splats,
        )
        
        print(f"[INFO] Scene::Reader() Requested total splats: {total_splats}")
        print(f"[INFO] Scene::Reader() Allocated total splats: {num_splats_per_triangle.sum()}")
        print(f"[INFO] Scene::Reader() Min/Max splats per triangle: {num_splats_per_triangle.min()}/{num_splats_per_triangle.max()}")
        print(f"[INFO] Scene::Reader() Mean/Std splats per triangle: {num_splats_per_triangle.mean():.2f}/{num_splats_per_triangle.std():.2f}")

        # save under {dataset_path}/policy/mesh_{mesh_type}/tri_{num_tri}/{policy_name}/
        # budget: {total_splats}.npy
        # weights: weights.npy (same for any budget using the same policy & same mesh)
        num_tri = num_splats_per_triangle.shape[0]
        allocation_save_path = Path(path) / f"policy/mesh_{mesh_type}/tri_{num_tri}/{budgeting_policy_name}/{total_splats}.npy"
        allocation_save_path.parent.mkdir(parents=True, exist_ok=True)
        weights_save_path = allocation_save_path.parent / f"weights.npy"
        
        np.save(allocation_save_path, num_splats_per_triangle)
        print(f"[INFO] Scene::Reader() Saving allocation policy file to: {allocation_save_path}")

        np.save(weights_save_path, budgeting_policy.weights)
        print(f"[INFO] Scene::Reader() Saving weights file to: {weights_save_path}")
        
        # [DOING] [DONE] save the weights[] 

    # Fallback: uniform distribution using num_splats
    else:
        num_splats_per_triangle = np.full(triangles.shape[0], num_splats, dtype=int)
        print(f"[WARNING] Scene::Reader() Fallback using uniform distribution: {num_splats} splats per triangle")

    num_pts = num_splats_per_triangle.sum()
    print(f"[INFO] Generating random point cloud ({num_pts})...")
    print(f"\tnumber of mesh faces:  {triangles.shape[0]}...")
    print(f"\tAverage points per triangle: {num_pts / triangles.shape[0] if triangles.shape[0] > 0 else 0}...")
    
    return num_splats_per_triangle

def my_get_num_splats_per_triangle(
    dataset_path: str,
    mesh_scene: trimesh.Trimesh,
    triangles, faces, vertices,
    viewpoint_camera_infos,
    total_splats: int = None,
    budgeting_policy_name: str = "uniform",
    policy_path: str = None,
    mesh_type: str = "sugar"
)-> np.ndarray: # [N,], number of splats on each triangle
    
    # define allocation_path only when policy_path provided
    allocation_path = Path(policy_path) if policy_path else None
    
    # load num_splats allocation policy from pre-computed file
    if allocation_path is not None and allocation_path.exists():
        print(f"[INFO] Loading splat allocation from: {allocation_path}")
        num_splats_per_triangle = np.load(allocation_path)
        print("[INFO] loaded distribution, max and min:", num_splats_per_triangle.max(), num_splats_per_triangle.min())
        # [DONE] load weights here
        
    # Use budgeting policy, computing on-the-fly
    elif total_splats is not None:
        print(f"[INFO] no pre-computed policy found")
        print(f"[INFO] Recalculating splat allocation on the fly")
        print(f"[INFO] Scene::Reader() Using budgeting policy: {budgeting_policy_name}")

        budgeting_policy = get_budgeting_policy(
            budgeting_policy_name, # choose "MyDistortionMapBudgetingPolicy"
            mesh=mesh_scene,
            mesh_type=mesh_type,
            viewpoint_camera_infos=viewpoint_camera_infos, # access camera objects from cam_infos in the allocator somehow
            dataset_path=dataset_path
        )
        num_splats_per_triangle = budgeting_policy.allocate(
            total_splats=total_splats,
        )
        
        print(f"[INFO] Scene::Reader() Requested total splats: {total_splats}")
        print(f"[INFO] Scene::Reader() Allocated total splats: {num_splats_per_triangle.sum()}")
        print(f"[INFO] Scene::Reader() Min/Max splats per triangle: {num_splats_per_triangle.min()}/{num_splats_per_triangle.max()}")
        print(f"[INFO] Scene::Reader() Mean/Std splats per triangle: {num_splats_per_triangle.mean():.2f}/{num_splats_per_triangle.std():.2f}")

        # save under {dataset_path}/policy/mesh_{mesh_type}/tri_{num_tri}/{policy_name}/
        # budget: {total_splats}.npy
        # weights: weights.npy (same for any budget using the same policy & same mesh)
        num_tri = num_splats_per_triangle.shape[0]
        allocation_save_path = Path(dataset_path) / f"policy" / f"mesh_{mesh_type}" / f"tri_{num_tri}" / f"{budgeting_policy_name}" / f"{total_splats}.npy"
        allocation_save_path.parent.mkdir(parents=True, exist_ok=True)
        weights_save_path = allocation_save_path.parent / f"weights.npy"
        
        np.save(allocation_save_path, num_splats_per_triangle)
        np.save(allocation_path, num_splats_per_triangle)
        print(f"[INFO] Scene::Reader() Saving allocation policy file to: {allocation_save_path} and {allocation_path}")

        np.save(weights_save_path, budgeting_policy.weights)
        np.save(allocation_path.parent / f"weights.npy", budgeting_policy.weights)
        print(f"[INFO] Scene::Reader() Saving weights file to: {weights_save_path} and {allocation_path.parent / f'weights.npy'}")
        
        # [DOING] [DONE] save the weights[] 

    # Fallback: uniform distribution using num_splats
    else:
        num_splats_per_triangle = np.full(triangles.shape[0], num_splats, dtype=int)
        print(f"[WARNING] Scene::Reader() Fallback using uniform distribution: {num_splats} splats per triangle")

    num_pts = num_splats_per_triangle.sum()
    print(f"[INFO] Generating random point cloud ({num_pts})...")
    print(f"\tnumber of mesh faces:  {triangles.shape[0]}...")
    print(f"\tAverage points per triangle: {num_pts / triangles.shape[0] if triangles.shape[0] > 0 else 0}...")
    
    return num_splats_per_triangle


def get_mesh_for_policy():
    # [TODO] load the mesh used for budgeting policy, which should be the same as the one used for training
    pass

# [YC] note: "Blender_Mesh"
def readNerfSyntheticMeshInfo( # don't use num_splats
        path, white_background, eval, num_splats, 
        extension=".png",
        texture_obj_path: str = None,
        policy_path: str = None,
        total_splats: int = None,  
        budget_per_tri: float = None, 
        budgeting_policy_name: str = "uniform",
        min_splats_per_tri: int = 0,
        max_splats_per_tri: int = 8,
        mesh_type: str = "sugar",
        textured_mesh = None,
        preload_gs_path: str = None
) -> SceneInfo:
    # ------------------------------- Read cameras ------------------------------- #
    print("[INFO] DatasetReader::Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("[INFO] DatasetReader::Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    # [YC] [TODO] this is weird, why merge train and test cams, even if not in --eval mode?
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ----------------------------- Load texture mesh ---------------------------- #
    # not priority for now: clean all the mesh loading logic into one place.
    # [DONE] a workaround to send loaded-texture-mesh to budgeting.py
    if texture_obj_path is None:
        print(f"[INFO] DatasetReader::Reading Mesh object from {path}/mesh.obj")
        mesh_scene = trimesh.load(f'{path}/mesh.obj', force='mesh')
    else:
        print(f"[INFO] DatasetReader::Reading Mesh object from {texture_obj_path}")
        mesh_scene = trimesh.load(texture_obj_path, force='mesh')

    # -------------------------- Transform texture mesh -------------------------- #
    # >>>> [YC] add
    # [YC] [NOTE]: because the MILO's mesh is generated from torch3d, so need to rotate
    mesh_scene.apply_transform(trimesh.transformations.rotation_matrix(
        angle=-np.pi/2, direction=[1, 0, 0], point=[0, 0, 0]
    ))
    # <<<< [YC] add
    
    vertices = mesh_scene.vertices
    vertices = transform_vertices_function(
        torch.tensor(vertices),
    )
    faces = mesh_scene.faces
    triangles = vertices[torch.tensor(mesh_scene.faces).long()].float()
    
    # >>>> [YC] add
    # [YC] [NOTE]: set to false support mesh from colmap; otherwise true for sugar-generated mesh
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
    # <<<< [YC] add
    

    ply_path = os.path.join(path, "points3d.ply") # [YC]: What is this points3d.ply? COLMAP data, generated point cloud, or just placeholder?
    print("ply_path:", ply_path)
    
    # if not os.path.exists(ply_path):
    if True:
        
        assert budget_per_tri is not None or total_splats is not None, "Either num_splats or total_splats must be provided for budgeting!"
        
        if total_splats is None:
            # total_splats = int(budget_per_tri * triangles.shape[0])
            # print(f"[INFO] total_splats not provided, computed from budget_per_tri: {total_splats} splats")
            pass
        else:
            print(f"[INFO] total_splats is provided: {total_splats} splats")
        
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
            # min_splats_per_tri=min_splats_per_tri,
            # max_splats_per_tri=max_splats_per_tri,
            textured_mesh=textured_mesh,
            mesh_type=mesh_type,
        )
        print(f"[INFO] num_splats_per_triangle: {num_splats_per_triangle}")
        # <<<< [SAM] Budgeting policy integration
        num_pts = num_splats_per_triangle.sum()
        
        # Since this data set has no colmap data, we start with random points sampled on the mesh surface

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
        color_list = []
        tri_indices_list = []
        
        static_alpha = torch.rand(100, 3) # [YC] add
        torch.save(static_alpha, 'static_alpha.pt') # [YC] add
        
        # [TODO] Build point-to-triangle mapping & triangle-to-point mapping
        for i in range(triangles.shape[0]):
            n = num_splats_per_triangle[i]
            if n == 0:
                continue
                
            # alpha = torch.rand(n, 3)
            alpha = static_alpha[:n]
            alpha = alpha / alpha.sum(dim=1, keepdim=True)  # normalize to barycentric coords

            pts = (alpha[:, 0:1] * triangles[i, 0] +
                alpha[:, 1:2] * triangles[i, 1] +
                alpha[:, 2:3] * triangles[i, 2])

            color = np.repeat(tri_avg_colors[i].reshape(1, 3), n, axis=0)  # (num_pts, 3)
            # print(color.shape) # [YC] debug
            
            xyz_list.append(pts)
            alpha_list.append(alpha)
            color_list.append(color)
            tri_indices_list.append(torch.full((n,), i, dtype=torch.long))

 
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
        # [NOTE] should save both gs-to-tri mapping and tri-to-gs triangle_indices
        
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

# [YC] note: This function only load the cameras
def readNerfSyntheticMeshInfoSimple( # don't use num_splats
        path, white_background, eval, extension=".png"
) -> SceneInfo:
    # ------------------------------- Read cameras ------------------------------- #
    print("[INFO] DatasetReader::Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("[INFO] DatasetReader::Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    # [YC] [TODO] this is weird, why merge train and test cams, even if not in --eval mode?
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None)
    return scene_info


def create_init_point_cloud(
    model_path,
    triangles, faces, vertices,
    num_splats_per_triangle,
    tri_avg_colors
):
    # We create random points inside the bounds triangles
    xyz_list = []
    alpha_list = []
    color_list = []
    tri_indices_list = []
        
    static_alpha = torch.rand(100, 3) # [YC] add
    torch.save(static_alpha, 'static_alpha.pt') # [YC] add
        
    # [TODO] Build point-to-triangle mapping & triangle-to-point mapping
    for i in range(triangles.shape[0]):
        n = num_splats_per_triangle[i]
        if n == 0:
            continue
            
        # alpha = torch.rand(n, 3)
        alpha = static_alpha[:n]
        alpha = alpha / alpha.sum(dim=1, keepdim=True)  # normalize to barycentric coords

        pts = (alpha[:, 0:1] * triangles[i, 0] +
            alpha[:, 1:2] * triangles[i, 1] +
            alpha[:, 2:3] * triangles[i, 2])

        color = np.repeat(tri_avg_colors[i].reshape(1, 3), n, axis=0)  # (num_pts, 3)
        # print(color.shape) # [YC] debug
        
        xyz_list.append(pts)
        alpha_list.append(alpha)
        color_list.append(color)
        tri_indices_list.append(torch.full((n,), i, dtype=torch.long))

 
    # [DEBUG] Check if xyz_list is populated
    print(f"[DEBUG] xyz_list length: {len(xyz_list)}")
    if len(xyz_list) == 0:
        print("[ERROR] xyz_list is empty! No points were generated from triangles.")
        print(f"[DEBUG] triangles shape: {triangles.shape}")
        print(f"[DEBUG] num_pts_each_triangle: {num_splats_per_triangle}")
        raise RuntimeError("Failed to generate random points inside triangles")
        
    num_pts = num_splats_per_triangle.sum()
       
    xyz = torch.cat(xyz_list, dim=0)
    xyz = xyz.reshape(num_pts, 3)
    
    alpha = torch.cat(alpha_list, dim=0)
    
    colors = np.concatenate(color_list, axis=0)
    
    tri_indices = torch.cat(tri_indices_list, dim=0)        
        
    pcd = MeshPointCloud(
        alpha=alpha,
        points=xyz,
        colors=colors/255.0,
        normals=np.zeros((num_pts, 3)),
        vertices=vertices,
        faces=faces,
        transform_vertices_function=transform_vertices_function,
        triangles=triangles.cuda(),
        triangle_indices=tri_indices
    )
    
    print("Created MeshPointCloud with", pcd.points.shape[0], "points.")

    # # storePly(ply_path, pcd.points, SH2RGB(shs) * 255)
    # ply_path = os.path.join(model_path, "points3d.ply") 
    # storePly(ply_path, pcd.points, colors)
    # print("Stored initial point cloud to", ply_path)
    
    return pcd
    
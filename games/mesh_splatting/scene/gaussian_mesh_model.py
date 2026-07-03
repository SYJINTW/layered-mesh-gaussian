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

import torch
import numpy as np
from plyfile import PlyData, PlyElement

from torch import nn

from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid, rot_to_quat_batch
from utils.sh_utils import RGB2SH
from games.mesh_splatting.utils.graphics_utils import MeshPointCloud


class GaussianMeshModel(GaussianModel):

    def __init__(self, sh_degree: int):

        super().__init__(sh_degree)
        self.point_cloud = None
        self._alpha = torch.empty(0)
        self._scale = torch.empty(0)
        self.alpha = torch.empty(0)
        self.softmax = torch.nn.Softmax(dim=2)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.update_alpha_func = self.softmax

        self.vertices = None
        self.faces = None
        self.triangles = None
        self.eps_s0 = 1e-8
        
        # >>>> [YC] add
        self.triangle_indices = None
        # <<<< [YC] add

    @property
    def get_xyz(self):
        return self._xyz

    def create_from_pcd(self, pcd: MeshPointCloud, spatial_lr_scale: float):
        self.point_cloud = pcd
        self.triangles = self.point_cloud.triangles
        self.spatial_lr_scale = spatial_lr_scale
        
        self.triangle_indices = pcd.triangle_indices.cuda() # [YC] add
        
        pcd_alpha_shape = pcd.alpha.shape

        print("Number of faces: ", pcd_alpha_shape[0])
        print("Number of points at initialisation in face: ", pcd_alpha_shape[1])

        alpha_point_cloud = pcd.alpha.float().cuda()
        scale = torch.ones((pcd.points.shape[0], 1)).float().cuda()

        print("Number of points at initialisation : ", alpha_point_cloud.shape[0] * alpha_point_cloud.shape[1])

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        opacities = inverse_sigmoid(0.1 * torch.ones((pcd.points.shape[0], 1), dtype=torch.float, device="cuda"))
        # opacities = inverse_sigmoid(0.01 * torch.ones((pcd.points.shape[0], 1), dtype=torch.float, device="cuda"))

        self.vertices = nn.Parameter(
            self.point_cloud.vertices.clone().detach().requires_grad_(True).cuda().float()
        )
        self.faces = torch.tensor(self.point_cloud.faces).cuda()

        self._alpha = nn.Parameter(alpha_point_cloud.requires_grad_(True))
        self.update_alpha()
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scale = nn.Parameter(scale.requires_grad_(True))
        self.prepare_scaling_rot()
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def _calc_xyz(self):
        """
        calculate the 3d Gaussian center in the coordinates xyz.

        The alphas that are taken into account are the distances
        to the vertices and the coordinates of
        the triangles forming the mesh.

        """
        # _xyz = torch.matmul(
        #     self.alpha,
        #     self.triangles
        # )
        # self._xyz = _xyz.reshape(
        #         _xyz.shape[0] * _xyz.shape[1], 3
        #     )
        
        triangle_idx = self.triangles[self.triangle_indices].detach()  # constants
        _xyz = torch.bmm(self.alpha.unsqueeze(1), triangle_idx).squeeze(1)
        self._xyz = _xyz.detach()
        
    def prepare_scaling_rot(self):
        """
        approximate covariance matrix and calculate scaling/rotation tensors

        covariance matrix is [v0, v1, v2], where
        v0 is a normal vector to each face
        v1 is a vector from centroid of each face and 1st vertex
        v2 is obtained by orthogonal projection of a vector from
        centroid to 2nd vertex onto subspace spanned by v0 and v1.
        """
        def dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)
        
        def proj(v, u):
            """
            projection of vector v onto subspace spanned by u

            vector u is assumed to be already normalized
            """
            coef = dot(v, u)
            return coef * u

        triangles = self.triangles
        normals = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1
        )
        v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + self.eps_s0)
        means = torch.mean(triangles, dim=1)
        v1 = triangles[:, 1] - means
        v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + self.eps_s0
        v1 = v1 / v1_norm
        v2_init = triangles[:, 2] - means
        v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1)  # Gram-Schmidt
        v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + self.eps_s0)

        s1 = v1_norm / 2.
        s2 = dot(v2_init, v2) / 2.
        s0 = self.eps_s0 * torch.ones_like(s1)
        scales = torch.concat((s0, s1, s2), dim=1).unsqueeze(dim=1)
        
        # scales = scales.broadcast_to((*self.alpha.shape[:2], 3))
        # self._scaling = torch.log(
        #     torch.nn.functional.relu(self._scale * scales.flatten(start_dim=0, end_dim=1)) + self.eps_s0
        # )
        # rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
        # rotation = rotation.broadcast_to((*self.alpha.shape[:2], 3, 3)).flatten(start_dim=0, end_dim=1)
        # rotation = rotation.transpose(-2, -1)
        # self._rotation = rot_to_quat_batch(rotation)
        
        scales = scales[self.triangle_indices]
        with torch.no_grad():
            s_input = self._scale * scales.flatten(start_dim=0, end_dim=1)
            s_input = torch.nn.functional.relu(s_input) + self.eps_s0
            new_scaling = torch.log(s_input)
        self._scaling = new_scaling.detach()
        
        rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
        # rotation = rotation.broadcast_to((*self.alpha.shape[:2], 3, 3)).flatten(start_dim=0, end_dim=1)
        rotation = rotation[self.triangle_indices]
        rotation = rotation.transpose(-2, -1)
        # print("[--> DEBUG] GaussianMeshModel::_rotation:", self._rotation)
        self._rotation = rot_to_quat_batch(rotation)
        
        
    def update_alpha(self):
        """
        Function to control the alpha value.

        Alpha is the distance of the center of the gauss
         from the vertex of the triangle of the mesh.
        Thus, for each center of the gauss, 3 alphas
        are determined: alpha1 + alpha2 + alpha3.
        For a point to be in the center of the vertex,
        the alphas must meet the assumptions:
        alpha1 + alpha2 + alpha3 = 1
        and alpha1 + alpha2 + alpha3 >= 0
        
        Implementation:
        Make sure all alphas are positive by applying ReLU
        Then normalize them so that they sum to 1 for each triangle.
        """
        self.alpha = torch.relu(self._alpha) + 1e-8
        self.alpha = self.alpha / self.alpha.sum(dim=-1, keepdim=True)
        if self.vertices is not None and self.faces is not None:
            self.triangles = self.vertices[self.faces]
            self._calc_xyz()

    def update_alpha_triangle(self):
        """
        Update alpha values triangle by triangle instead of all at once.
        """
    
    def training_setup(self, training_args):
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l_params = [
            {'params': [self.vertices], 'lr': training_args.vertices_lr, "name": "vertices"},
            {'params': [self._alpha], 'lr': training_args.alpha_lr, "name": "alpha"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scale], 'lr': training_args.scaling_lr, "name": "scaling"}
        ]

        self.optimizer = torch.optim.Adam(l_params, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration) -> None:
        """ Learning rate scheduling per step """
        pass

    def save_ply(self, path):
        self.update_alpha()
        self.prepare_scaling_rot()
        self._save_ply(path)

        attrs = self.__dict__
        additional_attrs = [
            '_alpha', 
            '_scale',
            'point_cloud',
            'triangles',
            'vertices',
            'faces'
        ]

        save_dict = {}
        for attr_name in additional_attrs:
            save_dict[attr_name] = attrs[attr_name]

        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        torch.save(save_dict, path_model)

    def load_ply(self, path, texture_obj_path=None):
        # print(f"[DEBUG] Loading GaussianMeshModel from ply file: {path}...")
        # Load from ply file
        self._load_ply(path)

        # Load from pt file
        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        params = torch.load(path_model)
        alpha = params['_alpha']
        scale = params['_scale']

        # ---------------------------------------------------------------------------- #
        #       "vertices", "triangles", "faces" cab be loaded from texture mesh       #
        # ---------------------------------------------------------------------------- #
        if texture_obj_path is None:
            raise ValueError("GaussianMeshModel.load_ply requires texture_obj_path to load the mesh")
        import trimesh
        mesh_scene = trimesh.load(texture_obj_path, force='mesh')
        mesh_scene.apply_transform(trimesh.transformations.rotation_matrix(
            angle=-np.pi/2, direction=[1, 0, 0], point=[0, 0, 0]
        ))
        # --------------------------------- VERTICES --------------------------------- #
        # print("[DEBUG] Loaded vertices from mesh")
        vertices = mesh_scene.vertices
        # transform_vertices_function()
        vertices = torch.tensor(vertices[:, [0, 2, 1]])
        vertices[:, 1] = -vertices[:, 1]
        vertices *= 1
        vertices = nn.Parameter(
            vertices.clone().detach().requires_grad_(True).cuda().float()
        )
        self.vertices = vertices
        # ----------------------------------- FACES ---------------------------------- #
        # print("[DEBUG] Loaded faces from mesh")
        faces = mesh_scene.faces
        faces = torch.tensor(faces).cuda()
        self.faces = faces
        # --------------------------------- TRIANGLES -------------------------------- #
        # print("[DEBUG] Loaded triangles from mesh")
        triangles = vertices[torch.tensor(mesh_scene.faces).long()].float().cuda()
        self.triangles = triangles
        
        # if 'vertices' in params:
        #     self.vertices = params['vertices']
        # if 'triangles' in params:
        #     self.triangles = params['triangles']
        # if 'faces' in params:
        #     self.faces = params['faces']
        # # point_cloud = params['point_cloud']
        
        # # Check if vertices, triangles, faces are loaded correctly
        # print("[DEBUG]:", vertices.type(), triangles.type(), faces.type())
        # print("[DEBUG]:", self.vertices.type(), self.triangles.type(), self.faces.type())
        # # Check if vertices, triangles, faces have the same values as the ones loaded from the mesh
        # print("[DEBUG] Checking loaded vertices, triangles, faces against mesh...")
        # if torch.equal(self.vertices, vertices):
        #     print("[DEBUG] Loaded vertices match mesh vertices.")
        # else:
        #     print("[DEBUG] Loaded vertices do NOT match mesh vertices.")
        # if torch.equal(self.triangles, triangles):
        #     print("[DEBUG] Loaded triangles match mesh triangles.")
        # else:
        #     print("[DEBUG] Loaded triangles do NOT match mesh triangles.")
        # if torch.equal(self.faces, faces):
        #     print("[DEBUG] Loaded faces match mesh faces.")
        # else:
        #     print("[DEBUG] Loaded faces do NOT match mesh faces.")

        self._alpha = nn.Parameter(alpha)
        self._scale = nn.Parameter(scale)

    def _load_ply(self, path):
        plydata = PlyData.read(path)
        
        num_of_gs = plydata.elements[0].count
        xyz = np.stack((np.zeros(num_of_gs), 
                        np.zeros(num_of_gs), 
                        np.zeros(num_of_gs)), axis=1) # dummy xyz, not used for gs_mesh
        # xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
        #                 np.asarray(plydata.elements[0]["y"]),
        #                 np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((num_of_gs, 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((num_of_gs, len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [name for name in ["scale_0", "scale_1", "scale_2"]]
        scales = np.zeros((num_of_gs, len(scale_names)))
        # scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        # scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        # scales = np.zeros((xyz.shape[0], len(scale_names)))
        # for idx, attr_name in enumerate(scale_names):
        #     scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [name for name in ["rot_0", "rot_1", "rot_2", "rot_3"]]
        rots = np.zeros((num_of_gs, len(rot_names)))
        # rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        # rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        # rots = np.zeros((xyz.shape[0], len(rot_names)))
        # for idx, attr_name in enumerate(rot_names):
        #     rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)) #! [YC] comment out for now
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)) #! [YC] comment out for now

        self.active_sh_degree = self.max_sh_degree
        
    def load_pt(self, path):
        params = torch.load(path)
        alpha = params['_alpha']
        scale = params['_scale']
        if 'vertices' in params:
            self.vertices = params['vertices']
        if 'triangles' in params:
            self.triangles = params['triangles']
        if 'faces' in params:
            self.faces = params['faces']
        # point_cloud = params['point_cloud']
        self._alpha = nn.Parameter(alpha)
        self._scale = nn.Parameter(scale)

class LMGModel(GaussianModel):

    def __init__(self, sh_degree: int):

        super().__init__(sh_degree)
        self.point_cloud = None
        self._alpha = torch.empty(0)
        self._scale = torch.empty(0)
        self.alpha = torch.empty(0)
        self.softmax = torch.nn.Softmax(dim=2)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.update_alpha_func = self.softmax

        self.vertices = None
        self.faces = None
        self.triangles = None
        self.eps_s0 = 1e-8
        
        # >>>> [YC] add
        self.triangle_indices = None
        # <<<< [YC] add

    @property
    def get_xyz(self):
        return self._xyz

    # For training
    def create_from_pcd(self, pcd: MeshPointCloud, spatial_lr_scale: float):
        self.point_cloud = pcd
        self.triangles = self.point_cloud.triangles
        self.spatial_lr_scale = spatial_lr_scale
        
        self.triangle_indices = pcd.triangle_indices.cuda() # [YC] add
        self.num_splats_per_triangle = pcd.num_splats_per_triangle # [YC] add
        self.prev_num_splats_per_triangle = pcd.prev_num_splats_per_triangle # [YC] add
        self.alpha_indices = pcd.alpha_indices.cuda() # [YC] add

        alpha_point_cloud = pcd.alpha.float().cuda()
        scale = torch.ones((pcd.points.shape[0], 1)).float().cuda()

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        opacities = inverse_sigmoid(0.1 * torch.ones((pcd.points.shape[0], 1), dtype=torch.float, device="cuda"))

        self.vertices = nn.Parameter(
            self.point_cloud.vertices.clone().detach().requires_grad_(True).cuda().float()
        )
        self.faces = torch.tensor(self.point_cloud.faces).cuda()

        self._alpha = nn.Parameter(alpha_point_cloud.requires_grad_(True))
        self.update_alpha()
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scale = nn.Parameter(scale.requires_grad_(True))
        self.prepare_scaling_rot()
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
   
    
    def _calc_xyz(self):
        """
        calculate the 3d Gaussian center in the coordinates xyz.

        The alphas that are taken into account are the distances
        to the vertices and the coordinates of
        the triangles forming the mesh.

        """
        triangle_idx = self.triangles[self.triangle_indices].detach()  # constants
        _xyz = torch.bmm(self.alpha.unsqueeze(1), triangle_idx).squeeze(1)
        self._xyz = _xyz.detach()
        
    def prepare_scaling_rot(self):
        """
        approximate covariance matrix and calculate scaling/rotation tensors

        covariance matrix is [v0, v1, v2], where
        v0 is a normal vector to each face
        v1 is a vector from centroid of each face and 1st vertex
        v2 is obtained by orthogonal projection of a vector from
        centroid to 2nd vertex onto subspace spanned by v0 and v1.
        """
        def dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)
        
        def proj(v, u):
            """
            projection of vector v onto subspace spanned by u

            vector u is assumed to be already normalized
            """
            coef = dot(v, u)
            return coef * u

        triangles = self.triangles
        normals = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1
        )
        v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + self.eps_s0)
        means = torch.mean(triangles, dim=1)
        v1 = triangles[:, 1] - means
        v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + self.eps_s0
        v1 = v1 / v1_norm
        v2_init = triangles[:, 2] - means
        v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1)  # Gram-Schmidt
        v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + self.eps_s0)

        s1 = v1_norm / 2.
        s2 = dot(v2_init, v2) / 2.
        s0 = self.eps_s0 * torch.ones_like(s1)
        scales = torch.concat((s0, s1, s2), dim=1).unsqueeze(dim=1)
        
        # scales = scales.broadcast_to((*self.alpha.shape[:2], 3))
        # self._scaling = torch.log(
        #     torch.nn.functional.relu(self._scale * scales.flatten(start_dim=0, end_dim=1)) + self.eps_s0
        # )
        # rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
        # rotation = rotation.broadcast_to((*self.alpha.shape[:2], 3, 3)).flatten(start_dim=0, end_dim=1)
        # rotation = rotation.transpose(-2, -1)
        # self._rotation = rot_to_quat_batch(rotation)
        
        scales = scales[self.triangle_indices]
        with torch.no_grad():
            s_input = self._scale * scales.flatten(start_dim=0, end_dim=1)
            s_input = torch.nn.functional.relu(s_input) + self.eps_s0
            new_scaling = torch.log(s_input)
        self._scaling = new_scaling.detach()
        
        rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
        # rotation = rotation.broadcast_to((*self.alpha.shape[:2], 3, 3)).flatten(start_dim=0, end_dim=1)
        rotation = rotation[self.triangle_indices]
        rotation = rotation.transpose(-2, -1)
        # print("[--> DEBUG] GaussianMeshModel::_rotation:", self._rotation)
        self._rotation = rot_to_quat_batch(rotation) 
        
    def update_alpha(self):
        """
        Function to control the alpha value.

        Alpha is the distance of the center of the gauss
         from the vertex of the triangle of the mesh.
        Thus, for each center of the gauss, 3 alphas
        are determined: alpha1 + alpha2 + alpha3.
        For a point to be in the center of the vertex,
        the alphas must meet the assumptions:
        alpha1 + alpha2 + alpha3 = 1
        and alpha1 + alpha2 + alpha3 >= 0
        
        Implementation:
        Make sure all alphas are positive by applying ReLU
        Then normalize them so that they sum to 1 for each triangle.
        """
        self.alpha = torch.relu(self._alpha) + 1e-8
        self.alpha = self.alpha / self.alpha.sum(dim=-1, keepdim=True)
        if self.vertices is not None and self.faces is not None:
            self.triangles = self.vertices[self.faces]
            self._calc_xyz()

    def update_alpha_triangle(self):
        """
        Update alpha values triangle by triangle instead of all at once.
        """
    
    def training_setup(self, training_args):
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l_params = [
            {'params': [self.vertices], 'lr': training_args.vertices_lr, "name": "vertices"},
            {'params': [self._alpha], 'lr': training_args.alpha_lr, "name": "alpha"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scale], 'lr': training_args.scaling_lr, "name": "scaling"}
        ]

        self.optimizer = torch.optim.Adam(l_params, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration) -> None:
        """ Learning rate scheduling per step """
        pass

    def save_ply(self, path):
        self.update_alpha()
        self.prepare_scaling_rot()
        self._save_ply(path)

        attrs = self.__dict__
        additional_attrs = [
            '_alpha',
            '_scale',
            'triangle_indices',
            'num_splats_per_triangle',
            'alpha_indices' # [YC] add
        ]

        save_dict = {}
        for attr_name in additional_attrs:
            save_dict[attr_name] = attrs[attr_name]
        if 'round_id' in attrs:
            # Per-splat provenance (which round created it), set by the in-memory
            # orchestrator. Not present on checkpoints from the older bash pipeline.
            save_dict['round_id'] = attrs['round_id']

        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        torch.save(save_dict, path_model)

    def load_lmg_gs(self, path, vertices, faces, alpha_path=None, trainable=True):
        if trainable:
            self._load_lmg_ply(path)
        else:
            self._load_lmg_foundation_ply(path)

        # Load from pt file
        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        params = torch.load(path_model)

        # # Load static alpha
        # static_alpha = torch.load(alpha_path)

        alpha = params['_alpha']
        scale = params['_scale']

        if "triangle_indices" in params:
            triangle_indices = params['triangle_indices']
        else:
            triangle_indices = params['point_cloud'].triangle_indices

        num_splats_per_triangle = params['num_splats_per_triangle'] # [YC] add
        alpha_indices = params['alpha_indices'] # [YC] add

        # --------------------------------- VERTICES --------------------------------- #
        self.vertices = nn.Parameter(
            vertices.clone().detach().requires_grad_(True).cuda().float()
        )
        # ----------------------------------- FACES ---------------------------------- #
        self.faces = torch.tensor(faces).cuda()
        # --------------------------------- TRIANGLES -------------------------------- #
        self.triangles = vertices[torch.tensor(faces).long()].float().cuda()

        self._alpha = nn.Parameter(alpha)
        self._scale = nn.Parameter(scale)
        self.triangle_indices = triangle_indices
        self.num_splats_per_triangle = num_splats_per_triangle # [YC] add
        self.alpha_indices = alpha_indices # [YC] add

        self.update_alpha()
        self.prepare_scaling_rot()

    def _load_lmg_ply(self, path):
        plydata = PlyData.read(path)
        
        num_of_gs = plydata.elements[0].count
        
        # Dummy xyz
        xyz = np.stack((np.zeros(num_of_gs), 
                        np.zeros(num_of_gs), 
                        np.zeros(num_of_gs)), axis=1) # dummy xyz, not used for gs_mesh
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((num_of_gs, 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((num_of_gs, len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # Dummy scale
        scale_names = [name for name in ["scale_0", "scale_1", "scale_2"]]
        scales = np.zeros((num_of_gs, len(scale_names)))
        
        # Dummy rotations
        rot_names = [name for name in ["rot_0", "rot_1", "rot_2", "rot_3"]]
        rots = np.zeros((num_of_gs, len(rot_names)))
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)) #! [YC] comment out for now
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)) #! [YC] comment out for now

        self.active_sh_degree = self.max_sh_degree
    
    def _load_lmg_foundation_ply(self, path):
        plydata = PlyData.read(path)
        
        num_of_gs = plydata.elements[0].count
        
        # Dummy xyz
        xyz = np.stack((np.zeros(num_of_gs), 
                        np.zeros(num_of_gs), 
                        np.zeros(num_of_gs)), axis=1) # dummy xyz, not used for gs_mesh
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((num_of_gs, 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((num_of_gs, len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # Dummy scale
        scale_names = [name for name in ["scale_0", "scale_1", "scale_2"]]
        scales = np.zeros((num_of_gs, len(scale_names)))
        
        # Dummy rotations
        rot_names = [name for name in ["rot_0", "rot_1", "rot_2", "rot_3"]]
        rots = np.zeros((num_of_gs, len(rot_names)))
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False)) #! [YC] comment out for now
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False)) #! [YC] comment out for now

        self.active_sh_degree = self.max_sh_degree
    
    def setup_frozen_mask(self, frozen_indices):
        """
        設定哪些 Gaussians 需要被凍結（在訓練時不更新參數）。
        參數:
            frozen_indices (torch.Tensor): 包含需要凍結的索引的 1D Tensor。
        """
        # 1. 取得當前的設備與總數量，確保變數都在同一個 GPU 上
        device = self._xyz.device
        total_num = self._xyz.shape[0]
        
        # 確保傳入的 index 也在正確的 device 上
        frozen_indices = frozen_indices.to(device)

        # 2. 建立 Boolean Mask (預設全為 False，代表都要訓練)
        self.frozen_mask = torch.zeros(total_num, dtype=torch.bool, device=device)
        
        # 將名單上的點設為 True (代表要凍結)
        self.frozen_mask[frozen_indices] = True

        # 3. 定義攔截梯度的 Hook 函數
        def freeze_hook(grad):
            # 為了避免 PyTorch 抱怨 in-place 修改破壞計算圖，我們 clone 一份梯度
            new_grad = grad.clone()
            # 將 mask 為 True 的位置，梯度強行歸零
            new_grad[self.frozen_mask] = 0.0
            return new_grad

        # 4. 將 Hook 註冊到所有 3DGS 的核心參數上
        # 配合你的 training_setup，更新需要綁定 Hook 的參數名單
        parameters_to_freeze = [
            # 'vertices',       # 修改這裡 (對應原本的 _xyz)
            '_alpha',         # 修改這裡 (對應原本的 _rotation 或其他特徵)
            '_features_dc', 
            '_features_rest', 
            '_opacity', 
            '_scale'          # 修改這裡 (對應原本的 _scaling)
        ]

        for param_name in parameters_to_freeze:
            if hasattr(self, param_name):
                param = getattr(self, param_name)
                # 確保該變數存在，且真的有在計算梯度才掛上 Hook
                if param is not None and param.requires_grad:
                    param.register_hook(freeze_hook)
                    
        print(f"[Info] 成功凍結了 {len(frozen_indices)} / {total_num} 個 Gaussians！")  
        
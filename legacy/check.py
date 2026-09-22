import cv2
import numpy as np

def generate_difference_heatmap(img1_path, img2_path, output_path):
    # 1. Load the images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Check if images loaded successfully
    if img1 is None or img2 is None:
        raise ValueError("Could not open or find one of the images.")
        
    # 2. Ensure both images are the same size
    if img1.shape != img2.shape:
        print(f"Resizing image 2 from {img2.shape[:2]} to match image 1 {img1.shape[:2]}")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 3. Convert to grayscale (FIXED)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 4. Calculate the absolute difference pixel-by-pixel
    diff = cv2.absdiff(gray1, gray2)
    
    # 5. Optional: Apply a slight blur to smooth out tiny compression noise
    diff_blurred = cv2.GaussianBlur(diff, (5, 5), 0)
    
    # 6. Apply the heatmap colormap
    heatmap = cv2.applyColorMap(diff_blurred, cv2.COLORMAP_JET)
    
    # 7. Save the final heatmap image
    cv2.imwrite(output_path, heatmap)
    print(f"Heatmap successfully saved to: {output_path}")
    

generate_difference_heatmap('/mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/distortion_debug_visualization/20260519_064034/heatmap/r_0.png', 
                            '/mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/distortion_debug_visualization/20260519_051406/heatmap/r_0.png', 
                            'difference_heatmap.jpg')
# generate_difference_heatmap('/mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/distortion_debug_visualization/20260518_183519/heatmap/r_0.png', 
#                             '/mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/distortion_debug_visualization/20260519_051406/heatmap/r_0.png', 
#                             'difference_heatmap.jpg')


    
# import torch
# import numpy as np
# from plyfile import PlyData, PlyElement

# from torch import nn

# from scene.gaussian_model import GaussianModel
# from utils.general_utils import inverse_sigmoid, rot_to_quat_batch
# from utils.sh_utils import RGB2SH
# from games.mesh_splatting.utils.graphics_utils import MeshPointCloud

# path_model = "/mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/output/sample_exp/hotdog/distortion_500_occlusion/point_cloud/iteration_0/model_params.pt"
# params = torch.load(path_model)
# print(params.keys())
# print(params['point_cloud'].triangle_indices[:5])

# new_path_model = "/mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/output/sample_exp_dynamic/hotdog/distortion_500_occlusion/point_cloud/iteration_0/model_params.pt"
# new_params = torch.load(new_path_model)
# print(new_params.keys())
# print(new_params['triangle_indices'][:5])
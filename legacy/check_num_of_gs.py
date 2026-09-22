import torch
import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt

from torch import nn

from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid, rot_to_quat_batch
from utils.sh_utils import RGB2SH
from games.mesh_splatting.utils.graphics_utils import MeshPointCloud

# Progressive
path_model = "/mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_2000/point_cloud/iteration_3000/model_params.pt"
params = torch.load(path_model)
array1 = params['num_splats_per_triangle']

# LMG
new_path_model = "/mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/output/progressive/hotdog/distortion_progressive_30000_occlusion/iteration_0/point_cloud/iteration_3000/model_params.pt"
new_params = torch.load(new_path_model)
array2 = new_params['num_splats_per_triangle']

array_diff = array1 - array2

# 建立 X 軸的索引位置
x_indices = np.arange(len(array1))

# 3. 開始畫圖 (設定一個 1x3 的畫布，讓三張圖並排顯示)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 第一張圖：Array 1
axes[0].bar(x_indices, array1, color='skyblue', edgecolor='black')
axes[0].set_title('Array 1')
axes[0].set_xlabel('Index')
axes[0].set_ylabel('Value')

# 第二張圖：Array 2
axes[1].bar(x_indices, array2, color='salmon', edgecolor='black')
axes[1].set_title('Array 2')
axes[1].set_xlabel('Index')

# 第三張圖：相減後的結果 (Array 1 - Array 2)
# 這裡加了一條 horizontal line (hlines) 在 y=0 的位置，方便看正負值
axes[2].bar(x_indices, array_diff, color='lightgreen', edgecolor='black')
axes[2].axhline(0, color='gray', linestyle='--', linewidth=1) 
axes[2].set_title('Difference (Array 1 - Array 2)')
axes[2].set_xlabel('Index')

# 自動調整佈局避免文字重疊
plt.tight_layout()

# 顯示圖表
plt.show()
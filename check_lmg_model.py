from games import (
    gaussianModel
)

gaussians_pt = gaussianModel["gs_mesh"](3) # [YC] note: nothing changing here
gaussians_pt.load_pt("/mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/output/sample_exp/hotdog/distortion_100_occlusion/point_cloud/iteration_1000/model_params.pt")
gaussians_pt.update_alpha()
print(gaussians_pt._xyz[:5])

gaussians_ply = gaussianModel["gs_mesh"](3)
gaussians_ply.load_ply("/mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/output/sample_exp/hotdog/distortion_100_occlusion/point_cloud/iteration_1000/point_cloud.ply")
print(gaussians_ply._xyz[:5])
from argparse import ArgumentParser, Namespace, Namespace

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", type=str, required=True, help="Path to the model directory.")
    args = parser.parse_args()
    
    print(f"Model directory: {args.m}")
    # print(f"Gaussian Splatting type: {args.gs_type}")
    # print(f"Skip training: {args.skip_train}")
    # print(f"Occlusion handling: {args.occlusion}")
    
    # Generate a dummy cfg_args based on the provided arguments
    # Namespace(alloc_policy='distortion', 
    #             budget_per_tri=1.0, 
    #             data_device='cuda', 
    #             eval=True,
    #             gs_type='gs_mesh', 
    #             images='images', 
    #             mesh_type='milo', 
    #             meshes=[], 
    #             model_path='output/sample_exp/hotdog/distortion_40000_occlusion/', 
    #             num_splats=[2], 
    #             resolution=-1, 
    #             sh_degree=3, 
    #             source_path='/mnt/data1/syjintw/MMSys26_extension/lmg/layered-mesh-gaussian/dataset/images/hotdog', 
    #             total_splats=40000, 
    #             warmup_only=False, 
    #             white_background=False)
    
    cfg_args = {
        "alloc_policy": "distortion",
        "budget_per_tri": 1.0,
        "data_device": "cuda",
        "eval": True,
        "gs_type": "gs_mesh",
        "images": "images",
        "mesh_type": "milo",
        "meshes": [],
        "model_path": "output/sample_exp/hotdog/distortion_40000_occlusion/",
        "num_splats": [2],
        "resolution": -1,
        "sh_degree": 3,
        "source_path": "/mnt/data1/syjintw/MMSys26_extension/lmg/layered-mesh-gaussian/dataset/images/hotdog",
        "total_splats": 40000,
        "warmup_only": False,
        "white_background": False
    }
    
    save_path = f"{args.m}/cfg_args"
    with open(save_path, "w") as f:
        f.write(str(Namespace(**cfg_args)))
    print(f"Dummy cfg_args saved to {save_path}")

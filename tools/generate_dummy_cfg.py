from argparse import ArgumentParser, Namespace

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("-s", type=str, default="", help="Dataset source_path (must match the trained scene).")
    args = parser.parse_args()

    print(f"Model directory: {args.m}")

    # cfg_args is read by render's get_combined_args. Only source_path/model_path matter here
    # (render overrides the rest via CLI). source_path MUST match the scene, else GT is wrong.
    cfg_args = {
        "alloc_policy": "distortion",
        "budget_per_tri": 1.0,
        "data_device": "cuda",
        "eval": True,
        "gs_type": "gs_mesh",
        "images": "images",
        "mesh_type": "milo",
        "meshes": [],
        "model_path": args.m,
        "num_splats": [2],
        "resolution": -1,
        "sh_degree": 3,
        "source_path": args.s,
        "total_splats": 40000,
        "warmup_only": False,
        "white_background": False
    }

    save_path = f"{args.m}/cfg_args"
    with open(save_path, "w") as f:
        f.write(str(Namespace(**cfg_args)))
    print(f"Dummy cfg_args saved to {save_path}")

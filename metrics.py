#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(gs_type, model_paths, skip_lpips=False):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        #try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / f"renders_{gs_type}"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                # TODO: also store these
                # - scene names
                # - #iterations
                # - SH degree
                # - #GS(budget)
                # - policy name
                # - file size (mesh + gs)
                # - time to train/render
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    
                    # [NOTE] skip LPIPS to save time for now
                    if skip_lpips:
                        lpipss.append(-1.0)
                    else:
                        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    
                    

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                # Extra provenance/cost fields (additive; existing readers key on PSNR/SSIM/LPIPS).
                extra = {"n_views": len(renders)}
                try:
                    extra["iteration"] = int(str(method).split("_")[-1])
                except (ValueError, IndexError):
                    pass
                ply = os.path.join(scene_dir, "point_cloud", f"iteration_{extra.get('iteration')}", "point_cloud.ply")
                if os.path.exists(ply):
                    extra["gs_ply_bytes"] = os.path.getsize(ply)
                for tj, pfx in [("train_timing.json", "train"), ("render_timing_test.json", "render")]:
                    tp = os.path.join(scene_dir, tj)
                    if os.path.exists(tp):
                        try:
                            td = json.load(open(tp))
                            if f"{pfx}_secs" in td: extra[f"{pfx}_secs"] = td[f"{pfx}_secs"]
                            if "peak_vram_mb" in td: extra[f"{pfx}_peak_vram_mb"] = td["peak_vram_mb"]
                        except (ValueError, OSError):
                            pass
                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item(),
                                                    **extra})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + f"/results_{gs_type}.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + f"/per_view_{gs_type}.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
                
            print(f"[INFO] Saved results to {scene_dir}/results_{gs_type}.json and per_view_{gs_type}.json")
        #except:
        #    print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Metrics script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--gs_type', type=str, default="gs_flat")
    parser.add_argument('--skip_lpips', action="store_true", help="Skip LPIPS computation to save time")
    args = parser.parse_args()
    evaluate(args.gs_type, args.model_paths, args.skip_lpips)

"""
python metrics.py \
-m /mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/output/non_progressive/hotdog/distortion_progressive_40000_occlusion/iteration_0 \
--gs_type lmg \
--skip_lpips
"""
    
"""
python metrics.py \
-m /mnt/data1/syjintw/MMSys26_extension/layered-mesh-gaussian/output/progressive/hotdog/distortion_progressive_10000_occlusion/iteration_45000 \
--gs_type lmg \
--skip_lpips
"""

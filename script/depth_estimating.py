import os
import sys

# Add Grounded-SAM-2 path manually
gsa2_path = os.path.join(os.path.dirname(__file__), "../FoundationStereo")
sys.path.append(gsa2_path)

import torch
import argparse
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
from depth_estimator import DepthEstimator


torch.autograd.set_grad_enabled(False)


def get_paths(directory, valid_ext=[".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]):
    cands = [p for p in os.listdir(directory) if os.path.splitext(p)[1] in valid_ext]
    numeric = []
    for p in cands:
        stem = os.path.splitext(p)[0]
        if stem.isdigit():
            numeric.append((int(stem), p))
    numeric.sort(key=lambda x: x[0])
    return [os.path.join(directory, p) for _, p in numeric]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video segmentation with Grounded-SAM2")
    parser.add_argument("left_image_dir", type=str, help="Directory to left image frames.")
    parser.add_argument("right_image_dir", type=str, help="Directory to right image frames.")
    parser.add_argument("intrinsic_dir", type=str, help="Directory to intrinsics.")
    parser.add_argument("--baseline", type=float, default=0.063, help="Baseline between left and right caemras.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save estimated depths.")
    args = parser.parse_args()

    left_images = get_paths(args.left_image_dir)
    right_images = get_paths(args.right_image_dir)
    intrinsics = get_paths(args.intrinsic_dir, valid_ext=[".txt"])

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(args.left_image_dir), "est_depth")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    depth_estimator = DepthEstimator()

    for i, (lcpath, rcpath, Kpath) in tqdm(enumerate(zip(left_images, right_images, intrinsics)), total=len(left_images)):
        left_image = imageio.imread(lcpath)
        right_image = imageio.imread(rcpath)
        K = np.loadtxt(Kpath)

        depth = depth_estimator.predict(
            left_image,
            right_image,
            K,
            args.baseline,
        )

        save_path = os.path.join(save_dir, f"{i:06d}.png")
        imageio.imwrite(save_path, (1000*depth).astype(np.uint16))

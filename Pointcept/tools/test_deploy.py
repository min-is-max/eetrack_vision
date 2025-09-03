#!/usr/bin/env python3
# tools/test_deploy_min.py
"""
Minimal deploy-time inference for welding line segmentation (no GT, no Open3D).
- Loads model from config + weight
- Runs inference on WeldingDataset (test split)
- Fits a robust spline on predicted edge points
- Saves:
    - pred_mask.npy                (boolean mask over points: edge==1)
    - pred_points.npy              (Nx3 predicted edge points in world coords)
    - curve_points.npy             (Mx3 fitted curve points in world coords; if fit ok)
    - preview_curve.png            (2D PCA-projection preview with curve overlay)
    - meta.json                    (basic info)
"""

import os
import json
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pointcept imports
from pointcept.datasets import build_dataset
from pointcept.models import build_model
from pointcept.utils.misc import make_dirs, robust_curve_from_pointcloud
from pointcept.utils.logger import get_root_logger
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
import pointcept.utils.comm as comm


def build_model_and_load(cfg):
    logger = get_root_logger()
    logger.info("=> Building model ...")
    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Num params: {n_params}")

    if not os.path.isfile(cfg.weight):
        raise RuntimeError(f"=> No checkpoint found at '{cfg.weight}'")
    logger.info(f"Loading weight at: {cfg.weight}")

    checkpoint = torch.load(cfg.weight, weights_only=False, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    # strip "module." if present
    new_state = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items())
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        logger.warning(f"Missing keys in state_dict: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys in state_dict: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")
    logger.info(f"=> Loaded weight '{cfg.weight}' (epoch {checkpoint.get('epoch','?')})")

    model = model.cuda().eval()
    return model


def build_test_loader(cfg):
    logger = get_root_logger()
    logger.info("=> Building test dataset & dataloader ...")
    test_dataset = build_dataset(cfg.data.test)
    logger.info(f"Totally {len(test_dataset)} x 1 samples in {getattr(test_dataset, 'split', 'test')} set.")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_worker_per_gpu if hasattr(cfg, "num_worker_per_gpu") else 1,
        pin_memory=True,
        collate_fn=lambda b: b,   # keep as list; we’ll index [0]
    )
    return test_loader


def sanitize_points(arr):
    """Ensure (N,3) float64, drop non-finite rows."""
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim == 1 and x.size == 3:
        x = x[None, :]
    elif x.ndim == 3:
        x = x.reshape(-1, 3)
    if x.ndim != 2 or x.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float64)
    mask = np.isfinite(x).all(axis=1)
    return x[mask]


def pca_project(points3d):
    """
    PCA로 3D -> 2D 투영용 로컬 좌표를 만든다.
    반환: (N,2), dict(mean, U)  (U: 3x3 주성분)
    """
    P = sanitize_points(points3d)
    if P.shape[0] < 2:
        return np.zeros((0, 2)), {"mean": None, "U": None}
    mu = P.mean(axis=0)
    X = P - mu
    # SVD 기반 PCA
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # Vt (3x3)의 상위 2개 주성분
    PC = Vt[:2, :]          # 2x3
    Y = (X @ PC.T)          # (N,2)
    return Y, {"mean": mu, "U": Vt}


def preview_curve_png(out_path: Path, cloud3d: np.ndarray, curve3d: np.ndarray | None):
    """
    전체 포인트(연한 회색) + 피팅 곡선(빨강) 2D 프리뷰를 저장.
    - PCA 평면에 투영
    - 곡선이 있으면 같은 평면으로 투영해서 라인으로 그림
    """
    cloud2d, pinfo = pca_project(cloud3d)
    plt.figure(figsize=(6, 6), dpi=160)
    if cloud2d.shape[0] > 0:
        plt.scatter(cloud2d[:, 0], cloud2d[:, 1], s=0.2, c="#aaaaaa", linewidths=0, alpha=0.8)

    if curve3d is not None and curve3d.shape[0] >= 2 and pinfo["U"] is not None:
        mu = pinfo["mean"]
        Vt = pinfo["U"]
        PC = Vt[:2, :]  # 2x3
        C = sanitize_points(curve3d)
        C2 = (C - mu) @ PC.T  # (M,2)
        plt.plot(C2[:, 0], C2[:, 1], "-", color="red", linewidth=2, alpha=0.95)

    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path.as_posix(), bbox_inches="tight", pad_inches=0)
    plt.close()


def run_once(model, data_dict, out_dir: Path, idx: int):
    """
    data_dict keys (WeldingDataset/test transform):
      - coord: (N,3) float tensor
      - color: (N,3) uint8 tensor (optional)
      - (no GT usage)
    """
    logger = get_root_logger()

    # Move tensors to CUDA
    input_dict = {}
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        else:
            input_dict[k] = v

    # Forward
    with torch.no_grad():
        out = model(input_dict)
    seg_logits = out["seg_logits"]                   # (N, num_classes)
    pred_cls = seg_logits.max(1)[1]                  # (N,)
    pred_mask = (pred_cls == 1)                      # edge=1

    # Coordinates → numpy
    coord = data_dict["coord"]
    coord_np = coord.cpu().numpy() if isinstance(coord, torch.Tensor) else np.asarray(coord)

    pred_mask_np = pred_mask.detach().cpu().numpy().astype(bool)
    pred_pts = coord_np[pred_mask_np]

    # --- Robust curve fit on predicted edge points ---
    curve_pts = None
    if pred_pts.shape[0] >= 5:  # 최소 몇 점 이상일 때만 시도
        try:
            res = robust_curve_from_pointcloud(
                pred_pts,
                voxel_size=0.02, sor_k=16, sor_std_ratio=1.0,
                knn_k=8, spline_s=0.0005, iter_refine=2,
                outlier_thresh_factor=3.0, plot=False
            )
            # 결과 키 정규화
            curve_pts = res.get("curve_points", None)
            if curve_pts is None:
                curve_pts = res.get("curve", None)
            if curve_pts is not None and len(curve_pts) < 2:
                curve_pts = None
        except Exception as e:
            logger.warning(f"[sample_{idx:06d}] curve fit error: {repr(e)}")

    # Save
    name = f"sample_{idx:06d}"
    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)

    np.save(case_dir / "pred_mask.npy", pred_mask_np)
    np.save(case_dir / "pred_points.npy", sanitize_points(pred_pts))
    meta = {
        "num_points": int(coord_np.shape[0]),
        "num_pred_edge_points": int(pred_pts.shape[0]),
        "has_curve": bool(curve_pts is not None),
    }
    with open(case_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    if curve_pts is not None:
        np.save(case_dir / "curve_points.npy", sanitize_points(curve_pts))

    # 2D preview (PCA projection)
    try:
        preview_curve_png(case_dir / "preview_curve.png", coord_np, curve_pts)
    except Exception as e:
        logger.warning(f"[sample_{idx:06d}] preview save failed: {repr(e)}")

    logger.info(f"[{name}] pred_edge_pts={pred_pts.shape[0]}, curve={'ok' if curve_pts is not None else 'none'}")


def parse_args():
    p = default_argument_parser()
    return p.parse_args()


def main():
    args = parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    cfg = default_setup(cfg)

    # 출력 디렉토리
    save_path = Path(cfg.save_path)
    out_dir = save_path / "deploy_result"
    make_dirs(out_dir.as_posix())

    # 모델/데이터
    model = build_model_and_load(cfg)
    test_loader = build_test_loader(cfg)

    logger = get_root_logger()
    logger.info(">>>>>>>>>>>>>>>>> Start Deploy Inference >>>>>>>>>>>>>>>>>")
    for idx, batch in enumerate(test_loader):
        data_dict = batch[0]  # bs=1
        run_once(model, data_dict, out_dir, idx)
    logger.info("<<<<<<<<<<<<<<<<< End Deploy Inference <<<<<<<<<<<<<<<<<")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
import cv2

def project_points(K, pts_xyz):
    """
    pts_xyz: (N,3) in camera coords
    returns: (N,2) pixel coords (u,v), float
    """
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    X, Y, Z = pts_xyz[:,0], pts_xyz[:,1], pts_xyz[:,2]
    eps = 1e-9
    Z = np.where(Z==0, eps, Z)
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return np.stack([u, v], axis=1)

def load_curve_points(result_dir: Path, sample_id: str, sample_dir: Path):
    """
    우선 순위:
      1) tester가 저장한 곡선 (curve_pred_{id}.npy or test_{idx}_curve.npy 등)
      2) 없으면 preprocess의 visible_edge.npy를 '대체 곡선'으로 사용
    """
    # 1) 여러 네이밍 후보 탐색
    candidates = [
        result_dir / f"curve_pred_{sample_id}.npy",
        result_dir / f"{sample_id}_curve_pred.npy",
        result_dir / f"test_{int(sample_id)}_curve.npy" if sample_id.isdigit() else None,
    ]
    for c in candidates:
        if c is not None and c.exists():
            arr = np.load(c)
            arr = np.asarray(arr, dtype=np.float64).reshape(-1,3)
            return arr

    # 2) fallback: visible_edge.npy
    fallback = sample_dir / "visible_edge.npy"
    if fallback.exists():
        arr = np.load(fallback).astype(np.float64).reshape(-1,3)
        return arr

    return None

def main():
    ap = argparse.ArgumentParser("Overlay predicted curve on original left images")
    ap.add_argument("--data_root", type=str, required=True, help="Pointcept data root (…/Pointcept/data/welding_deploy)")
    ap.add_argument("--result_dir", type=str, required=True, help="Tester save_path/result directory")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--out_prefix", type=str, default="overlay_left_")
    ap.add_argument("--thickness", type=int, default=2)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    split_dir = data_root / args.split
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted([p for p in split_dir.glob("*") if p.is_dir()])
    if len(sample_dirs) == 0:
        print(f"[overlay] no samples under {split_dir}")
        return

    for sd in sample_dirs:
        sid = sd.name  # sample_id "000000" …
        meta_path = sd / "meta.json"
        K_path = sd / "K.npy"
        if (not meta_path.exists()) or (not K_path.exists()):
            print(f"[overlay] skip {sid}: missing meta.json or K.npy")
            continue

        # load meta/K
        try:
            meta = json.loads((meta_path).read_text())
            left_img_path = meta.get("left_image_path", None)
            if left_img_path is None:
                print(f"[overlay] skip {sid}: meta has no left_image_path")
                continue
            left_img = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
            if left_img is None:
                print(f"[overlay] skip {sid}: cannot read left image {left_img_path}")
                continue
        except Exception as e:
            print(f"[overlay] skip {sid}: meta err {repr(e)}")
            continue

        K = np.load(K_path).astype(np.float64).reshape(3,3)

        # load curve (pred > fallback)
        curve = load_curve_points(result_dir, sid, sd)
        if curve is None or curve.shape[0] < 2:
            print(f"[overlay] skip {sid}: no curve points")
            continue

        # project and draw
        H, W = left_img.shape[:2]
        uv = project_points(K, curve)
        uv = np.round(uv).astype(int)
        # clamp
        uv[:,0] = np.clip(uv[:,0], 0, W-1)
        uv[:,1] = np.clip(uv[:,1], 0, H-1)

        # polyline
        overlay = left_img.copy()
        for i in range(1, uv.shape[0]):
            p0 = tuple(uv[i-1].tolist())
            p1 = tuple(uv[i].tolist())
            cv2.line(overlay, p0, p1, (0, 255, 255), args.thickness, cv2.LINE_AA)  # BGR: yellow

        out_path = result_dir / f"{args.out_prefix}{sid}.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"[overlay] saved {out_path}")

if __name__ == "__main__":
    main()

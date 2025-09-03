#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay visible_edge / spline curve onto the original left RGB image.

Inputs:
- Trajectory folder (with left/color & left/intrinsic)
- Pointcept preprocessed outputs (…/Pointcept/data/welding*_*/{split}/{frame_id}/*.npy)

For each sample_id under the split, this script:
  1) Loads left/color/{sample_id}.(png|jpg)
  2) Loads K (left/intrinsic/{sample_id}.txt OR single K.txt)
  3) Loads visible_edge.npy and spl_t.npy/spl_c.npy/spl_k.npy
  4) Projects 3D points to image plane and draws:
       - visible_edge: green (small dots + thin polyline)
       - spline curve: magenta (polyline)
  5) Saves overlay image to out_dir/overlay_{sample_id}.png
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import imageio.v2 as imageio
from scipy.interpolate import splev

def _load_K_for_frame(intrin_dir: Path, sample_id: str) -> np.ndarray:
    """
    Load K:
    - If only one txt exists: use it for all frames
    - Else, try to match {sample_id}.txt, or fallback to sorted[0]
    """
    ks = sorted(intrin_dir.glob("*.txt"))
    if len(ks) == 0:
        raise FileNotFoundError(f"No intrinsic txt found in: {intrin_dir}")
    if len(ks) == 1:
        return np.loadtxt(ks[0]).astype(np.float32)
    # multiple txts
    exact = intrin_dir / f"{sample_id}.txt"
    if exact.exists():
        return np.loadtxt(exact).astype(np.float32)
    return np.loadtxt(ks[0]).astype(np.float32)

def _find_left_image(color_dir: Path, sample_id: str) -> Path:
    """
    Find {sample_id}.png|jpg|jpeg under left/color
    """
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        p = color_dir / f"{sample_id}{ext}"
        if p.exists():
            return p
    # fallback: any image (rare)
    cands = sorted(list(color_dir.glob("*.*")))
    if not cands:
        raise FileNotFoundError(f"No left/color image found for id={sample_id} in {color_dir}")
    return cands[0]

def _project_xyz_to_uv(xyz: np.ndarray, K: np.ndarray):
    """
    xyz: (N,3), K: (3,3)
    returns (u,v) float arrays of shape (N,), mask valid (Z>0)
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    z = xyz[:, 2]
    valid = z > 1e-6
    u = np.empty_like(z); v = np.empty_like(z)
    u[:] = np.nan; v[:] = np.nan
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u[valid] = (xyz[valid, 0] * fx / z[valid]) + cx
    v[valid] = (xyz[valid, 1] * fy / z[valid]) + cy
    return u, v, valid

def _draw_polyline(img: np.ndarray, uv: np.ndarray, color_bgr, thickness=2, closed=False):
    """
    uv: (N,2) float -> clipped to integer inside image; ignores out-of-bounds
    """
    H, W = img.shape[:2]
    uv_int = np.round(uv).astype(np.int32)
    # keep points inside image
    m = (uv_int[:,0] >= 0) & (uv_int[:,0] < W) & (uv_int[:,1] >= 0) & (uv_int[:,1] < H)
    pts = uv_int[m]
    if pts.shape[0] >= 2:
        cv2.polylines(img, [pts.reshape(-1,1,2)], isClosed=closed, color=color_bgr, thickness=thickness)

def _draw_points(img: np.ndarray, uv: np.ndarray, color_bgr, radius=1):
    H, W = img.shape[:2]
    uv_int = np.round(uv).astype(np.int32)
    m = (uv_int[:,0] >= 0) & (uv_int[:,0] < W) & (uv_int[:,1] >= 0) & (uv_int[:,1] < H)
    pts = uv_int[m]
    for (x, y) in pts:
        cv2.circle(img, (int(x), int(y)), radius, color_bgr, -1, lineType=cv2.LINE_AA)

def _eval_bspline_points(spl_t: np.ndarray, spl_c: np.ndarray, spl_k: int, n_samples=600) -> np.ndarray:
    """
    Reconstruct dense points from (t, c, k). We saved spl_c as (3, Ncoef)
    Returns (n_samples, 3)
    """
    t = np.asarray(spl_t, dtype=np.float64).ravel()
    c = np.asarray(spl_c, dtype=np.float64)
    if c.ndim == 1:
        c = c[None, :]  # (1, Ncoef)
    # SciPy expects tuple (t, [cx,cy,cz], k)
    if c.shape[0] == 3:
        c_tuple = (c[0], c[1], c[2])
    elif c.shape[0] == 2:
        c_tuple = (c[0], c[1])  # 2D curve
    else:
        # fallback: treat first 3 rows as xyz
        c_tuple = (c[0], c[1], c[2])
    k = int(np.asarray(spl_k).ravel()[0])
    u = np.linspace(0.0, 1.0, n_samples)
    out = splev(u, (t, c_tuple, k))
    pts = np.vstack(out).T  # (n, dim)
    # if 2D, promote Z=1.0 to avoid divide-by-zero later
    if pts.shape[1] == 2:
        z = np.ones((pts.shape[0], 1), dtype=pts.dtype)
        pts = np.concatenate([pts, z], axis=1)
    return pts

def main():
    ap = argparse.ArgumentParser("Overlay edge/spline on left RGB")
    ap.add_argument("--traj_dir", required=True, help="Trajectory root containing left/color and left/intrinsic")
    ap.add_argument("--pc_root", required=True, help="Pointcept data root (e.g., Pointcept/data/welding_deploy)")
    ap.add_argument("--split", default="test", help="Split folder name under pc_root (default: test)")
    ap.add_argument("--out_dir", required=True, help="Where to save overlay images")
    ap.add_argument("--draw_points", action="store_true", help="Draw visible_edge as points (in addition to a thin polyline)")
    args = ap.parse_args()

    traj_dir   = Path(args.traj_dir)
    color_dir  = traj_dir / "left" / "color"
    intrin_dir = traj_dir / "left" / "intrinsic"
    pc_split   = Path(args.pc_root) / args.split
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pc_split.exists():
        raise FileNotFoundError(f"Split not found: {pc_split}")

    sample_dirs = sorted([d for d in pc_split.iterdir() if d.is_dir()])
    if not sample_dirs:
        print(f"[WARN] no samples found under {pc_split}")
        return

    print(f"[INFO] Found {len(sample_dirs)} samples under {pc_split}")
    for d in sample_dirs:
        sample_id = d.name  # e.g., 000123
        try:
            img_path = _find_left_image(color_dir, sample_id)
            K = _load_K_for_frame(intrin_dir, sample_id)

            # Load vis edge (required)
            vis_edge_path = d / "visible_edge.npy"
            if not vis_edge_path.exists():
                print(f"[SKIP] {sample_id}: visible_edge.npy missing")
                continue
            visible_edge = np.load(vis_edge_path)  # (Nv,3)

            # Try spline (optional)
            spl_pts = None
            spl_t = d / "spl_t.npy"
            spl_c = d / "spl_c.npy"
            spl_k = d / "spl_k.npy"
            if spl_t.exists() and spl_c.exists() and spl_k.exists():
                try:
                    t = np.load(spl_t)
                    c = np.load(spl_c)
                    k = np.load(spl_k)
                    spl_pts = _eval_bspline_points(t, c, int(np.asarray(k).ravel()[0]), n_samples=800)
                except Exception as e:
                    print(f"[WARN] {sample_id}: spline eval failed: {repr(e)}")

            # Load image (BGR for cv2)
            img = np.array(Image.open(img_path).convert("RGB"))[:, :, ::-1].copy()
            H, W = img.shape[:2]

            # Project and draw visible_edge
            if visible_edge.ndim == 1 and visible_edge.size == 3:
                visible_edge = visible_edge[None, :]
            u, v, mask = _project_xyz_to_uv(visible_edge.reshape(-1, 3), K)
            uv = np.stack([u, v], axis=1)
            uv = uv[np.isfinite(uv).all(axis=1)]
            # thin green line + optional points
            _draw_polyline(img, uv, color_bgr=(0, 255, 0), thickness=1, closed=False)
            if args.draw_points:
                _draw_points(img, uv, color_bgr=(0, 200, 0), radius=1)

            # Project and draw spline (if any)
            if spl_pts is not None and spl_pts.shape[0] >= 2:
                u2, v2, m2 = _project_xyz_to_uv(spl_pts.reshape(-1, 3), K)
                uv2 = np.stack([u2, v2], axis=1)
                uv2 = uv2[np.isfinite(uv2).all(axis=1)]
                _draw_polyline(img, uv2, color_bgr=(255, 0, 255), thickness=2, closed=False)

            # Save
            out_path = out_dir / f"overlay_{sample_id}.png"
            imageio.imwrite(out_path.as_posix(), img[:, :, ::-1])  # back to RGB for saving
            print(f"[OK] {sample_id} -> {out_path}")

        except Exception as e:
            print(f"[ERR] {sample_id}: {repr(e)}")

if __name__ == "__main__":
    main()

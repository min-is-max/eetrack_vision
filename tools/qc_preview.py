#!/usr/bin/env python3
# tools/qc_preview.py
import argparse
from pathlib import Path

import numpy as np
import cv2
import imageio.v2 as imageio
from PIL import Image, ImageFont, ImageDraw

def list_frames_mixed(p):
    return sorted(list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg")))

def make_panel(color_bgr, depth_mm, plate_mask, pillar_mask, vmax_m=None):
    H, W = color_bgr.shape[:2]

    # depth → m → normalize → colormap
    depth_m = depth_mm.astype(np.float32) / 1000.0
    valid = depth_m > 0
    if vmax_m is None:
        vmax_m = np.percentile(depth_m[valid], 99) if np.any(valid) else 1.0
    norm = np.clip(depth_m / max(vmax_m, 1e-6), 0, 1)
    depth_vis = (norm * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

    # mask overlays
    overlay = color_bgr.copy()
    if plate_mask is not None:
        overlay[plate_mask > 0] = (0.6*overlay[plate_mask>0] + 0.4*np.array([0,255,0])).astype(np.uint8)  # green
    if pillar_mask is not None:
        overlay[pillar_mask > 0] = (0.6*overlay[pillar_mask>0] + 0.4*np.array([0,0,255])).astype(np.uint8)  # red→BGR: blue

    # boundary
    def contour_from_mask(m):
        if m is None: return []
        m8 = (m>0).astype(np.uint8)*255
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return cnts
    plate_cnt = contour_from_mask(plate_mask)
    pillar_cnt = contour_from_mask(pillar_mask)
    depth_with_edge = depth_vis.copy()
    cv2.drawContours(depth_with_edge, plate_cnt, -1, (0,255,0), 1)
    cv2.drawContours(depth_with_edge, pillar_cnt, -1, (0,0,255), 1)

    # 2x2 panel
    top = np.hstack([color_bgr, overlay])
    bottom = np.hstack([depth_vis, depth_with_edge])
    panel = np.vstack([top, bottom])
    return panel

def main():
    ap = argparse.ArgumentParser("Quick QC preview")
    ap.add_argument("data_dir", type=str, help="e.g., sample_data")
    ap.add_argument("--plate_name", type=str, default="horizontal plate")
    ap.add_argument("--pillar_name", type=str, default="vertical plate")
    ap.add_argument("--out_dir", type=str, default="preview")
    ap.add_argument("--max_frames", type=int, default=24)
    args = ap.parse_args()

    root = Path(args.data_dir)
    lcol = root / "left" / "color"
    rcol = root / "right" / "color"
    estd = root / "left" / "est_depth"
    pmask_dir = root / "left" / "mask" / args.plate_name.replace(" ","_")
    qmask_dir = root / "left" / "mask" / args.pillar_name.replace(" ","_")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    frames = list_frames_mixed(lcol)
    for i, fp in enumerate(frames[:args.max_frames]):
        idx = int(fp.stem) if fp.stem.isdigit() else i

        color = cv2.imread(str(fp))  # BGR
        if color is None: continue
        H,W = color.shape[:2]

        depth_path = estd / f"{idx:06d}.png"
        if not depth_path.exists(): continue
        depth_mm = imageio.imread(depth_path)
        # 읽은 게 16-bit인지 가볍게 체크
        if depth_mm.dtype != np.uint16:
            depth_mm = depth_mm.astype(np.uint16)

        ppath = pmask_dir / f"{idx:06d}.png"
        qpath = qmask_dir / f"{idx:06d}.png"
        plate = imageio.imread(ppath) if ppath.exists() else None
        pillar = imageio.imread(qpath) if qpath.exists() else None

        panel = make_panel(color, depth_mm, plate, pillar)
        cv2.imwrite(str(out / f"{idx:06d}_panel.png"), panel)

    print(f"[ok] saved panels to {out}")

if __name__ == "__main__":
    main()

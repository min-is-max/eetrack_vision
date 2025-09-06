#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deploy preprocessor for Pointcept Welding — GSAM2(DINO as-is) per-frame

- 좌/우 + K + baseline만으로 inference 전처리
- FoundationStereo: 좌/우 → est_depth(mm, uint16)
- GSAM2(DINO as-is → SAM2): 프레임별 plate/pillar 마스크 생성
- 경계(2D) → 3D 역투영 → B-spline 적합/샘플
- Pointcept 입력 NPY 세트 저장
- 엣지/깊이/세그 기준을 만족하지 못하는 프레임은 저장 스킵

입력 구조:
  {data_dir}/
    left/
      color/      000000.png|jpg, ...
      intrinsic/  K.txt (또는 000000.txt, ...)
    right/
      color/      000000.png|jpg, ...

출력 구조(기본 save_dir=../Pointcept/data/welding):
  {save_dir}/
    all/{frame_id}/...npy
    {split}/{frame_id}/...npy
"""

import os
import sys
import argparse
from pathlib import Path
import shutil
from typing import List, Optional, Tuple
import json
import re

import numpy as np
from PIL import Image
import imageio.v2 as imageio
import cv2
import open3d as o3d
import torch
from tqdm import tqdm
from scipy.interpolate import splprep, splev

# ---------------------------------------------------------
# import: GSAM2(DINO as-is) runner
#   /home/user/welding_line_detection/script/gsam2_from_dino_demo.py
# ---------------------------------------------------------
SCRIPT_DIR = (Path(__file__).parent / "../script").resolve()
sys.path.append(SCRIPT_DIR.as_posix())
try:
    gsam2_mod = __import__("gsam2_from_dino_demo")
    run_once = gsam2_mod.run_once
except Exception as e:
    raise ImportError(
        f"Failed to import run_once from {SCRIPT_DIR}/gsam2_from_dino_demo.py\n{repr(e)}"
    )

# ======================= utils =======================

def sanitize_name(name: str) -> str:
    return name.strip().replace(" ", "_")

def _canon_label(s: str) -> str:
    s = s.lower().strip()
    s = s.rstrip(".")
    s = re.sub(r"\([^)]*\)$", "", s).strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clear_and_make(p: Path):
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)

def segment2onehot(segment: np.ndarray, num_classes: int) -> np.ndarray:
    mask = segment != -1
    seg_onehot = np.zeros((len(segment), num_classes), dtype=np.float32)
    seg_onehot[mask, segment[mask]] = 1
    return seg_onehot

def read_intrinsic_paths(intrin_dir: Path, n_frames: int) -> List[Path]:
    if not intrin_dir.exists():
        raise FileNotFoundError(f"Intrinsic dir not found: {intrin_dir}")
    ks = sorted(intrin_dir.glob("*.txt"))
    if len(ks) == 0:
        raise FileNotFoundError(f"No intrinsic txt found in: {intrin_dir}")
    if len(ks) == 1:
        return [ks[0]] * n_frames
    if len(ks) != n_frames:
        print(f"[WARN] #intrinsics({len(ks)}) != #frames({n_frames}). Using min length.")
    return ks

def list_frames_mixed(left_col_dir: Path) -> List[Path]:
    return sorted(list(left_col_dir.glob("*.png")) + list(left_col_dir.glob("*.jpg")) + list(left_col_dir.glob("*.jpeg")))

# -------- geometry / spline --------
def point_cloud_from_rgbd(
    color, depth, intrinsic, extrinsic=np.eye(4),
    depth_scale=1000.0, depth_trunc=np.inf,
):
    h, w = color.shape[:2]
    color_o3d = o3d.geometry.Image(color)
    depth_o3d = o3d.geometry.Image(depth)

    # Camera intrinsics (adjust to your sensor)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=w,
        height=h,
        intrinsic_matrix=intrinsic,
    )
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_o3d,
        depth=depth_o3d,
        convert_rgb_to_intensity=False,
        depth_scale=depth_scale,  # if depth is in mm
        depth_trunc=depth_trunc,      # ignore depth > 5m
    )

    # Generate point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        image=rgbd,
        intrinsic=intrinsics,
        extrinsic=extrinsic,
    )

    return pcd

def backproject_uv_to_xyz(u: np.ndarray, v: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    Z = depth_m
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    return np.stack([X, Y, Z], axis=-1)

def bspline_points_down_sample(points: np.ndarray, target_num_pts: int, **kwargs) -> np.ndarray:
    if points.ndim == 2:
        tck, _ = splprep(points.T, **kwargs)
        u = np.linspace(0, 1, target_num_pts)
        return np.array(splev(u, tck)).T
    raise ValueError("points must be (N,2) or (N,3).")

def fit_spline(points: np.ndarray, **kwargs):
    if points.ndim != 2:
        raise ValueError("points must be 2D array.")
    tck, _ = splprep(points.T, **kwargs)
    t = tck[0]
    c = np.array(tck[1])
    k = np.array([tck[2]])
    return t, c, k

def extract_ordered_boundary(plate_mask: np.ndarray, pillar_mask: np.ndarray) -> np.ndarray:
    boundary = (plate_mask.astype(np.uint8) ^ pillar_mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cnts) == 0:
        return np.empty((0, 2), dtype=np.int32)
    cnt = max(cnts, key=lambda c: c.shape[0]).squeeze(1)  # (N,2), (x,y)
    ordered_uv = np.stack([cnt[:, 0], cnt[:, 1]], axis=1)  # (u,v)
    return ordered_uv

# -------- Open3D helper --------

def point_cloud_from_points(points: np.ndarray, colors: np.ndarray = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if colors is not None:
        if not isinstance(colors, np.ndarray):
            colors = np.asarray(colors)
        colors = colors.reshape(-1, 3)
        if len(colors) == 1:
            colors = colors.repeat(len(points), 0)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)
    return pcd

# ======================= depth =======================

def ensure_depth_for_traj(traj_dir: Path, baseline_m: float, force: bool = False):
    left_col_dir = traj_dir / "left" / "color"
    right_col_dir = traj_dir / "right" / "color"
    intrin_dir = traj_dir / "left" / "intrinsic"
    est_dir = traj_dir / "left" / "est_depth"

    left_images = list_frames_mixed(left_col_dir)
    right_images = list_frames_mixed(right_col_dir)
    n_frames = min(len(left_images), len(right_images))
    if n_frames == 0:
        return

    exist = sorted(est_dir.glob("*.png"))
    if (len(exist) == n_frames) and not force:
        return

    fs_path = (Path(__file__).parent / "../FoundationStereo").resolve()
    sys.path.append(fs_path.as_posix())
    from depth_estimator import DepthEstimator  # noqa: E402

    ensure_dir(est_dir)
    Ks = read_intrinsic_paths(intrin_dir, n_frames)

    torch.autograd.set_grad_enabled(False)
    depth_estimator = DepthEstimator()

    print(f"[FS] Running FoundationStereo on: {traj_dir}")
    for i in tqdm(range(n_frames), desc=f"FS {traj_dir.name}"):
        lcpath = left_images[i]
        rcpath = right_images[i]
        Kpath = Ks[i]
        left_image = np.asarray(Image.open(lcpath).convert("RGB"))
        right_image = np.asarray(Image.open(rcpath).convert("RGB"))
        K = np.loadtxt(Kpath).astype(np.float32)

        est_depth_m = depth_estimator.predict(left_image, right_image, K, baseline=baseline_m)  # meters
        est_depth_mm_u16 = np.clip(est_depth_m * 1000.0, 0, 65535).astype(np.uint16)
        imageio.imwrite((est_dir / f"{i:06d}.png").as_posix(), est_depth_mm_u16)

    del depth_estimator

# ======================= GSAM2 (from DINO) =======================

def _choose_class_masks_by_json(json_path: Path, plate_name: str, pillar_name: str) -> Tuple[Optional[int], Optional[int]]:
    if not json_path.exists():
        return None, None
    with open(json_path, "r") as f:
        data = json.load(f)
    anns = data.get("annotations", [])
    plate_canon = _canon_label(plate_name)
    pillar_canon = _canon_label(pillar_name)
    p_idx, q_idx = None, None
    for i, ann in enumerate(anns):
        cname = _canon_label(str(ann.get("class_name", "")))
        if p_idx is None and cname == plate_canon:
            p_idx = i
        elif q_idx is None and cname == pillar_canon:
            q_idx = i
    return p_idx, q_idx

def _save_mask_np_to_png(src_mask_path: Path, dst_png_path: Path):
    arr = imageio.imread(src_mask_path)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    mask = (arr > 0).astype(np.uint8) * 255
    imageio.imwrite(dst_png_path.as_posix(), mask)

def ensure_masks_for_traj_with_runner(
    traj_dir: Path,
    plate_name: str,
    pillar_name: str,
    *,
    gdino_cfg: str,
    gdino_ckpt: str,
    sam2_cfg: str,
    sam2_ckpt: str,
    device: str = "cuda",
    box_thresh: float = 0.35,
    text_thresh: float = 0.25,
    nms_iou: Optional[float] = None,
    per_class_topk: Optional[int] = None,
    force: bool = False,
):
    left_col_dir = traj_dir / "left" / "color"
    frames = list_frames_mixed(left_col_dir)
    n_frames = len(frames)
    if n_frames == 0:
        return

    mask_root = traj_dir / "left" / "mask"
    plate_dir = mask_root / sanitize_name(plate_name)
    pillar_dir = mask_root / sanitize_name(pillar_name)

    if plate_dir.exists() and pillar_dir.exists():
        p_png = sorted(plate_dir.glob("*.png"))
        q_png = sorted(pillar_dir.glob("*.png"))
        if len(p_png) == n_frames and len(q_png) == n_frames and not force:
            return

    ensure_dir(plate_dir)
    ensure_dir(pillar_dir)

    tmp_root = traj_dir / "left" / "mask_tmp"
    shutil.rmtree(tmp_root, ignore_errors=True)
    ensure_dir(tmp_root)

    print(f"[GSAM2:runner] Generating masks per-frame on: {traj_dir}")
    for idx, img_path in enumerate(tqdm(frames, desc=f"GSAM2 {traj_dir.name}")):
        out_dir = tmp_root / f"{idx:06d}"
        shutil.rmtree(out_dir, ignore_errors=True)
        ensure_dir(out_dir)

        text_prompt = f"{plate_name}. {pillar_name}."
        try:
            run_once(
                img_path=str(img_path),
                text_prompt=text_prompt,
                gdino_cfg=gdino_cfg,
                gdino_ckpt=gdino_ckpt,
                sam2_cfg=sam2_cfg,
                sam2_ckpt=sam2_ckpt,
                out_dir=out_dir,
                box_thresh=box_thresh,
                text_thresh=text_thresh,
                nms_iou=nms_iou,
                per_class_topk=per_class_topk,  # None이면 DINO 결과 그대로 사용
                device=device,
                dump_json=True,
            )
        except Exception as e:
            print(f"[GSAM2:runner][WARN] failed on frame {idx:06d}: {repr(e)}")
            continue

        jpath = out_dir / "grounded_sam2_results.json"
        p_i, q_i = _choose_class_masks_by_json(jpath, plate_name, pillar_name)

        if p_i is not None:
            src = out_dir / f"mask_{p_i:02d}.png"
            if src.exists():
                _save_mask_np_to_png(src, plate_dir / f"{idx:06d}.png")
        if q_i is not None:
            src = out_dir / f"mask_{q_i:02d}.png"
            if src.exists():
                _save_mask_np_to_png(src, pillar_dir / f"{idx:06d}.png")

    shutil.rmtree(tmp_root, ignore_errors=True)

# ======================= per-frame Pointcept =======================

def process_frame(
    traj_dir: Path,
    frame_idx: int,
    plate_name: str,
    pillar_name: str,
    save_dir_split: Path,
    save_dir_all: Path,
    dist_thresh_m: float = 0.003,
    *,
    min_uv_edge: int = 20,
    min_xyz_edge: int = 20,
    min_seg_points: int = 3000,
) -> bool:
    left_color_dir = traj_dir / "left" / "color"
    right_color_dir = traj_dir / "right" / "color"
    frames_l = list_frames_mixed(left_color_dir)
    frames_r = list_frames_mixed(right_color_dir)
    if frame_idx >= min(len(frames_l), len(frames_r)):
        return False

    lcpath = frames_l[frame_idx]
    rcpath = frames_r[frame_idx]
    estpath = (traj_dir / "left" / "est_depth" / f"{frame_idx:06d}.png")
    intrin_dir = traj_dir / "left" / "intrinsic"
    mask_p_dir = traj_dir / "left" / "mask" / sanitize_name(plate_name)
    mask_q_dir = traj_dir / "left" / "mask" / sanitize_name(pillar_name)
    plpath = mask_p_dir / f"{frame_idx:06d}.png"
    pipath = mask_q_dir / f"{frame_idx:06d}.png"

    # 필수 파일 체크
    if not (plpath.exists() and pipath.exists() and estpath.exists()):
        print(f"[SKIP] missing files at {traj_dir.name} idx={frame_idx:06d}")
        return False

    left_color = np.asarray(Image.open(lcpath).convert("RGB"))
    _ = np.asarray(Image.open(rcpath).convert("RGB"))  # right_color (현재 로직에서는 미사용)
    est_depth = imageio.imread(estpath)
    Ks = read_intrinsic_paths(intrin_dir, len(frames_l))
    K = np.loadtxt(Ks[frame_idx]).astype(np.float32)

    plate_mask = (imageio.imread(plpath) > 0)
    pillar_mask = (imageio.imread(pipath) > 0)
    H, W = est_depth.shape[:2]

    valid = est_depth > 0
    if valid.sum() < 1000:
        print(f"[SKIP] too few valid depth at {traj_dir.name} idx={frame_idx:06d} (valid={int(valid.sum())})")
        return False

    est_depth_m = est_depth.astype(np.float32) / 1000.0
    uu_full, vv_full = np.meshgrid(np.arange(W), np.arange(H))
    o3d.visualization.draw_geometries([point_cloud_from_rgbd(left_color, est_depth, K)])
    uu_v = uu_full[valid].astype(np.int32)
    vv_v = vv_full[valid].astype(np.int32)
    coord = backproject_uv_to_xyz(uu_v, vv_v, est_depth_m[vv_v, uu_v], K)
    color = left_color[vv_v, uu_v]

    # 객체 세그 레이블
    obj_segment_img = -1 * np.ones((H, W), dtype=np.int32)
    obj_segment_img[plate_mask] = 0
    obj_segment_img[pillar_mask] = 1
    obj_segment = obj_segment_img[valid]
    obj_segment_onehot = segment2onehot(obj_segment, 2)

    # 2D 경계 추출
    ordered_uv = extract_ordered_boundary(plate_mask, pillar_mask)
    if ordered_uv.shape[0] < min_uv_edge:
        print(f"[SKIP] too few 2D edge pixels ({ordered_uv.shape[0]} < {min_uv_edge}) at {traj_dir.name} idx={frame_idx:06d}")
        return False

    # 2D 에지 중 depth 유효한 포인트만 사용
    uu_e = ordered_uv[:, 0].astype(np.int32)
    vv_e = ordered_uv[:, 1].astype(np.int32)
    in_img = (uu_e >= 0) & (uu_e < W) & (vv_e >= 0) & (vv_e < H)
    depth_ok = np.zeros_like(in_img, dtype=bool)
    depth_ok[in_img] = est_depth_m[vv_e[in_img], uu_e[in_img]] > 0
    valid_edge = in_img & depth_ok
    uu_e = uu_e[valid_edge]
    vv_e = vv_e[valid_edge]
    if uu_e.size < min_uv_edge:
        print(f"[SKIP] too few valid-depth 2D edge ({uu_e.size} < {min_uv_edge}) at {traj_dir.name} idx={frame_idx:06d}")
        return False

    # 3D 에지 포인트
    edge_pts3d = backproject_uv_to_xyz(uu_e, vv_e, est_depth_m[vv_e, uu_e], K)
    if edge_pts3d.shape[0] < min_xyz_edge:
        print(f"[SKIP] too few 3D edge points ({edge_pts3d.shape[0]} < {min_xyz_edge}) at {traj_dir.name} idx={frame_idx:06d}")
        return False

    # 스플라인(충분할 때만)
    if edge_pts3d.shape[0] >= 8:
        try:
            spl_t, spl_c, spl_k = fit_spline(edge_pts3d, s=0.0, k=2)
            dense_visible_edge = bspline_points_down_sample(edge_pts3d, 500, s=0.0, k=2)
            edge_ds = bspline_points_down_sample(edge_pts3d, 64, s=0.0, k=2)
        except Exception as e:
            print(f"[SKIP] spline fit failed at {traj_dir.name} idx={frame_idx:06d}: {repr(e)}")
            return False
    else:
        # 최소 보존(그대로 사용)
        dense_visible_edge = edge_pts3d
        edge_ds = edge_pts3d
        spl_t = np.linspace(0.0, 1.0, max(2, edge_pts3d.shape[0]))
        spl_c = np.array(edge_pts3d).T
        spl_k = np.array([1])

    # 거리 기반 라벨 생성
    try:
        pcd = point_cloud_from_points(coord)
        dense_edge_pcd = point_cloud_from_points(dense_visible_edge)
        dist = np.asarray(pcd.compute_point_cloud_distance(dense_edge_pcd))
    except Exception as e:
        print(f"[SKIP] distance compute failed at {traj_dir.name} idx={frame_idx:06d}: {repr(e)}")
        return False

    edge_segment = (dist < dist_thresh_m).astype(np.int32)
    if int(edge_segment.sum()) < min_seg_points:
        print(f"[SKIP] too few seg=1 points ({int(edge_segment.sum())} < {min_seg_points}) at {traj_dir.name} idx={frame_idx:06d}")
        return False

    # ---- 저장 ----
    sample_id = f"{frame_idx:06d}"
    out_dir = save_dir_split / sample_id
    ensure_dir(out_dir)
    out_all = save_dir_all / sample_id
    ensure_dir(out_all)

    np.save(out_dir / "color.npy", color)
    np.save(out_dir / "coord.npy", coord)
    np.save(out_dir / "obj_segment.npy", obj_segment)
    np.save(out_dir / "obj_segment_onehot.npy", obj_segment_onehot)
    np.save(out_dir / "edge.npy", edge_pts3d)
    np.save(out_dir / "edge_ds.npy", edge_ds)
    np.save(out_dir / "visible_edge.npy", dense_visible_edge)
    np.save(out_dir / "spl_t.npy", spl_t)
    np.save(out_dir / "spl_c.npy", spl_c)
    np.save(out_dir / "spl_k.npy", spl_k)
    np.save(out_dir / "segment.npy", edge_segment)

    for fname in [
        "color.npy", "coord.npy", "obj_segment.npy", "obj_segment_onehot.npy",
        "edge.npy", "edge_ds.npy", "visible_edge.npy", "spl_t.npy", "spl_c.npy", "spl_k.npy",
        "segment.npy",
    ]:
        shutil.copy2(out_dir / fname, out_all / fname)

    return True

# ======================= main =======================

def main():
    parser = argparse.ArgumentParser(description="Preprocess (deploy) for Pointcept Welding — GSAM2(DINO) per-frame")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--plate_name", type=str, default="bottom_plate")
    parser.add_argument("--pillar_name", type=str, default="pillar")
    parser.add_argument("--baseline", type=float, default=0.063)

    parser.add_argument("--save_dir", type=str,
                        default=(Path(__file__).parent / "../Pointcept/data/welding").resolve().as_posix())
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    parser.add_argument("--force_depth", action="store_true")

    # GSAM2(DINO as-is) config
    parser.add_argument("--gdino_cfg", type=str,
                        default="Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--gdino_ckpt", type=str,
                        default="Grounded-SAM-2/gdino_checkpoints/grounded_dino_finetuned.pth")
    parser.add_argument("--sam2_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2_ckpt", type=str, default="Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--gsam2_device", type=str, default="cuda", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--box_thresh", type=float, default=0.35)
    parser.add_argument("--text_thresh", type=float, default=0.25)
    parser.add_argument("--nms_iou", type=float, default=None)
    parser.add_argument("--per_class_topk", type=int, default=1, help="클래스별 최대 보존 박스 수 (기본 1 추천)")

    # 라벨 거리 임계
    parser.add_argument("--dist_thresh_mm", type=float, default=3.0)

    # 프레임 스킵 기준
    parser.add_argument("--min_uv_edge", type=int, default=20,
                        help="2D 경계 픽셀 최소 개수(visible depth 있는 에지 픽셀)")
    parser.add_argument("--min_xyz_edge", type=int, default=20,
                        help="3D로 역투영된 에지 포인트 최소 개수")
    parser.add_argument("--min_seg_points", type=int, default=200,
                        help="Pointcept label=1로 마킹될 포인트 최소 개수(거리 기반 세그)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    save_root = Path(args.save_dir).resolve()
    split = args.split

    clear_and_make(save_root)
    save_dir_all = save_root / "all"
    ensure_dir(save_dir_all)
    save_dir_split = save_root / split
    ensure_dir(save_dir_split)

    traj_left_color_dirs = sorted(data_dir.rglob("left/color"))
    if len(traj_left_color_dirs) == 0 and (data_dir / "left" / "color").exists():
        traj_left_color_dirs = [data_dir / "left" / "color"]
    traj_dirs = sorted({d.parents[1] for d in traj_left_color_dirs})
    if len(traj_dirs) == 0:
        raise FileNotFoundError(f"No trajectories with left/color found under {data_dir}")

    # 1) depth
    for traj in traj_dirs:
        ensure_depth_for_traj(traj, baseline_m=args.baseline, force=args.force_depth)

    # 2) masks via GSAM2(DINO)
    for traj in traj_dirs:
        ensure_masks_for_traj_with_runner(
            traj,
            plate_name=args.plate_name,
            pillar_name=args.pillar_name,
            gdino_cfg=args.gdino_cfg,
            gdino_ckpt=args.gdino_ckpt,
            sam2_cfg=args.sam2_cfg,
            sam2_ckpt=args.sam2_ckpt,
            device=args.gsam2_device,
            box_thresh=args.box_thresh,
            text_thresh=args.text_thresh,
            nms_iou=args.nms_iou,
            per_class_topk=args.per_class_topk,
            force=True,  # 새 파이프라인 검증 중이라 강제 재생성 권장
        )

    # 3) export to Pointcept format
    total_ok = 0
    total_frames = 0
    print("[PROC] Generating Pointcept inputs...")
    for traj in traj_dirs:
        left_images = list_frames_mixed(traj / "left" / "color")
        n_frames = len(left_images)
        total_frames += n_frames
        for i in tqdm(range(n_frames), desc=f"PROC {traj.name}"):
            ok = process_frame(
                traj_dir=traj,
                frame_idx=i,
                plate_name=args.plate_name,
                pillar_name=args.pillar_name,
                save_dir_split=save_dir_split,
                save_dir_all=save_dir_all,
                dist_thresh_m=args.dist_thresh_mm / 1000.0,
                min_uv_edge=args.min_uv_edge,
                min_xyz_edge=args.min_xyz_edge,
                min_seg_points=args.min_seg_points,
            )
            if ok:
                total_ok += 1

    print(f"[DONE] Saved to: {save_root.as_posix()}  (ok: {total_ok}/{total_frames})")

if __name__ == "__main__":
    main()

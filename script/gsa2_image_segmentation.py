import os
import sys
from pathlib import Path
import argparse
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

# Add Grounded-SAM-2 path manually
gsa2_path = os.path.join(os.path.dirname(__file__), "../Grounded-SAM-2")
sys.path.append(gsa2_path)

from gsa2_video_tracker import GSA2VideoTracker


def compute_mask_iou(mask_gt, mask_pred):
    intersection = np.logical_and(mask_gt, mask_pred)
    union = np.logical_or(mask_gt, mask_pred)
    return (np.sum(intersection) + 1e-6) / (np.sum(union) + 1e-6)


def list_frames(img_dir):
    """jpg/jpeg/png 전부 수집, 숫자 스템이면 숫자 기준 정렬, 아니면 이름 기준"""
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths = []
    for pat in patterns:
        paths.extend(Path(img_dir).glob(pat))
    def key_fn(p):
        s = p.stem
        return (0, int(s)) if s.isdigit() else (1, str(s))
    return sorted(paths, key=key_fn)


def safe_join(base, child):
    return os.path.join(base, child) if base else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image segmentation with Grounded-SAM2")
    parser.add_argument("image_dir", type=str, help="Directory to image frames.")
    parser.add_argument("--plate_prompt",  type=str, default="bottom metal plate", help="Text prompt for plate")
    parser.add_argument("--pillar_prompt", type=str, default="upper metal plate",  help="Text prompt for pillar")
    parser.add_argument("--mask_dir",      type=str, default=None, help="Directory root to save masks")
    parser.add_argument("--result_dir",    type=str, default=None, help="Directory root to save results")
    parser.add_argument("--plate_mask_gt_dir",  type=str, default=None, help="Directory to GT plate masks")
    parser.add_argument("--pillar_mask_gt_dir", type=str, default=None, help="Directory to GT pillar masks")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to load the model.")
    args = parser.parse_args()

    if not args.plate_prompt.endswith('.'):
        args.plate_prompt += '.'
    if not args.pillar_prompt.endswith('.'):
        args.pillar_prompt += '.'

    text_prompt = args.plate_prompt + ' ' + args.pillar_prompt
    video_tracker = GSA2VideoTracker(args.device)

    img_paths = list_frames(args.image_dir)
    num_frames = len(img_paths)

    # GT를 스템으로 매핑해 동일 이름만 비교
    def gt_map(gt_dir):
        if not gt_dir:
            return {}
        d = {}
        for p in Path(gt_dir).glob("*.png"):
            d[p.stem] = p
        return d

    plate_gt = gt_map(args.plate_mask_gt_dir)
    pillar_gt = gt_map(args.pillar_mask_gt_dir)

    plate_mask_ious, pillar_mask_ious = [], []
    plate_detect_success = pillar_detect_success = 0
    plate_detect_fail_ids, pillar_detect_fail_ids = [], []

    for i, img_path in tqdm(list(enumerate(img_paths)), total=num_frames):
        stem = img_path.stem

        mask_dir_i   = safe_join(args.mask_dir,   str(i))
        result_dir_i = safe_join(args.result_dir, str(i))

        masks, class_names, confidences = video_tracker.predict_image(
            img_path,
            text_prompt,
            mask_dir=mask_dir_i,
            result_dir=result_dir_i,
        )

        # plate
        plate_gt_path = plate_gt.get(stem)
        if plate_gt_path is not None and (args.plate_prompt[:-1] in class_names):
            plate_mask_gt = imageio.imread(plate_gt_path)
            ids = [j for j, cn in enumerate(class_names) if cn == args.plate_prompt[:-1]]
            top1_idx = max(ids, key=lambda j: confidences[j])  # confidences가 list일 수 있음
            plate_mask_pred = masks[top1_idx]
            plate_mask_ious.append(compute_mask_iou(plate_mask_gt, plate_mask_pred))
            plate_detect_success += 1
        else:
            plate_detect_fail_ids.append(i)

        # pillar
        pillar_gt_path = pillar_gt.get(stem)
        if pillar_gt_path is not None and (args.pillar_prompt[:-1] in class_names):
            pillar_mask_gt = imageio.imread(pillar_gt_path)
            ids = [j for j, cn in enumerate(class_names) if cn == args.pillar_prompt[:-1]]
            top1_idx = max(ids, key=lambda j: confidences[j])
            pillar_mask_pred = masks[top1_idx]
            pillar_mask_ious.append(compute_mask_iou(pillar_mask_gt, pillar_mask_pred))
            pillar_detect_success += 1
        else:
            pillar_detect_fail_ids.append(i)

    # 출력 요약 (GT가 있을 때만 IoU 통계 출력)
    if args.plate_mask_gt_dir:
        denom = max(1, len(plate_gt))
        print("-----PLATE detection success rate: ", plate_detect_success / denom, "-----")
        print("Detection fail ids: ", plate_detect_fail_ids)
        print("Mean IoU: ", float(np.mean(plate_mask_ious)) if plate_mask_ious else float('nan'))
        print("Std IoU: ", float(np.std(plate_mask_ious))  if plate_mask_ious else float('nan'))

    if args.pillar_mask_gt_dir:
        denom = max(1, len(pillar_gt))
        print("-----PILLAR detection success rate: ", pillar_detect_success / denom, "-----")
        print("Detection fail ids: ", pillar_detect_fail_ids)
        print("Mean IoU: ", float(np.mean(pillar_mask_ious)) if pillar_mask_ious else float('nan'))
        print("Std IoU: ", float(np.std(pillar_mask_ious))  if pillar_mask_ious else float('nan'))

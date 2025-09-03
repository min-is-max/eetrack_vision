#!/usr/bin/env python3
# script/gsam2_from_dino_demo.py
# - Grounding DINO: "네 미니멀 데모" 코드 그대로 사용 (checkpoint 래핑 X)
# - SAM2: DINO가 낸 박스를 받아 마스크 생성/시각화/JSON 저장

import argparse
import os
import re
import sys
import json
from pathlib import Path
from contextlib import nullcontext
from flask import ctx
import numpy as np
import torch
import cv2
import supervision as sv
import pycocotools.mask as mask_util
from torchvision.ops import box_convert, nms
from PIL import Image, ImageDraw, ImageFont

# ---------- GroundingDINO 동적 임포트 (editable 설치 없이도 동작) ----------
from importlib import util as importutil
GDINO_INIT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'Grounded-SAM-2', 'grounding_dino', 'groundingdino', '__init__.py'
))
spec = importutil.spec_from_file_location("groundingdino", GDINO_INIT)
module = importutil.module_from_spec(spec)
sys.modules["groundingdino"] = module
spec.loader.exec_module(module)
# -------------------------------------------------------------------------

# === Grounding DINO 미니멀 데모에서 쓰는 것들 그대로 ===
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

# === SAM2 ===
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ---------------- Grounding DINO: 네 데모와 동일한 유틸 ----------------
def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model_gdino(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if (not cpu_only and torch.cuda.is_available()) else "cpu"
    model = build_model(args)

    ckpt = torch.load(model_checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt  # raw state_dict 가정
    load_res = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    print(load_res)
    model.eval()
    return model

def _normalize_caption(caption: str) -> str:
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    return caption

def get_grounding_output(model,
                         image,
                         caption,
                         box_threshold,
                         text_threshold=None,
                         with_logits=True,
                         cpu_only=False,
                         token_spans=None):
    """
    Returns:
      boxes_filt: Tensor[M, 4]  (cx, cy, w, h) normalized
      pred_phrases: list[str]
      scores_max: Tensor[M]
    """
    assert (text_threshold is not None) or (token_spans is not None), \
        "text_threshold and token_spans should not be None at the same time!"

    caption = _normalize_caption(caption)
    device = "cuda" if (not cpu_only and torch.cuda.is_available()) else "cpu"

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes  = outputs["pred_boxes"][0]             # (nq, 4) cxcywh

    if token_spans is None:
        logits_filt = logits.detach().cpu().clone()
        boxes_filt  = boxes.detach().cpu().clone()
        filt_mask   = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt  = boxes_filt[filt_mask]

        scores_max = logits_filt.max(dim=1)[0].cpu()

        tokenizer  = model.tokenizer
        tokenized  = tokenizer(caption)
        pred_phrases = []
        for logit in logits_filt:
            phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            pred_phrases.append(phrase + (f"({str(logit.max().item())[:4]})" if with_logits else ""))
        return boxes_filt, pred_phrases, scores_max
    else:
        tokenizer = model.tokenizer
        positive_maps = create_positive_map_from_span(
            tokenizer(caption), token_span=token_spans
        ).to(image.device)  # n_phrase, 256

        logits_for_phrases = (positive_maps @ logits.T).cpu()
        all_boxes, all_labels, all_scores = [], [], []
        for span_scores in logits_for_phrases:
            keep = span_scores > box_threshold
            if not torch.any(keep):
                continue
            sel_scores = span_scores[keep]
            sel_boxes  = boxes.detach().cpu()[keep]
            all_boxes.append(sel_boxes)
            all_scores.append(sel_scores)
            all_labels.extend([f"phrase({float(s):.2f})" if with_logits else "phrase"
                               for s in sel_scores])
        if len(all_boxes) == 0:
            return torch.empty((0,4)), [], torch.empty((0,))
        boxes_filt = torch.cat(all_boxes, dim=0)
        scores_max = torch.cat(all_scores, dim=0)
        return boxes_filt, all_labels, scores_max
# -----------------------------------------------------------------------


def _canon(s: str) -> str:
    s = s.lower().strip()
    s = s.rstrip(".")
    s = re.sub(r"\([^)]*\)$", "", s).strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _single_mask_to_rle(mask: np.ndarray) -> dict:
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def run_once(
    img_path: str,
    text_prompt: str,
    gdino_cfg: str,
    gdino_ckpt: str,
    sam2_cfg: str,
    sam2_ckpt: str,
    out_dir: Path,
    box_thresh: float = 0.35,
    text_thresh: float = 0.25,
    nms_iou: float | None = None,
    per_class_topk: int | None = None,   # None이면 DINO 결과를 그대로 사용
    device: str = "auto",
    dump_json: bool = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    # --- Grounding DINO (as-is) ---
    model = load_model_gdino(gdino_cfg, gdino_ckpt, cpu_only=(device=="cpu"))
    image_pil, image = load_image(img_path)
    H, W = image_pil.size[1], image_pil.size[0]  # (H,W)

    boxes_cxcywh, phrases, scores = get_grounding_output(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
        with_logits=True,
        cpu_only=(device=="cpu"),
        token_spans=None
    )

    if len(boxes_cxcywh) == 0:
        print(f"[warn] no DINO boxes for {img_path}")
        return

    # NMS (옵션, DINO 결과 위에서)
    boxes_xyxy_pix = box_convert(
        boxes=boxes_cxcywh * torch.tensor([W, H, W, H]), in_fmt="cxcywh", out_fmt="xyxy"
    )
    conf = scores.clone()
    labs = phrases[:]

    if nms_iou is not None and len(boxes_xyxy_pix) > 1:
        keep = nms(boxes_xyxy_pix, conf, iou_threshold=nms_iou)
        boxes_xyxy_pix = boxes_xyxy_pix[keep]
        conf = conf[keep]
        labs = [labs[i] for i in keep.tolist()]

    # 클래스별 Top-K (원하면 사용)
    if per_class_topk is not None and per_class_topk > 0:
        wanted = [_canon(c) for c in _normalize_caption(text_prompt).split(".") if c.strip()]
        canon_labs = [_canon(x) for x in labs]
        keep_idxs = []
        conf_np = conf.detach().cpu().numpy()
        for cls in wanted:
            idxs = [i for i, l in enumerate(canon_labs) if l == cls]
            if not idxs: 
                continue
            idxs_sorted = sorted(idxs, key=lambda i: conf_np[i], reverse=True)
            keep_idxs.extend(idxs_sorted[:per_class_topk])
        if keep_idxs:
            keep_idxs = sorted(set(keep_idxs))
            boxes_xyxy_pix = boxes_xyxy_pix[keep_idxs]
            conf = conf[keep_idxs]
            labs = [labs[i] for i in keep_idxs]

    # --- SAM2 (DINO 박스를 그대로 input) ---
    sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=device)  # 파일 경로 권장
    predictor = SAM2ImagePredictor(sam2_model)

    # SAM2는 원본 RGB np.uint8(H,W,3) 입력 필요
    img_rgb = np.array(image_pil)  # PIL -> np
    predictor.set_image(img_rgb)

    boxes_np = boxes_xyxy_pix.detach().cpu().numpy()

    # 안정성/성능 옵션
    ctx = torch.amp.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
    with ctx:
        masks, sam_scores, logits = predictor.predict(
            point_coords=None, point_labels=None, box=boxes_np, multimask_output=False,
        )
    if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    masks, sam_scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes_np,
        multimask_output=False,
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)  # (n,H,W)

    # --- 시각화/저장 ---
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    dets = sv.Detections(
        xyxy=boxes_np,
        mask=masks.astype(bool),
        class_id=np.arange(len(labs))
    )

    vis = sv.BoxAnnotator().annotate(scene=img_bgr.copy(), detections=dets)
    lab_txt = [f"{l} {float(c):.2f}" for l, c in zip(labs, conf.detach().cpu().numpy().tolist())]
    vis = sv.LabelAnnotator().annotate(scene=vis, detections=dets, labels=lab_txt)
    cv2.imwrite(str(out_dir / "groundingdino_annotated_image.jpg"), vis)

    vis2 = sv.MaskAnnotator().annotate(scene=vis, detections=dets)
    cv2.imwrite(str(out_dir / "grounded_sam2_annotated_image_with_mask.jpg"), vis2)

    # 마스크 개별 저장
    for i, m in enumerate(masks.astype(np.uint8) * 255):
        cv2.imwrite(str(out_dir / f"mask_{i:02d}.png"), m)

    # JSON 저장
    if dump_json:
        rles = [_single_mask_to_rle(m) for m in masks]
        results = {
            "image_path": img_path,
            "annotations": [
                {
                    "class_name": labs[i],
                    "bbox": boxes_np[i].tolist(),     # xyxy
                    "segmentation": rles[i],
                    "score": float(sam_scores[i]) if hasattr(sam_scores, "__len__") else float(sam_scores),
                }
                for i in range(len(labs))
            ],
            "box_format": "xyxy",
            "img_width": int(W),
            "img_height": int(H),
        }
        with open(out_dir / "grounded_sam2_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"[ok] saved -> {out_dir}")


def parse_args():
    p = argparse.ArgumentParser("GSAM2 (DINO as-is → SAM2 on top)")
    p.add_argument("--img_path", type=str, required=True)
    p.add_argument("--text_prompt", type=str, default="vertical plate. horizontal plate.")
    # DINO
    p.add_argument("--gdino_cfg", type=str, required=True,
                   help="Grounding DINO config (.py) — ex) GroundingDINO_SwinT_OGC.py")
    p.add_argument("--gdino_ckpt", type=str, required=True,
                   help="Grounding DINO checkpoint (.pth) — as-is (no wrap)")
    # SAM2 (파일 경로를 권장)
    p.add_argument("--sam2_cfg", type=str, required=True,
                   help="SAM2 yaml config file path ex) Grounded-SAM-2/configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2_ckpt", type=str, required=True,
                   help="SAM2 checkpoint path ex) Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    # 옵션
    p.add_argument("--box_thresh", type=float, default=0.35)
    p.add_argument("--text_thresh", type=float, default=0.25)
    p.add_argument("--nms_iou", type=float, default=None)
    p.add_argument("--per_class_topk", type=int, default=None, help="None이면 DINO 결과 그대로 사용")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--out_dir", type=str, default="outputs/gsam2_as_is")
    p.add_argument("--dump_json", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_once(
        img_path=args.img_path,
        text_prompt=args.text_prompt,
        gdino_cfg=args.gdino_cfg,
        gdino_ckpt=args.gdino_ckpt,
        sam2_cfg=args.sam2_cfg,
        sam2_ckpt=args.sam2_ckpt,
        out_dir=Path(args.out_dir),
        box_thresh=args.box_thresh,
        text_thresh=args.text_thresh,
        nms_iou=args.nms_iou,
        per_class_topk=args.per_class_topk,   # 기본 None → DINO 결과 건드리지 않음
        device=args.device,
        dump_json=args.dump_json,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Grounding DINO minimal demo (class-wise top-1 visualization)
import argparse
import os
import re
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ---------- Dynamic module import (so editable install isn't strictly required) ----------
from importlib import util as importutil
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           '..', 'Grounded-SAM-2', 'grounding_dino',
                                           'groundingdino', '__init__.py'))
spec = importutil.spec_from_file_location("groundingdino", module_path)
module = importutil.module_from_spec(spec)
sys.modules["groundingdino"] = module
spec.loader.exec_module(module)
# ----------------------------------------------------------------------------------------

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


def plot_boxes_to_image(image_pil, tgt):
    """
    Draw boxes (cx,cy,w,h normalized) + labels on an image.
    tgt: {"size":[H,W], "boxes": Tensor[M,4], "labels":[str]*M}
    """
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    for box, label in zip(boxes, labels):
        # normalized cxcywh -> scale to pixels
        box = box * torch.tensor([W, H, W, H], dtype=box.dtype)
        # cxcywh -> xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]

        color = tuple(np.random.randint(0, 255, size=3).tolist())
        x0, y0, x1, y1 = [int(v) for v in box.tolist()]

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        font = ImageFont.load_default()
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font=font)
        else:
            w, h = draw.textsize(str(label), font=font)
            bbox = (x0, y0, x0 + w, y0 + h)
        draw.rectangle(bbox, fill=color)
        draw.text((bbox[0], bbox[1]), str(label), fill="white", font=font)

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


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


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if (not cpu_only and torch.cuda.is_available()) else "cpu"
    model = build_model(args)

    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # assume raw state_dict
    load_res = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    print(load_res)
    model.eval()
    return model


def _normalize_caption(caption: str) -> str:
    """
    Make caption friendly for tokenizer:
    - lowercase, strip, ensure trailing dot
    """
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
      pred_phrases: list[str]    labels (optionally with "(score)")
      scores_max: Tensor[M]      per-box max token score (after filtering)
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
    boxes = outputs["pred_boxes"][0]              # (nq, 4) cxcywh in [0,1]

    if token_spans is None:
        logits_filt = logits.detach().cpu().clone()
        boxes_filt = boxes.detach().cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]      # (M, 256)
        boxes_filt = boxes_filt[filt_mask]        # (M, 4)

        # per-box score = max token score
        scores_max = logits_filt.max(dim=1)[0].cpu()

        # phrases
        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)
        pred_phrases = []
        for logit in logits_filt:
            phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            pred_phrases.append(phrase + (f"({str(logit.max().item())[:4]})" if with_logits else ""))
        return boxes_filt, pred_phrases, scores_max
    else:
        # Given-phrase mode (token_spans)
        tokenizer = model.tokenizer
        positive_maps = create_positive_map_from_span(
            tokenizer(caption),
            token_span=token_spans
        ).to(image.device)  # n_phrase, 256

        # phrase-wise scores over queries
        logits_for_phrases = (positive_maps @ logits.T).cpu()  # (n_phrase, nq)

        all_boxes = []
        all_labels = []
        all_scores = []
        for span_scores in logits_for_phrases:
            # mask queries above threshold
            keep = span_scores > box_threshold
            if not torch.any(keep):
                continue
            sel_scores = span_scores[keep]
            sel_boxes = boxes.detach().cpu()[keep]

            all_boxes.append(sel_boxes)
            all_scores.append(sel_scores)

            # reconstruct phrase text for this span:
            # NOTE: token_spans는 원문 문자열 인덱스를 받는 포맷을 가정
            # 여기서는 라벨을 단순히 "phrase(i)" 형식으로 두거나,
            # 필요시 실제 문자열 조합 로직을 추가할 수 있음.
            all_labels.extend([f"phrase({float(s):.2f})" if with_logits else "phrase"
                               for s in sel_scores])

        if len(all_boxes) == 0:
            return torch.empty((0, 4)), [], torch.empty((0,))

        boxes_filt = torch.cat(all_boxes, dim=0)
        scores_max = torch.cat(all_scores, dim=0)
        return boxes_filt, all_labels, scores_max


def _canon_phrase(s: str) -> str:
    """
    Canonicalize a predicted phrase or prompt:
    - lowercase, remove trailing '.', remove trailing '(score)'
    - collapse non-alnum to single space
    """
    s = s.lower().strip()
    s = s.rstrip('.')
    s = re.sub(r'\([^)]*\)$', '', s).strip()      # drop "(0.87)"
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def main():
    parser = argparse.ArgumentParser("Grounding DINO example (class-wise top-1)", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt (use natural language, end with '.')")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None,
                        help="Python-list-like string for spans (use only in trusted environment).")
    parser.add_argument("--cpu-only", action="store_true", help="run on CPU only")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load image
    image_pil, image = load_image(args.image_path)
    image_pil.save(os.path.join(args.output_dir, "raw_image.jpg"))

    # 2) Load model
    model = load_model(args.config_file, args.checkpoint_path, cpu_only=args.cpu_only)

    # 3) Run model
    # token_spans 안전성 주의: 신뢰된 입력에서만 사용 권장
    spans = None
    if args.token_spans is not None:
        spans = eval(f"{args.token_spans}")
        # token_spans 모드에서는 text_threshold를 None으로 둔다.
        text_threshold = None
    else:
        text_threshold = args.text_threshold

    boxes_filt, pred_phrases, scores_max = get_grounding_output(
        model=model,
        image=image,
        caption=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=text_threshold,
        with_logits=True,
        cpu_only=args.cpu_only,
        token_spans=spans
    )

    # 4) Class-wise Top-1 reduction
    # 기대 클래스 목록은 프롬프트에서 '.' 구분자로 분리해 캐논화
    expected_classes = [_canon_phrase(c) for c in args.text_prompt.split('.') if c.strip()]
    canon_labels = [_canon_phrase(lbl) for lbl in pred_phrases]

    keep_boxes = []
    keep_labels = []

    if len(boxes_filt) > 0:
        scores_np = scores_max.detach().cpu().numpy()
        boxes_np = boxes_filt.detach().cpu().numpy()

        for cls in expected_classes:
            idxs = [i for i, lab in enumerate(canon_labels) if lab == cls]
            if not idxs:
                continue
            top_i = max(idxs, key=lambda i: scores_np[i])
            keep_boxes.append(boxes_np[top_i])
            keep_labels.append(pred_phrases[top_i])

    # 안전장치: 하나도 못 골랐으면 원본 박스/라벨로 fallback
    if keep_boxes:
        boxes_show = torch.from_numpy(np.stack(keep_boxes, axis=0))
        labels_show = keep_labels
    else:
        boxes_show = boxes_filt
        labels_show = pred_phrases

    # 5) Visualize & save
    size = image_pil.size  # (W, H)
    pred_dict = {
        "boxes": boxes_show,              # normalized cxcywh
        "size": [size[1], size[0]],       # H, W
        "labels": labels_show,
    }
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    image_with_box.save(os.path.join(args.output_dir, "pred.jpg"))


if __name__ == "__main__":
    main()

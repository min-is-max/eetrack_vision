#!/bin/bash
set -e

# ---------------- 사용자 입력 ----------------
DATA_DIR="./input"   # 좌/우 이미지 + intrinsic 있는 폴더
PLATE_NAME="horizontal plate"
PILLAR_NAME="vertical plate"
BASELINE=0.12

GDINO_CFG="Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CKPT="Grounded-SAM-2/gdino_checkpoints/grounded_dino_finetuned.pth"
SAM2_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT="Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"

SAVE_ROOT_PC="./Pointcept/data/welding_deploy_2"
EXP_DIR="./Pointcept/exp/welding/seg-pt-v3m1-0-base-deploy"
CONFIG_FILE="./configs/welding/seg-pt-v3m1-0-base.py" 
WEIGHT="./exp/welding/seg-pt-v3m1-0-base-specimen-default-16trajs/model/model_last.pth"

# ---------------- Stage 1: Preprocess ----------------
echo "[Stage 1/3] Preprocess → ${SAVE_ROOT_PC}"
python data_gen/preprocess_poincept_deploy.py \
  "${DATA_DIR}" \
  --plate_name "${PLATE_NAME}" \
  --pillar_name "${PILLAR_NAME}" \
  --baseline "${BASELINE}" \
  --save_dir "${SAVE_ROOT_PC}" \
  --split test \
  --gdino_cfg "${GDINO_CFG}" \
  --gdino_ckpt "${GDINO_CKPT}" \
  --sam2_cfg "${SAM2_CFG}" \
  --sam2_ckpt "${SAM2_CKPT}" \
  --gsam2_device cuda \
  --box_thresh 0.35 \
  --text_thresh 0.25 \
  --nms_iou 0.5 \
  --per_class_topk 1

# ---------------- Stage 2: Pointcept Test ----------------
echo "[Stage 2/3] Run Pointcept Test → ${EXP_DIR}"
cd Pointcept
PYTHONPATH=./ \
python tools/test.py \
  --config-file "${CONFIG_FILE}" \
  --num-gpus 1 \
  --options \
    data_root="${SAVE_ROOT_PC}" \
    save_path="${EXP_DIR}" \
    weight="${WEIGHT}"
cd ..



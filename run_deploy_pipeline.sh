#!/bin/bash
set -euo pipefail
# ============== 사용자 입력(고정값/모델 체크포인트 등) ==============
PLATE_NAME="horizontal plate"
PILLAR_NAME="vertical plate"
BASELINE=0.12
GDINO_CFG="Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CKPT="Grounded-SAM-2/gdino_checkpoints/grounded_dino_finetuned.pth"
SAM2_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT="Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
CONFIG_FILE="./Pointcept/configs/welding/seg-pt-v3m1-0-base.py"
WEIGHT="./Pointcept/exp/welding/seg-pt-v3m1-0-base-specimen-default-16trajs/model/model_last.pth"
# ============== 실행별 입력(매번 달라질 수 있는 값) ==============
DATA_DIR="./input"
# ============== 실행 ID/경로(매 실행마다 새 디렉토리) ==============
RUN_ID="$(date +%Y%m%d_%H%M%S)-$RANDOM"
SAVE_ROOT_PC="./Pointcept/data/welding_deploy_${RUN_ID}"
EXP_DIR="./Pointcept/exp/welding/run_${RUN_ID}"
# 심볼릭 링크 'latest'로 가장 최근 실행을 가리키게 하기(편의용)
LATEST_DATA_LINK="./Pointcept/data/latest_deploy"
LATEST_EXP_LINK="./Pointcept/exp/welding/latest_run"
# ============== 절대경로 변환(경로 꼬임 방지) ==============
abs() { python3 - "$1" <<'PY'
import os,sys; print(os.path.realpath(sys.argv[1]))
PY
}
DATA_DIR_ABS="$(abs "$DATA_DIR")"
SAVE_ROOT_PC_ABS="$(abs "$SAVE_ROOT_PC")"
EXP_DIR_ABS="$(abs "$EXP_DIR")"
CONFIG_FILE_ABS="$(abs "$CONFIG_FILE")"
WEIGHT_ABS="$(abs "$WEIGHT")"
GDINO_CFG_ABS="$(abs "$GDINO_CFG")"
GDINO_CKPT_ABS="$(abs "$GDINO_CKPT")"
SAM2_CFG_ABS="$(abs "$SAM2_CFG")"
SAM2_CKPT_ABS="$(abs "$SAM2_CKPT")"
mkdir -p "$SAVE_ROOT_PC_ABS" "$EXP_DIR_ABS"
echo ">>> RUN_ID: ${RUN_ID}"
echo ">>> DATA_DIR: ${DATA_DIR_ABS}"
echo ">>> SAVE_ROOT_PC: ${SAVE_ROOT_PC_ABS}"
echo ">>> EXP_DIR: ${EXP_DIR_ABS}"

# ============== Stage 1: Preprocess ==============
echo "[Stage 1/3] Preprocess → ${SAVE_ROOT_PC_ABS}"
python data_gen/preprocess_poincept_deploy.py \
  "${DATA_DIR_ABS}" \
  --plate_name "${PLATE_NAME}" \
  --pillar_name "${PILLAR_NAME}" \
  --baseline "${BASELINE}" \
  --save_dir "${SAVE_ROOT_PC_ABS}" \
  --split test \
  --gdino_cfg "${GDINO_CFG_ABS}" \
  --gdino_ckpt "${GDINO_CKPT_ABS}" \
  --sam2_cfg "${SAM2_CFG}" \
  --sam2_ckpt "${SAM2_CKPT_ABS}" \
  --gsam2_device cuda \
  --box_thresh 0.35 \
  --text_thresh 0.25 \
  --nms_iou 0.5 \
  --per_class_topk 1 \
  --min_uv_edge 5 --min_xyz_edge 5 --min_seg_points 50
# --- 전처리 “성공 여부” 판정: test 샘플 수 세기 ---
TEST_DIR="${SAVE_ROOT_PC_ABS}/test"
SAMPLE_COUNT="$(find "${TEST_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l || true)"
echo "[Check] Preprocess produced ${SAMPLE_COUNT} test sample(s)"
if [[ "${SAMPLE_COUNT}" -eq 0 ]]; then
  echo "❌ Preprocess 실패: 저장된 test 샘플이 0개입니다. 파이프라인을 중단합니다."
  # 상태 파일로 남겨두기
  printf '{"run_id":"%s","stage":"preprocess","status":"failed","reason":"no_test_samples"}\n' "$RUN_ID" \
    > "${EXP_DIR_ABS}/status.json"
  exit 2
fi
# ============== Stage 2: Pointcept Test ==============
echo "[Stage 2/3] Run Pointcept Test → ${EXP_DIR_ABS}"
pushd Pointcept >/dev/null
PYTHONPATH=./ \
python tools/test.py \
  --config-file "${CONFIG_FILE_ABS}" \
  --num-gpus 1 \
  --options \
    save_path="${EXP_DIR_ABS}" \
    weight="${WEIGHT_ABS}" \
    data.train.data_root="${SAVE_ROOT_PC_ABS}" \
    data.val.data_root="${SAVE_ROOT_PC_ABS}" \
    data.test.data_root="${SAVE_ROOT_PC_ABS}"
popd >/dev/null
# # --- Pointcept “성공 여부” 판정: 처리된 샘플 로그 개수 세기 ---
# # LOG_FILE="${EXP_DIR_ABS}/test.log"
# # PROCESSED_COUNT=0
# # if [[ -f "${LOG_FILE}" ]]; then
# #   # "Test: ..." 라인이 몇 개 나왔는지로 샘플 처리 수 추정
# #   PROCESSED_COUNT="$(grep -cE '^Test:' "${LOG_FILE}" || true)"
# # fi
# # echo "[Check] Pointcept processed ${PROCESSED_COUNT} sample(s)"
# # if [[ "${PROCESSED_COUNT}" -eq 0 ]]; then
# #   echo "❌ Pointcept 실패: 처리된 샘플이 0개입니다. (log=${LOG_FILE})"
# #   printf '{"run_id":"%s","stage":"pointcept","status":"failed","reason":"no_processed_samples"}\n' "$RUN_ID" \
# #     > "${EXP_DIR_ABS}/status.json"
# #   exit 3
# # fi
# # ============== Stage 3: Run summary & latest 링크 갱신 ==============
# printf '{"run_id":"%s","stage":"all","status":"ok","samples_preprocessed":%d,"samples_processed":%d}\n' \
#   "$RUN_ID" "$SAMPLE_COUNT" "$PROCESSED_COUNT" > "${EXP_DIR_ABS}/status.json"
# # latest 심볼릭 링크 갱신(있으면 대체)
# ln -sfn "${SAVE_ROOT_PC_ABS}" "${LATEST_DATA_LINK}"
# ln -sfn "${EXP_DIR_ABS}"      "${LATEST_EXP_LINK}"
# echo "✅ 완료"
# echo "   - data: ${SAVE_ROOT_PC_ABS}   (latest → ${LATEST_DATA_LINK})"
# echo "   - exp : ${EXP_DIR_ABS}        (latest → ${LATEST_EXP_LINK})"
# echo "   - log : ${EXP_DIR_ABS}/test.log"
# echo "   - status: ${EXP_DIR_ABS}/status.json"

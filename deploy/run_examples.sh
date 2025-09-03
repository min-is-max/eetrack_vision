#!/usr/bin/env bash
set -e
IMG="ghcr.io/<ORG>/wld:runtime-2.1.0-cu124"   # ← 바꿔 넣기

# 1) Depth
docker run --gpus all --rm -it --shm-size=16g \
  -v $PWD/sample_data:/data -v $PWD/outputs:/outputs \
  $IMG wld depth \
    --left_dir /data/left \
    --right_dir /data/right \
    --K_dir /data/intrinsics \
    --baseline 0.12 \
    --result_dir /outputs/depth

# 2) Segmentation
docker run --gpus all --rm -it \
  -v $PWD/sample_data:/data -v $PWD/outputs:/outputs \
  $IMG wld seg \
    --image_dir /data/left \
    --text_query "upper plate, bottom plate" \
    --result_dir /outputs/seg

# 3) (옵션) Tracking
docker run --gpus all --rm -it \
  -v $PWD/sample_data:/data -v $PWD/outputs:/outputs \
  $IMG wld track \
    --image_dir /data/left \
    --mask_dir /outputs/seg \
    --result_dir /outputs/seg_track

# 4) Line (단일 프레임 예시)
docker run --gpus all --rm -it \
  -v $PWD/sample_data:/data -v $PWD/outputs:/outputs \
  $IMG wld line \
    --left  /data/left/000000.png \
    --depth /outputs/depth/000000.png \
    --K     /data/intrinsics/000000.txt \
    --outdir /outputs/lines

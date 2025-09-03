#!/usr/bin/env bash
set -e
IMG="$1"

# 최소 스모크: depth 한 프레임 뽑히는지만 빠르게 확인
docker run --gpus all --rm -v $PWD/sample_data:/data -v $PWD/outputs:/outputs \
  $IMG wld depth --left_dir /data/left --right_dir /data/right --K_dir /data/intrinsics \
                 --baseline 0.12 --result_dir /outputs/depth

test -f outputs/depth/000000.png && echo "[OK] depth"

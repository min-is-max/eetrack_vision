#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/welding_line_detection"
PY="/workspace/isaaclab/_isaac_sim/python.sh"   # Isaac Lab Python
export PYTHONUNBUFFERED=1
export ROOT
# 모듈 경로(레포 구조 기준)
export PYTHONPATH="$ROOT:$ROOT/script:$ROOT/FoundationStereo:$ROOT/Grounded-SAM-2:$ROOT/Pointcept:${PYTHONPATH:-}"

# 특정 파일이 있는 디렉토리를 찾아 PYTHONPATH에 추가
ensure_module_dir () {
  local target="$1"
  local dir
  dir=$("$PY" - <<'PY'
import os, glob
root = os.environ["ROOT"]
target = os.environ["TARGET"]
c = sorted(glob.glob(os.path.join(root, "**", target), recursive=True))
print(os.path.dirname(c[0]) if c else "")
PY
)
  if [ -n "$dir" ]; then
    export PYTHONPATH="$dir:$PYTHONPATH"
  fi
}

cmd="${1:-help}"; shift || true
cd "$ROOT" || true

case "$cmd" in
  depth)
    # depth_estimator.py 위치를 찾아 경로 추가
    TARGET="depth_estimator.py" ensure_module_dir "depth_estimator.py"
    exec "$PY" script/depth_estimating.py "$@"
    ;;
  seg)
    exec "$PY" script/gsa2_image_segmentation.py "$@"
    ;;
  track)
    exec "$PY" script/gsa2_video_tracking.py "$@"
    ;;
  line)
    exec "$PY" script/line_detecting.py "$@"
    ;;
  help|--help|-h|"")
    cat <<'USAGE'
Usage: wld {depth|seg|track|line} [args...]
  depth : stereo -> depth (FoundationStereo)
  seg   : Grounded-SAM2 segmentation
  track : mask video tracking
  line  : RGB-D -> PCD -> B-spline fitting
USAGE
    ;;
  *)
    echo "Unknown command: $cmd"; exit 1 ;;
esac

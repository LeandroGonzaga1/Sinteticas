#!/bin/bash
set -e
echo "=== Wan2.1 I2V — models pre-baked, starting ComfyUI ==="
exec python main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --disable-auto-launch \
  --highvram \
  --cuda-device 0

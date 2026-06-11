#!/bin/bash
set -e

MODELS_DIR="/app/ComfyUI/models"
I2V_MODEL="$MODELS_DIR/diffusion_models/I2V/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors"

echo "=== Wan2.1 I2V Entrypoint ==="

if [ ! -f "$I2V_MODEL" ]; then
    echo "Downloading Wan2.1 I2V models..."
    pip install huggingface_hub -q
    python -c "
from huggingface_hub import hf_hub_download

# I2V model
hf_hub_download(
    repo_id='Kijai/WanVideo_comfy',
    filename='Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors',
    local_dir='/app/ComfyUI/models/diffusion_models/I2V'
)

# T5 encoder
hf_hub_download(
    repo_id='Kijai/WanVideo_comfy',
    filename='umt5-xxl-enc-fp8_e4m3fn.safetensors',
    local_dir='/app/ComfyUI/models/text_encoders'
)

# VAE
hf_hub_download(
    repo_id='Kijai/WanVideo_comfy',
    filename='Wan2_1_VAE_bf16.safetensors',
    local_dir='/app/ComfyUI/models/vae/wanvideo'
)

# CLIP Vision
hf_hub_download(
    repo_id='openai/clip-vit-large-patch14',
    filename='pytorch_model.bin',
    local_dir='/app/ComfyUI/models/clip_vision'
)
print('Models OK')
"
fi

echo "Starting ComfyUI on A100..."
exec python main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --disable-auto-launch \
    --highvram \
    --cuda-device 0

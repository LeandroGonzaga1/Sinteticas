"""Bake Wan2.1 models into the Docker image at build time.

Filenames verified from original entrypoint.sh (commit c7ec0a5).
CLIP ViT-H-14 downloaded from h94/IP-Adapter and renamed to match workflow.
"""
import os
import shutil
from huggingface_hub import hf_hub_download

# I2V model, T5 encoder, VAE — all from Kijai/WanVideo_comfy
DIRECT_DOWNLOADS = [
    ("Kijai/WanVideo_comfy",
     "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
     "/app/ComfyUI/models/diffusion_models/I2V"),
    ("Kijai/WanVideo_comfy",
     "umt5-xxl-enc-fp8_e4m3fn.safetensors",
     "/app/ComfyUI/models/text_encoders"),
    ("Kijai/WanVideo_comfy",
     "Wan2_1_VAE_bf16.safetensors",
     "/app/ComfyUI/models/vae"),
]

for repo, fname, dest in DIRECT_DOWNLOADS:
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading {fname} from {repo} ...")
    hf_hub_download(repo_id=repo, filename=fname, local_dir=dest)
    path = os.path.join(dest, fname)
    size_gb = os.path.getsize(path) / 1e9
    print(f"  -> {path}  ({size_gb:.1f} GB)")

# CLIP ViT-H-14 — h94/IP-Adapter has it as "models/image_encoder/model.safetensors"
# Download to /tmp then rename to the filename the ComfyUI workflow expects.
print("Downloading CLIP ViT-H-14 from h94/IP-Adapter ...")
clip_dest_dir = "/app/ComfyUI/models/clip_vision"
os.makedirs(clip_dest_dir, exist_ok=True)
clip_tmp = hf_hub_download(
    repo_id="h94/IP-Adapter",
    filename="models/image_encoder/model.safetensors",
    local_dir="/tmp/clip_dl",
)
clip_final = os.path.join(clip_dest_dir, "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors")
shutil.copy2(clip_tmp, clip_final)
size_gb = os.path.getsize(clip_final) / 1e9
print(f"  -> {clip_final}  ({size_gb:.1f} GB)")

print("All models downloaded.")

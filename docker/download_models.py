"""Bake Wan2.1 models into the Docker image at build time."""
import os
from huggingface_hub import hf_hub_download

DOWNLOADS = [
    ("Kijai/WanVideo_comfy",
     "Wan2_1-I2V-14B-480p_fp8_e4m3fn.safetensors",
     "/app/ComfyUI/models/diffusion_models/I2V"),
    ("Kijai/WanVideo_comfy",
     "umt5-xxl-enc-fp8_e4m3fnuz.safetensors",
     "/app/ComfyUI/models/text_encoders"),
    ("Kijai/WanVideo_comfy",
     "Wan2_1_VAE_bf16.safetensors",
     "/app/ComfyUI/models/vae"),
    ("openai/clip-vit-large-patch14",
     "model.safetensors",
     "/app/ComfyUI/models/clip"),
]

for repo, fname, dest in DOWNLOADS:
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading {fname} from {repo} ...")
    hf_hub_download(repo_id=repo, filename=fname, local_dir=dest)
    path = os.path.join(dest, fname)
    size_gb = os.path.getsize(path) / 1e9
    print(f"  -> {path}  ({size_gb:.1f} GB)")

print("All models downloaded.")

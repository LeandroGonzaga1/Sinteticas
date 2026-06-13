"""
Bake Wan2.1 models at Docker build time.

Robust: lists actual files in HuggingFace repo before downloading,
then creates canonical symlinks so the ComfyUI workflow never needs
to know the exact HuggingFace filename.

Canonical names (used in i2v_workflow_a100.py):
  diffusion_models/I2V/wan_i2v_480p.safetensors
  text_encoders/wan_t5_encoder.safetensors
  vae/wan_vae.safetensors
  clip_vision/wan_clip_h14.safetensors
"""
import os
import shutil

from huggingface_hub import HfApi, hf_hub_download

api = HfApi()

# ── 1. Discover Kijai/WanVideo_comfy ────────────────────────────────────────
print("\n=== Listing Kijai/WanVideo_comfy ===")
wan_files = list(api.list_repo_files("Kijai/WanVideo_comfy"))
for f in wan_files:
    print(f"  {f}")

# I2V 480p (fp8) — match any variant (_scaled_KJ, plain, etc.)
i2v_candidates = sorted(
    f for f in wan_files
    if "I2V" in f and "480p" in f and f.endswith(".safetensors")
)
if not i2v_candidates:
    raise RuntimeError(f"No I2V 480p model found. Files available: {wan_files}")
i2v_file = i2v_candidates[0]
print(f"\n[I2V] using: {i2v_file}")

# UMT5 text encoder (fp8, any variant)
t5_candidates = sorted(
    f for f in wan_files
    if "umt5" in f.lower() and f.endswith(".safetensors")
)
if not t5_candidates:
    raise RuntimeError(f"No UMT5 encoder found. Files available: {wan_files}")
t5_file = t5_candidates[0]
t5_quant = "fp8_e4m3fnuz" if "fnuz" in t5_file else "fp8_e4m3fn"
print(f"[T5 ] using: {t5_file}  (quantization={t5_quant})")

# VAE
vae_candidates = sorted(
    f for f in wan_files
    if "VAE" in f and f.endswith(".safetensors")
)
if not vae_candidates:
    raise RuntimeError(f"No VAE found. Files available: {wan_files}")
vae_file = vae_candidates[0]
print(f"[VAE] using: {vae_file}")

# ── 2. Download + canonical symlinks ────────────────────────────────────────
def dl_and_link(repo, filename, dest_dir, canonical_name):
    os.makedirs(dest_dir, exist_ok=True)
    hf_hub_download(repo_id=repo, filename=filename, local_dir=dest_dir)
    actual   = os.path.join(dest_dir, filename)
    symlink  = os.path.join(dest_dir, canonical_name)
    if os.path.lexists(symlink):
        os.remove(symlink)
    os.symlink(actual, symlink)
    size_gb = os.path.getsize(actual) / 1e9
    print(f"  OK  {actual} ({size_gb:.1f} GB)")
    print(f"  sym {symlink} -> {filename}")

dl_and_link(
    "Kijai/WanVideo_comfy", i2v_file,
    "/app/ComfyUI/models/diffusion_models/I2V",
    "wan_i2v_480p.safetensors",
)
dl_and_link(
    "Kijai/WanVideo_comfy", t5_file,
    "/app/ComfyUI/models/text_encoders",
    "wan_t5_encoder.safetensors",
)
dl_and_link(
    "Kijai/WanVideo_comfy", vae_file,
    "/app/ComfyUI/models/vae",
    "wan_vae.safetensors",
)

# ── 3. CLIP ViT-H-14 (h94/IP-Adapter, ViT-H/14 safetensors) ────────────────
print("\n[CLIP] downloading from h94/IP-Adapter ...")
CLIP_DIR = "/app/ComfyUI/models/clip_vision"
os.makedirs(CLIP_DIR, exist_ok=True)
clip_tmp = hf_hub_download(
    repo_id="h94/IP-Adapter",
    filename="models/image_encoder/model.safetensors",
    local_dir="/tmp/clip_dl",
)
clip_dest = os.path.join(CLIP_DIR, "wan_clip_h14.safetensors")
shutil.copy2(clip_tmp, clip_dest)
size_gb = os.path.getsize(clip_dest) / 1e9
print(f"  OK  {clip_dest} ({size_gb:.1f} GB)")

# ── 4. Write model config (read by submit_i2v_job.py for quantization) ──────
import json
config = {
    "i2v_model":    f"I2V/{i2v_file}",
    "i2v_canonical": "I2V/wan_i2v_480p.safetensors",
    "t5_model":     t5_file,
    "t5_canonical": "wan_t5_encoder.safetensors",
    "t5_quant":     t5_quant,
    "vae_model":    vae_file,
    "vae_canonical": "wan_vae.safetensors",
    "clip_model":   "wan_clip_h14.safetensors",
}
config_path = "/app/wanvideo_models.json"
with open(config_path, "w") as fh:
    json.dump(config, fh, indent=2)
print(f"\nModel config: {config_path}")
print(json.dumps(config, indent=2))
print("\n=== All models baked into image ===")

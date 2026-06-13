"""
Bake Wan2.1 I2V models into the Docker image at build time.

Robustness model:
  - EXACT filenames, verified to exist in the HF repos on 2026-06-13.
    (An earlier sorted()[0] heuristic silently picked the WRONG files:
    a Lightx2v LoRA instead of the base I2V model, a bf16 encoder instead
    of fp8, and an MTVCrafter VAE. So: explicit names, no guessing.)
  - FAIL LOUD: before downloading, list the repo and assert each exact file
    exists. If a file was renamed upstream, the build fails immediately and
    prints the available candidates instead of baking a wrong model.
  - Canonical symlinks: the ComfyUI workflow references stable names
    (wan_i2v_480p.safetensors, ...). Only this file changes if HF renames.

Canonical names (must match spheron/i2v_workflow_a100.py):
  diffusion_models/I2V/wan_i2v_480p.safetensors
  text_encoders/wan_t5_encoder.safetensors
  vae/wan_vae.safetensors
  clip_vision/wan_clip_h14.safetensors
"""
import json
import os
import shutil

from huggingface_hub import HfApi, hf_hub_download

api = HfApi()

# (repo_id, exact_filename, dest_dir, canonical_symlink_name)
WAN_REPO = "Kijai/WanVideo_comfy"
PLAN = [
    (WAN_REPO,
     "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",      # NOTE: 480P uppercase, no _scaled_KJ
     "/app/ComfyUI/models/diffusion_models/I2V",
     "wan_i2v_480p.safetensors"),
    (WAN_REPO,
     "umt5-xxl-enc-fp8_e4m3fn.safetensors",
     "/app/ComfyUI/models/text_encoders",
     "wan_t5_encoder.safetensors"),
    (WAN_REPO,
     "Wan2_1_VAE_bf16.safetensors",
     "/app/ComfyUI/models/vae",
     "wan_vae.safetensors"),
]

# Fail-loud existence check for the Kijai repo files.
print(f"=== Verifying files exist in {WAN_REPO} ===")
wan_files = set(api.list_repo_files(WAN_REPO))
missing = [fname for repo, fname, _, _ in PLAN if repo == WAN_REPO and fname not in wan_files]
if missing:
    print("!!! MISSING files — upstream may have renamed them. Candidates:")
    for f in sorted(x for x in wan_files if x.endswith(".safetensors")):
        print("   ", f)
    raise SystemExit(f"FATAL: these exact files are not in {WAN_REPO}: {missing}")
print("All Kijai files present.")


def dl_and_link(repo, filename, dest_dir, canonical_name):
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Downloading {repo}/{filename} ...")
    hf_hub_download(repo_id=repo, filename=filename, local_dir=dest_dir)
    actual = os.path.join(dest_dir, filename)
    link = os.path.join(dest_dir, canonical_name)
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(actual, link)
    size_gb = os.path.getsize(actual) / 1e9
    print(f"  OK  {actual} ({size_gb:.1f} GB)")
    print(f"  sym {link} -> {filename}")


for repo, fname, dest, canon in PLAN:
    dl_and_link(repo, fname, dest, canon)

# CLIP ViT-H-14 visual encoder — from h94/IP-Adapter, copied (not symlinked,
# because hf nests it under models/image_encoder/).
print("\nDownloading CLIP ViT-H-14 from h94/IP-Adapter ...")
clip_dir = "/app/ComfyUI/models/clip_vision"
os.makedirs(clip_dir, exist_ok=True)
clip_src = hf_hub_download(
    repo_id="h94/IP-Adapter",
    filename="models/image_encoder/model.safetensors",
    local_dir="/tmp/clip_dl",
)
clip_dst = os.path.join(clip_dir, "wan_clip_h14.safetensors")
shutil.copy2(clip_src, clip_dst)
print(f"  OK  {clip_dst} ({os.path.getsize(clip_dst)/1e9:.1f} GB)")

# Config consumed by the workflow builder for the T5 quantization mode.
config = {
    "i2v_model":    "I2V/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
    "i2v_canonical": "I2V/wan_i2v_480p.safetensors",
    "i2v_quant":    "fp8_e4m3fn",
    "t5_model":     "umt5-xxl-enc-fp8_e4m3fn.safetensors",
    "t5_canonical": "wan_t5_encoder.safetensors",
    "t5_quant":     "fp8_e4m3fn",
    "vae_canonical": "wan_vae.safetensors",
    "clip_canonical": "wan_clip_h14.safetensors",
}
with open("/app/wanvideo_models.json", "w") as fh:
    json.dump(config, fh, indent=2)
print("\nModel config /app/wanvideo_models.json:")
print(json.dumps(config, indent=2))
print("\n=== All models baked correctly ===")

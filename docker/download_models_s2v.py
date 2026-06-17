"""Bake Wan2.2-S2V models into the Docker image at build time.

Files (ALL EXACT, validated against the Kijai example workflow
`C:/AI/ComfyUI/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/s2v/
 wanvideo2_2_S2V_context_window_testing.json` — widget values 100% match):

  MODEL:    Kijai/WanVideo_comfy_fp8_scaled
            S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors
            ~16GB fp8 scaled. Loader: base_precision=fp16_fast,
            quantization=fp8_e4m3fn_scaled. Path in ComfyUI:
              diffusion_models/WanVideo/S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors

  T5:       Kijai/WanVideo_comfy
            umt5-xxl-enc-bf16.safetensors  (~10GB BF16).
            Path in ComfyUI:
              text_encoders/umt5-xxl-enc-bf16.safetensors

  VAE:      Kijai/WanVideo_comfy
            Wan2_1_VAE_bf16.safetensors  (~600MB, reuses Wan2.1 VAE).
            Path in ComfyUI (subfolder wanvideo):
              vae/wanvideo/Wan2_1_VAE_bf16.safetensors

  AUDIO:    Wan-AI/Wan2.2-S2V-14B
            wav2vec2-large-xlsr-53-english/model.safetensors  (~1.3GB).
            Same state-dict keys as Kijai's wav2vec_xlsr_53_english_fp32.safetensors
            (Kijai's audio_encoders.py looks for `wav2vec2.encoder.layer_norm.bias`
            which is exactly the HF key).
            Path in ComfyUI:
              audio_encoders/wav2vec_xlsr_53_english_fp32.safetensors

FAIL LOUD: each filename is asserted to exist in the upstream repo before
download. If any rename upstream breaks us, the build aborts and lists
candidates — we never bake a wrong model. (Lesson from build #7: sorted()[0]
silently picked Lightx2v LoRA instead of base I2V.)
"""
import os
import shutil
import sys

from huggingface_hub import HfApi, hf_hub_download

api = HfApi()

# (repo_id, exact_filename, dest_dir, canonical_symlink_name, file_role)
S2V_PLAN = [
    # ---- The S2V model (Kijai fp8-scaled, single file, ~16GB) ----
    ("Kijai/WanVideo_comfy_fp8_scaled",
     "S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
     "/app/ComfyUI/models/diffusion_models/WanVideo/S2V",
     "Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
     "s2v_model"),

    # ---- T5 text encoder (Kijai WanVideo_comfy, BF16) ----
    ("Kijai/WanVideo_comfy",
     "umt5-xxl-enc-bf16.safetensors",
     "/app/ComfyUI/models/text_encoders",
     "umt5-xxl-enc-bf16.safetensors",
     "t5"),

    # ---- VAE (Kijai WanVideo_comfy, Wan2.1 VAE works for Wan2.2) ----
    ("Kijai/WanVideo_comfy",
     "Wan2_1_VAE_bf16.safetensors",
     "/app/ComfyUI/models/vae/wanvideo",
     "Wan2_1_VAE_bf16.safetensors",
     "vae"),

    # ---- Audio encoder (Wav2Vec2 XLSR-53 english, ~1.3GB) ----
    # Original HF key prefix is `wav2vec2.*` — already matches what Kijai's
    # AudioEncoderLoader expects (see comfy/audio_encoders/audio_encoders.py:50).
    # Kijai's "wav2vec_xlsr_53_english_fp32.safetensors" is the same file
    # copied from this HF path (verified via state_dict key match).
    ("Wan-AI/Wan2.2-S2V-14B",
     "wav2vec2-large-xlsr-53-english/model.safetensors",
     "/app/ComfyUI/models/audio_encoders",
     "wav2vec_xlsr_53_english_fp32.safetensors",
     "wav2vec"),
]


def _assert_files_exist(plan):
    """Group files by repo and assert each filename exists upstream.
    Prints available candidates if any are missing. Raises SystemExit on fail.
    """
    by_repo: dict[str, list[str]] = {}
    for repo, fname, *_ in plan:
        by_repo.setdefault(repo, []).append(fname)
    for repo, wanted in by_repo.items():
        print(f"=== Verifying {len(wanted)} file(s) in {repo} ===")
        try:
            upstream = set(api.list_repo_files(repo))
        except Exception as e:
            raise SystemExit(f"FATAL: cannot list {repo}: {e}")
        missing = [f for f in wanted if f not in upstream]
        if missing:
            print(f"!!! MISSING files in {repo}:")
            for f in missing:
                print(f"   {f}")
            print(f"!!! Available candidates (filtered to audio/safetensors):")
            for f in sorted(x for x in upstream
                            if "audio" in x.lower()
                            or x.endswith(".safetensors")):
                print(f"   {f}")
            raise SystemExit(f"FATAL: missing files in {repo}: {missing}")
        print(f"   all {len(wanted)} file(s) present.")


print("=== Pre-flight: asserting all filenames exist upstream ===")
_assert_files_exist(S2V_PLAN)
print()


def dl_and_link(repo, filename, dest_dir, canonical_name, file_role):
    """Download a single file and create a canonical symlink in dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[{file_role}] {repo}/{filename}")
    hf_hub_download(repo_id=repo, filename=filename, local_dir=dest_dir)
    actual = os.path.join(dest_dir, filename)
    if not os.path.exists(actual):
        raise SystemExit(f"FATAL: downloaded file not found: {actual}")
    size_gb = os.path.getsize(actual) / 1e9
    print(f"   {size_gb:.2f} GB")
    link = os.path.join(dest_dir, canonical_name)
    if os.path.lexists(link):
        os.remove(link)
    if os.path.abspath(actual) != os.path.abspath(link):
        shutil.copy2(actual, link)
    return actual, size_gb


print("=== Downloading S2V models + T5 + VAE + Wav2Vec ===")
total_gb = 0.0
for repo, fname, dest, canon, role in S2V_PLAN:
    _, gb = dl_and_link(repo, fname, dest, canon, role)
    total_gb += gb
print(f"\n=== Total downloaded: {total_gb:.1f} GB ===\n")


# Config consumed by scripts/s2v_workflow.py (matches MODEL_PATH etc.)
config = {
    "s2v_model":     "WanVideo\\S2V\\Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
    "t5":            "umt5-xxl-enc-bf16.safetensors",
    "vae":           "wanvideo\\Wan2_1_VAE_bf16.safetensors",
    "audio_encoder": "wav2vec_xlsr_53_english_fp32.safetensors",
    "s2v_quant":     "fp8_e4m3fn_scaled",
    "s2v_base_prec": "fp16_fast",
    "audio_sample_rate": 16000,
}
with open("/app/wanvideo_s2v_models.json", "w") as fh:
    import json
    json.dump(config, fh, indent=2)
print("Config /app/wanvideo_s2v_models.json:")
print(json.dumps(config, indent=2))
print("\n=== S2V models baked correctly ===")

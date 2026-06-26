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

# (repo_id, exact_filename, dest_dir, canonical_symlink_name, file_role,
#  min_size_gb)
#
# ORDER MATTERS: smallest files first. The GitHub-hosted runner has limited
# disk even after free-disk-space (~25-30GB usable once the build's own
# layers — torch base, ComfyUI clone, custom node deps — are accounted for).
# Build #27670524148 baked the ~16GB S2V model + ~10GB T5 successfully but
# silently produced empty/missing VAE (~0.6GB) and T5 dropdowns at runtime —
# consistent with disk exhaustion partway through the 4-file sequence (the
# 2 largest files first leave the least headroom for the smaller ones after
# them). Downloading small-to-large fails fast on the cheap files instead of
# burning 25+ min before discovering the run was always going to run out of
# space.
S2V_PLAN = [
    # ---- VAE (Kijai WanVideo_comfy, Wan2.1 VAE works for Wan2.2, ~0.6GB) ----
    ("Kijai/WanVideo_comfy",
     "Wan2_1_VAE_bf16.safetensors",
     "/app/ComfyUI/models/vae/wanvideo",
     "Wan2_1_VAE_bf16.safetensors",
     "vae",
     0.3),

    # ---- Audio encoder (Wav2Vec2 XLSR-53 english, ~1.3GB) ----
    # Original HF key prefix is `wav2vec2.*` — already matches what Kijai's
    # AudioEncoderLoader expects (see comfy/audio_encoders/audio_encoders.py:50).
    # Kijai's "wav2vec_xlsr_53_english_fp32.safetensors" is the same file
    # copied from this HF path (verified via state_dict key match).
    ("Wan-AI/Wan2.2-S2V-14B",
     "wav2vec2-large-xlsr-53-english/model.safetensors",
     "/app/ComfyUI/models/audio_encoders",
     "wav2vec_xlsr_53_english_fp32.safetensors",
     "wav2vec",
     0.8),

    # ---- T5 text encoder (Kijai WanVideo_comfy, BF16, ~10GB) ----
    ("Kijai/WanVideo_comfy",
     "umt5-xxl-enc-bf16.safetensors",
     "/app/ComfyUI/models/text_encoders",
     "umt5-xxl-enc-bf16.safetensors",
     "t5",
     6.0),

    # ---- The S2V model (Kijai fp8-scaled, single file, ~16GB) ----
    ("Kijai/WanVideo_comfy_fp8_scaled",
     "S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
     "/app/ComfyUI/models/diffusion_models/WanVideo/S2V",
     "Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
     "s2v_model",
     10.0),
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


def _free_gb(path="/"):
    usage = shutil.disk_usage(path)
    return usage.free / 1e9


def dl_and_link(repo, filename, dest_dir, canonical_name, file_role, min_size_gb):
    """Download a single file and create a canonical symlink in dest_dir.

    FAIL LOUD on two distinct failure modes that previously passed silently:
      1. hf_hub_download raises -> already propagates (no change needed).
      2. hf_hub_download (or the disk under it) truncates/short-writes the
         file without raising -> `os.path.exists()` alone does NOT catch
         this (a 0-byte or partial file still "exists"). We now also assert
         a minimum size per file, derived from the known model size, so a
         disk-full mid-download or interrupted transfer aborts the build
         instead of baking a corrupt/empty model.
    """
    os.makedirs(dest_dir, exist_ok=True)
    free_before = _free_gb()
    print(f"[{file_role}] {repo}/{filename}  (disk free: {free_before:.1f} GB)")
    hf_hub_download(repo_id=repo, filename=filename, local_dir=dest_dir)
    actual = os.path.join(dest_dir, filename)
    if not os.path.exists(actual):
        raise SystemExit(
            f"FATAL [{file_role}]: downloaded file not found: {actual} "
            f"(disk free was {free_before:.1f} GB before this download)"
        )
    size_gb = os.path.getsize(actual) / 1e9
    if size_gb < min_size_gb:
        raise SystemExit(
            f"FATAL [{file_role}]: {actual} is {size_gb:.3f} GB, expected "
            f">= {min_size_gb} GB. File is truncated/empty — likely disk "
            f"exhaustion mid-download (disk free was {free_before:.1f} GB "
            f"before this download started, now {_free_gb():.1f} GB)."
        )
    print(f"   {size_gb:.2f} GB  (disk free now: {_free_gb():.1f} GB)")
    link = os.path.join(dest_dir, canonical_name)
    if os.path.lexists(link):
        os.remove(link)
    if os.path.abspath(actual) != os.path.abspath(link):
        shutil.copy2(actual, link)
        if not os.path.exists(link) or os.path.getsize(link) < min_size_gb * 1e9:
            raise SystemExit(
                f"FATAL [{file_role}]: copy2 to canonical name {link} "
                f"failed or produced a truncated file (disk free: "
                f"{_free_gb():.1f} GB)."
            )
    return actual, size_gb


print("=== Downloading S2V models + T5 + VAE + Wav2Vec ===")
print(f"Disk free at start: {_free_gb():.1f} GB")
total_gb = 0.0
for repo, fname, dest, canon, role, min_gb in S2V_PLAN:
    _, gb = dl_and_link(repo, fname, dest, canon, role, min_gb)
    total_gb += gb
print(f"\n=== Total downloaded: {total_gb:.1f} GB  (disk free now: {_free_gb():.1f} GB) ===\n")


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

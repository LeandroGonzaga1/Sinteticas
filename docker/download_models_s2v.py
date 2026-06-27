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
#
# No hardcoded min_size_gb guesses — build #28250424800 failed in 3min16s
# because the guessed VAE threshold (0.3GB) was HIGHER than the real file
# (0.254GB), a false-positive FATAL on the very first (smallest) download.
# Real sizes are now fetched from the HF API at pre-flight time (see
# _assert_files_exist_and_get_sizes) and used directly as the size-check
# floor (95% of real size — tolerant of metadata rounding, still catches
# truncation).
#
# ORDER MATTERS: smallest files first — fails fast on cheap files instead of
# burning 25+ min before discovering a real problem (disk, auth, network).
S2V_PLAN = [
    # ---- VAE (Kijai WanVideo_comfy, Wan2.1 VAE works for Wan2.2, ~0.25GB) ----
    ("Kijai/WanVideo_comfy",
     "Wan2_1_VAE_bf16.safetensors",
     "/app/ComfyUI/models/vae/wanvideo",
     "Wan2_1_VAE_bf16.safetensors",
     "vae"),

    # ---- Audio encoder (Wav2Vec2 XLSR-53 english, ~1.26GB) ----
    # Original HF key prefix is `wav2vec2.*` — already matches what Kijai's
    # AudioEncoderLoader expects (see comfy/audio_encoders/audio_encoders.py:50).
    # Kijai's "wav2vec_xlsr_53_english_fp32.safetensors" is the same file
    # copied from this HF path (verified via state_dict key match).
    ("Wan-AI/Wan2.2-S2V-14B",
     "wav2vec2-large-xlsr-53-english/model.safetensors",
     "/app/ComfyUI/models/audio_encoders",
     "wav2vec_xlsr_53_english_fp32.safetensors",
     "wav2vec"),

    # ---- T5 text encoder (Kijai WanVideo_comfy, BF16, ~11.4GB) ----
    ("Kijai/WanVideo_comfy",
     "umt5-xxl-enc-bf16.safetensors",
     "/app/ComfyUI/models/text_encoders",
     "umt5-xxl-enc-bf16.safetensors",
     "t5"),

    # ---- The S2V model (Kijai fp8-scaled, single file, ~16.7GB) ----
    ("Kijai/WanVideo_comfy_fp8_scaled",
     "S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
     "/app/ComfyUI/models/diffusion_models/WanVideo/S2V",
     "Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
     "s2v_model"),
]


def _assert_files_exist_and_get_sizes(plan):
    """Group files by repo, assert each filename exists upstream, and return
    {(repo, filename): size_bytes} from HF's own metadata.

    Real sizes replace hardcoded min_size_gb guesses — build #28250424800
    failed in 3min16s because a guessed VAE threshold (0.3GB) was HIGHER
    than the real file (0.254GB): a false-positive FATAL on the very first
    (smallest) download. Asking HF for the real size up front removes the
    entire class of "guessed the wrong constant" bug.
    """
    by_repo: dict[str, list[str]] = {}
    for repo, fname, *_ in plan:
        by_repo.setdefault(repo, []).append(fname)
    sizes: dict[tuple[str, str], int] = {}
    for repo, wanted in by_repo.items():
        print(f"=== Verifying {len(wanted)} file(s) in {repo} ===")
        try:
            info = api.repo_info(repo, files_metadata=True)
        except Exception as e:
            raise SystemExit(f"FATAL: cannot inspect {repo}: {e}")
        upstream = {f.rfilename: f.size for f in info.siblings}
        missing = [f for f in wanted if f not in upstream]
        if missing:
            print(f"!!! MISSING files in {repo}:")
            for f in missing:
                print(f"   {f}")
            print(f"!!! Available candidates (filtered to audio/safetensors):")
            for f in sorted(x for x in upstream
                            if "audio" in x.lower()
                            or x.endswith(".safetensors")):
                print(f"   {x}")
            raise SystemExit(f"FATAL: missing files in {repo}: {missing}")
        for f in wanted:
            sizes[(repo, f)] = upstream[f]
            print(f"   {f}: {upstream[f]/1e9:.3f} GB (real size, from HF metadata)")
    return sizes


print("=== Pre-flight: asserting all filenames exist upstream ===")
REAL_SIZES = _assert_files_exist_and_get_sizes(S2V_PLAN)
print()


def _free_gb(path="/"):
    usage = shutil.disk_usage(path)
    return usage.free / 1e9


def dl_and_link(repo, filename, dest_dir, canonical_name, file_role, real_size_bytes):
    """Download a single file into the default HF cache, then MOVE (rename,
    not copy) its bytes into dest_dir/canonical_name.

    Build #7 (ad83b0d) proved hf_hub_download(..., local_dir=dest_dir) is
    not safe inside a Docker RUN layer: the file existed with the right size
    in the SAME RUN/layer, yet the NEXT layer's `test -s` guard saw it as
    missing/empty — the signature of a hardlink (two paths, one inode) that
    BuildKit's layer-diff didn't fully promote.

    Build #8 (a396251) fixed that with shutil.copy2() from a no-local_dir
    cache download, but copy2 needs BOTH the cache blob and the destination
    copy on disk simultaneously — for the largest file (16.7GB S2V model)
    that's a ~33GB peak just for one file, on top of whatever's already
    baked, and disk ran out (1h25min run, far past where build #7 died in
    3min, then failed).

    Fix v1 (build #9, c21daa0): shutil.move() instead of copy2(). FAILED in
    4.5min — on Linux, hf_hub_download's cache path is a SYMLINK with a
    RELATIVE target (e.g. snapshots/<rev>/file -> ../../blobs/<hash>).
    shutil.move's first attempt is os.rename(src, dst), which on a symlink
    renames the LINK ITSELF (doesn't follow it) — moving that relative
    symlink into a different directory makes its relative target resolve
    to the wrong place from the new location, producing a broken symlink.
    os.path.exists() on a broken symlink is False, which our own check
    correctly caught as FATAL (fast failure, no exception needed).

    Fix v2: os.path.realpath() the cache path FIRST, so we move the actual
    blob file (no symlink involved at all) instead of a relative symlink
    that breaks when relocated.
    """
    min_size_bytes = real_size_bytes * 0.95
    os.makedirs(dest_dir, exist_ok=True)
    free_before = _free_gb()
    print(f"[{file_role}] {repo}/{filename}  (disk free: {free_before:.1f} GB, "
          f"expecting {real_size_bytes/1e9:.3f} GB)")
    cache_path = os.path.realpath(hf_hub_download(repo_id=repo, filename=filename))
    link = os.path.join(dest_dir, canonical_name)
    if os.path.lexists(link):
        os.remove(link)
    shutil.move(cache_path, link)
    if not os.path.exists(link):
        raise SystemExit(
            f"FATAL [{file_role}]: move to {link} did not produce a file "
            f"(disk free was {free_before:.1f} GB before this download)"
        )
    size_bytes = os.path.getsize(link)
    size_gb = size_bytes / 1e9
    if size_bytes < min_size_bytes:
        raise SystemExit(
            f"FATAL [{file_role}]: {link} is {size_gb:.3f} GB, expected "
            f"~{real_size_bytes/1e9:.3f} GB (>= 95%). File is truncated/empty "
            f"— likely disk exhaustion mid-download (disk free was "
            f"{free_before:.1f} GB before this download started, now "
            f"{_free_gb():.1f} GB)."
        )
    # Clear leftover HF cache scaffolding (snapshot dirs, now-dangling
    # symlinks pointing at the moved blob) — harmless but tidy, and tiny
    # compared to the model files themselves.
    shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)
    print(f"   {size_gb:.3f} GB  (disk free now: {_free_gb():.1f} GB)")
    return link, size_gb


print("=== Downloading S2V models + T5 + VAE + Wav2Vec ===")
print(f"Disk free at start: {_free_gb():.1f} GB")
total_gb = 0.0
for repo, fname, dest, canon, role in S2V_PLAN:
    _, gb = dl_and_link(repo, fname, dest, canon, role, REAL_SIZES[(repo, fname)])
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

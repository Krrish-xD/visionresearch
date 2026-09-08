"""
Download Qwen2.5-VL-7B-Instruct from HuggingFace to models/qwen2.5-vl-7b-instruct.

Run from visionresearch/ root:
    python scripts/download_qwen.py
"""

import os
import sys

TARGET_DIR = "models/qwen2.5-vl-7b-instruct"
HF_REPO    = "Qwen/Qwen2.5-VL-7B-Instruct"


def main():
    if os.path.exists(TARGET_DIR):
        files = os.listdir(TARGET_DIR)
        safetensors = [f for f in files if f.endswith(".safetensors")]
        if safetensors:
            print(f"[download_qwen] Model already present at {TARGET_DIR} "
                  f"({len(safetensors)} safetensor shard(s)). Skipping download.")
            return
        else:
            print(f"[download_qwen] Directory exists but appears incomplete. Re-downloading.")

    print(f"[download_qwen] Downloading {HF_REPO} -> {TARGET_DIR}")
    print("[download_qwen] This is ~14 GB. Please be patient...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    os.makedirs(TARGET_DIR, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"[download_qwen] Download complete -> {TARGET_DIR}")


if __name__ == "__main__":
    main()

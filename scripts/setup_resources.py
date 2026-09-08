"""
Unified Resource Manager for VisionResearch.
Manages downloading, checking, and preparing benchmark datasets, evaluation splits, and model weights.
"""

import argparse
import sys
import os
import json
import subprocess
import shlex

# Ensure repo root is on sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from src.datasets.mmvp import prepare_mmvp_dataset
from src.datasets.clevr import generate_clevr_subset
from src.datasets.gqa import generate_gqa_subset
from src.evaluation.make_splits import make_all_splits


def load_json(rel_path: str) -> dict:
    """Load JSON file from root directory."""
    full_path = os.path.join(root_dir, rel_path)
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def list_resources():
    """Print complete status dashboard of datasets, splits, and models."""
    datasets_reg = load_json("datasets.json")
    models_reg = load_json("models.json")
    splits_dir = os.path.join(root_dir, "data", "splits")
    weights_dir = os.path.join(root_dir, "weights")

    print("\n" + "=" * 80)
    print("📊 VISIONRESEARCH RESOURCE STATUS DASHBOARD")
    print("=" * 80)

    # 1. Datasets
    print("\n[DATASETS]")
    print(f"{'Name':<12} {'Status':<12} {'Target File':<32} {'Description'}")
    print("-" * 80)
    for name, info in datasets_reg.items():
        rel_path = info.get("path", f"data/processed/{name}.jsonl")
        status = "✅ Ready" if os.path.exists(os.path.join(root_dir, rel_path)) else "❌ Missing"
        desc = info.get("description", "")
        print(f"{name:<12} {status:<12} {rel_path:<32} {desc[:30]}...")

    # 2. Splits
    print("\n[EVALUATION SPLITS]")
    print(f"{'Split File':<32} {'Status':<12}")
    print("-" * 80)
    for name in ["mmvp", "clevr", "gqa"]:
        split_file = f"{name}_splits.json"
        status = "✅ Ready" if os.path.exists(os.path.join(splits_dir, split_file)) else "❌ Missing"
        print(f"data/splits/{split_file:<20} {status:<12}")

    # 3. Models
    print("\n[MODELS (Local Weights)]")
    print(f"{'Model Key':<20} {'Status':<12} {'Hugging Face ID':<36} {'4-bit VRAM'}")
    print("-" * 80)
    for key, info in models_reg.items():
        local_dir = os.path.join(weights_dir, key)
        status = "✅ Ready" if (os.path.exists(local_dir) and os.listdir(local_dir)) else "❌ Not Downloaded"
        hf_id = info.get("hf_id", "")
        vram = f"~{info.get('vram_4bit_gb', 'N/A')} GB"
        print(f"{key:<20} {status:<12} {hf_id:<36} {vram}")

    print("=" * 80 + "\n")


def setup_datasets(target_dataset="all", force=False, clevr_size=1000, gqa_size=1000):
    """Check and prepare datasets."""
    datasets_reg = load_json("datasets.json")
    print("Checking and preparing benchmark datasets...")

    # 1. MMVP
    if target_dataset in ["all", "mmvp"]:
        target_path = datasets_reg.get("mmvp", {}).get("path", "data/processed/mmvp.jsonl")
        full_target = os.path.join(root_dir, target_path)
        if force or not os.path.exists(full_target):
            print(f"  -> Preparing MMVP Benchmark -> {target_path}...")
            try:
                items = prepare_mmvp_dataset(output_path=full_target, use_cache_if_exists=not force, save_images=True)
                print(f"     ✅ MMVP ready ({len(items)} items).")
            except Exception as e:
                print(f"     ⚠️ Error preparing MMVP: {e}")
        else:
            print(f"  -> MMVP already exists: {target_path}")

    # 2. CLEVR
    if target_dataset in ["all", "clevr"]:
        target_path = datasets_reg.get("clevr", {}).get("path", "data/processed/clevr.jsonl")
        full_target = os.path.join(root_dir, target_path)
        if force or not os.path.exists(full_target):
            print(f"  -> Generating CLEVR Grounding Subset ({clevr_size} items) -> {target_path}...")
            items = generate_clevr_subset(output_path=full_target, subset_size=clevr_size)
            print(f"     ✅ CLEVR ready ({len(items)} items).")
        else:
            print(f"  -> CLEVR already exists: {target_path}")

    # 3. GQA
    if target_dataset in ["all", "gqa"]:
        target_path = datasets_reg.get("gqa", {}).get("path", "data/processed/gqa.jsonl")
        full_target = os.path.join(root_dir, target_path)
        if force or not os.path.exists(full_target):
            print(f"  -> Generating GQA Relational Subset ({gqa_size} items) -> {target_path}...")
            items = generate_gqa_subset(output_path=full_target, subset_size=gqa_size)
            print(f"     ✅ GQA ready ({len(items)} items).")
        else:
            print(f"  -> GQA already exists: {target_path}")


def setup_splits(force=False):
    """Check and generate stratified evaluation splits."""
    splits_dir = os.path.join(root_dir, "data", "splits")
    required = ["mmvp_splits.json", "clevr_splits.json", "gqa_splits.json"]
    missing = [s for s in required if not os.path.exists(os.path.join(splits_dir, s))]

    if force or missing:
        print("Generating stratified evaluation splits (40% Cal / 60% Eval)...")
        try:
            make_all_splits(os.path.join(root_dir, "configs", "experiment.yaml"))
            print("  ✅ Stratified splits successfully generated.")
        except Exception as e:
            print(f"  ⚠️ Error generating splits: {e}")
    else:
        print("  -> Stratified splits already exist.")


def download_model(model_key: str):
    """Download model weights using huggingface-hub CLI or Python API."""
    models_reg = load_json("models.json")
    if model_key not in models_reg:
        print(f"❌ Unknown model '{model_key}'. Available models: {list(models_reg.keys())}")
        return

    info = models_reg[model_key]
    hf_id = info["hf_id"]
    local_dir = os.path.join(root_dir, "weights", model_key)
    os.makedirs(local_dir, exist_ok=True)

    print(f"\n-> Downloading {model_key} ({hf_id}) into weights/{model_key}...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=hf_id, local_dir=local_dir, local_dir_use_symlinks=False)
        print(f"✅ Successfully downloaded {model_key} to {local_dir}")
    except ImportError:
        # Fallback to huggingface-cli
        cmd = ["huggingface-cli", "download", hf_id, "--local-dir", local_dir]
        subprocess.run(cmd, check=True)
        print(f"✅ Successfully downloaded {model_key} to {local_dir}")
    except Exception as e:
        print(f"❌ Failed to download {model_key}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Unified Resource Manager for VisionResearch (Datasets, Splits, Models).")
    parser.add_argument("--list", action="store_true", help="List status of all datasets, splits, and models")
    parser.add_argument("--datasets", action="store_true", help="Prepare all datasets")
    parser.add_argument("--dataset", default="all", choices=["all", "mmvp", "clevr", "gqa"], help="Specific dataset to prepare")
    parser.add_argument("--splits", action="store_true", help="Generate stratified splits")
    parser.add_argument("--models", action="store_true", help="Download all configured models")
    parser.add_argument("--model", type=str, help="Specific model key to download (e.g. internvl3-8b, llava-onevision-7b)")
    parser.add_argument("--force", action="store_true", help="Force re-download / re-generation")
    parser.add_argument("--clevr-size", type=int, default=1000, help="Number of CLEVR subset items (default: 1000)")
    parser.add_argument("--gqa-size", type=int, default=1000, help="Number of GQA subset items (default: 1000)")

    args = parser.parse_args()

    # 1. List
    if args.list:
        list_resources()
        return

    # 2. Download specific or all models
    if args.model:
        download_model(args.model)
        return

    if args.models:
        models_reg = load_json("models.json")
        for m_key in models_reg.keys():
            download_model(m_key)
        return

    # 3. Specific dataset or splits
    if args.splits and not args.datasets:
        setup_splits(force=args.force)
        return

    # 4. Default: Auto-check and setup missing datasets and splits
    print("=" * 60)
    print("VisionResearch Auto-Resource Setup")
    print("=" * 60)
    setup_datasets(target_dataset=args.dataset, force=args.force, clevr_size=args.clevr_size, gqa_size=args.gqa_size)
    setup_splits(force=args.force)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

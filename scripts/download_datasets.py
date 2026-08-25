"""
Convenience CLI script to download and prepare datasets locally.
Allows team members to clone the lightweight repository and prepare
any dataset with a single command.
"""

import argparse
import sys
import os

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.mmvp import prepare_mmvp_dataset
from src.datasets.clevr import generate_clevr_subset
from src.datasets.gqa import generate_gqa_subset
from src.formalization.validators import validate_item


def main():
    parser = argparse.ArgumentParser(description="Download and prepare vision benchmark datasets locally.")
    parser.add_argument(
        "--dataset",
        choices=["all", "mmvp", "clevr", "gqa"],
        default="all",
        help="Which dataset to prepare (default: all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download / regeneration even if processed file exists"
    )
    parser.add_argument(
        "--clevr-size",
        type=int,
        default=1000,
        help="Number of CLEVR subset items to generate (default: 1000)"
    )
    parser.add_argument(
        "--gqa-size",
        type=int,
        default=1000,
        help="Number of GQA subset items to generate (default: 1000)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("VisionResearch Local Dataset Setup")
    print("=" * 60)

    # 1. MMVP
    if args.dataset in ["all", "mmvp"]:
        print("\n[1/3] Preparing MMVP Benchmark (300 Visual Pairs)...")
        try:
            items = prepare_mmvp_dataset(
                output_path="data/processed/mmvp.jsonl",
                use_cache_if_exists=not args.force,
                save_images=True
            )
            print(f"  -> MMVP ready: {len(items)} items.")
        except Exception as e:
            print(f"  -> Error preparing MMVP: {e}")

    # 2. CLEVR
    if args.dataset in ["all", "clevr"]:
        print(f"\n[2/3] Preparing CLEVR Grounding Subset ({args.clevr_size} items)...")
        items = generate_clevr_subset(
            output_path="data/processed/clevr.jsonl",
            subset_size=args.clevr_size
        )
        print(f"  -> CLEVR ready: {len(items)} items.")

    # 3. GQA
    if args.dataset in ["all", "gqa"]:
        print(f"\n[3/3] Preparing GQA Relational Subset ({args.gqa_size} items)...")
        items = generate_gqa_subset(
            output_path="data/processed/gqa.jsonl",
            subset_size=args.gqa_size
        )
        print(f"  -> GQA ready: {len(items)} items.")

    print("\nDataset preparation finished successfully!")

if __name__ == "__main__":
    main()

"""Unified dataset preparation script."""

import argparse
import yaml
import os
import sys

# Ensure repository root is on sys.path when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.datasets.mmvp import prepare_mmvp_dataset
from src.datasets.clevr import generate_clevr_subset
from src.datasets.gqa import generate_gqa_subset
from src.formalization.validators import validate_item


def prepare_all(config_path: str = "configs/experiment.yaml"):
    """Prepare all datasets specified in experiment configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg.get("datasets", {})
    total_valid = 0
    total_invalid = 0

    if ds_cfg.get("mmvp", {}).get("enabled", True):
        mmvp_path = ds_cfg.get("mmvp", {}).get("processed_path", "data/processed/mmvp.jsonl")
        items = prepare_mmvp_dataset(output_path=mmvp_path)
        for it in items:
            valid, msg = validate_item(it)
            if valid:
                total_valid += 1
            else:
                total_invalid += 1
                print(f"[Schema Error] MMVP item {it.get('item_id')}: {msg}")

    if ds_cfg.get("clevr", {}).get("enabled", True):
        clevr_path = ds_cfg.get("clevr", {}).get("processed_path", "data/processed/clevr.jsonl")
        size = ds_cfg.get("clevr", {}).get("subset_size", 1000)
        items = generate_clevr_subset(output_path=clevr_path, subset_size=size)
        for it in items:
            valid, msg = validate_item(it)
            if valid:
                total_valid += 1
            else:
                total_invalid += 1
                print(f"[Schema Error] CLEVR item {it.get('item_id')}: {msg}")

    if ds_cfg.get("gqa", {}).get("enabled", True):
        gqa_path = ds_cfg.get("gqa", {}).get("processed_path", "data/processed/gqa.jsonl")
        size = ds_cfg.get("gqa", {}).get("subset_size", 1000)
        items = generate_gqa_subset(output_path=gqa_path, subset_size=size)
        for it in items:
            valid, msg = validate_item(it)
            if valid:
                total_valid += 1
            else:
                total_invalid += 1
                print(f"[Schema Error] GQA item {it.get('item_id')}: {msg}")

    print(f"\nDataset preparation complete! Total valid items: {total_valid}, Invalid: {total_invalid}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare benchmark datasets")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to config YAML")
    args = parser.parse_args()
    prepare_all(args.config)

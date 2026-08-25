"""Deterministic Stratified Split Generator (40% Calibration / 60% Evaluation)."""

import os
import json
import random
import yaml
from collections import defaultdict
from typing import Dict, List, Any

def generate_stratified_splits(
    data_path: str,
    output_path: str,
    cal_ratio: float = 0.40,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Generate deterministic stratified calibration/evaluation item ID splits.
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    items = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    # Group by category
    by_category = defaultdict(list)
    for it in items:
        by_category[it["category"]].append(it["item_id"])

    cal_ids = []
    eval_ids = []

    for cat, ids in by_category.items():
        # Shuffle deterministically
        ids_shuffled = list(ids)
        random.shuffle(ids_shuffled)
        
        n_cal = int(round(len(ids_shuffled) * cal_ratio))
        # Ensure at least 1 calibration item per category if possible
        if n_cal == 0 and len(ids_shuffled) > 1:
            n_cal = 1

        cal_ids.extend(ids_shuffled[:n_cal])
        eval_ids.extend(ids_shuffled[n_cal:])

    # Sort for deterministic ordering
    cal_ids.sort()
    eval_ids.sort()

    split_dict = {
        "dataset_file": data_path,
        "seed": seed,
        "cal_ratio": cal_ratio,
        "eval_ratio": 1.0 - cal_ratio,
        "total_items": len(items),
        "calibration_count": len(cal_ids),
        "evaluation_count": len(eval_ids),
        "calibration_ids": cal_ids,
        "evaluation_ids": eval_ids
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2)

    print(f"Generated splits for {os.path.basename(data_path)}: {len(cal_ids)} Cal, {len(eval_ids)} Eval -> {output_path}")
    return split_dict

def make_all_splits(config_path: str = "configs/experiment.yaml"):
    """Create splits for all processed datasets in config."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("project", {}).get("seed", 42)
    cal_ratio = cfg.get("splits", {}).get("calibration_ratio", 0.40)
    splits_dir = cfg.get("splits", {}).get("splits_dir", "data/splits")

    for ds_name in ["mmvp", "clevr", "gqa"]:
        data_file = f"data/processed/{ds_name}.jsonl"
        out_file = os.path.join(splits_dir, f"{ds_name}_splits.json")
        if os.path.exists(data_file):
            generate_stratified_splits(data_file, out_file, cal_ratio=cal_ratio, seed=seed)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create calibration and evaluation splits")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to config YAML")
    args = parser.parse_args()
    make_all_splits(args.config)

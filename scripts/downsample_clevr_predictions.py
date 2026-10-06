"""
downsample_clevr_predictions.py
================================
Stratified downsampling of CLEVR raw-prediction JSONL files.

Key design choice
-----------------
Category is read directly from claim.predicate inside each prediction file.
No external dataset file is needed, so this works for every model regardless
of which dataset variant (original vs processed) it was evaluated on.

Predicate -> category mapping used:
  count            -> counting
  exists           -> existence
  attribute        -> attribute
  spatial_relation -> spatial_relation

Actual state of the data (as diagnosed):
  llava-1.5-7b  : 80,000 predictions, 3 predicates (count/exists/attribute)
  internvl3-8b  :  4,072 predictions, 3 predicates (count/exists/attribute)
  qwen2.5-vl-7b :  1,000 predictions, 3 predicates (count/exists/attribute)
                   Uses different item_ids (clevr_0001...) -- evaluated on
                   original CLEVR, NOT the processed dataset. That is fine;
                   we downsample it on its own data.

Output
------
Each model's JSONL is downsampled to N_PER_CATEGORY per predicate class.
Originals are backed up as *.jsonl.bak.
The CLEVR split files (data/splits/clevr_splits_<model>.json) are
regenerated per-model so the evaluation pipeline can run each model
independently on its own data.

Usage
-----
    python scripts/downsample_clevr_predictions.py               # 250/cat
    python scripts/downsample_clevr_predictions.py --n 333       # 999 total
    python scripts/downsample_clevr_predictions.py --dry-run
    python scripts/downsample_clevr_predictions.py --models llava-1.5-7b
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict

MODELS = ["llava-1.5-7b", "internvl3-8b", "qwen2.5-vl-7b"]
DATASET = "clevr"
PREDICTIONS_DIR = "results/raw_predictions"
SPLITS_DIR = "data/splits"
DEFAULT_N_PER_CATEGORY = 250
SEED = 42

# Map claim predicate -> canonical category name
PREDICATE_TO_CATEGORY = {
    "count":            "counting",
    "exists":           "existence",
    "attribute":        "attribute",
    "spatial_relation": "spatial_relation",
}


def get_category(pred: dict) -> str:
    """Extract category from a prediction's claim.predicate."""
    claim = pred.get("claim")
    if claim is None:
        return "unknown"
    predicate = claim.get("predicate", "unknown")
    return PREDICATE_TO_CATEGORY.get(predicate, predicate)


def downsample_file(model: str, pred_path: str, n_per_cat: int,
                    dry_run: bool) -> dict:
    """
    Load predictions, group by category (from claim.predicate),
    stratified-sample n_per_cat per category, backup original, overwrite.

    Returns summary dict {category: {before, sampled}}.
    """
    preds = []
    with open(pred_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                preds.append(json.loads(line))

    # Group by category derived from claim.predicate
    by_cat = defaultdict(list)
    for p in preds:
        by_cat[get_category(p)].append(p)

    # Drop unknowns
    unknown_count = len(by_cat.pop("unknown", []))
    if unknown_count:
        print("  WARNING: {} predictions have no claim/predicate - dropped.".format(
            unknown_count))

    # Stratified sample
    rng = random.Random(SEED)
    sampled = []
    summary = {}
    for cat in sorted(by_cat.keys()):
        items = list(by_cat[cat])
        rng.shuffle(items)
        take = min(n_per_cat, len(items))
        sampled.extend(items[:take])
        summary[cat] = {"before": len(items), "sampled": take}

    total_before = len(preds)
    total_after  = len(sampled)

    print("\n  Model : {}".format(model))
    print("  File  : {}".format(pred_path))
    print("  Total before : {:,}".format(total_before))
    for cat in sorted(summary.keys()):
        s = summary[cat]
        print("    {:<22}  {:6,}  ->  {}".format(cat, s["before"], s["sampled"]))
    print("  Total after  : {:,}".format(total_after))

    if dry_run:
        print("  [DRY RUN] No files written.")
        return summary

    # Backup original
    bak = pred_path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(pred_path, bak)
        print("  Backup -> {}".format(bak))
    else:
        print("  Backup already exists -> {} (not overwritten)".format(bak))

    # Overwrite with downsampled data sorted by item_id
    sampled.sort(key=lambda x: x["item_id"])
    with open(pred_path, "w", encoding="utf-8") as fh:
        for p in sampled:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("  DONE: {:,} records written -> {}".format(total_after, pred_path))
    return summary


def regenerate_splits(model: str, pred_path: str, dry_run: bool,
                      cal_ratio: float = 0.40) -> None:
    """
    Build a per-model clevr split file from the (already downsampled)
    prediction file, grouped by category, so the eval pipeline can run
    this model on its own data.

    Output: data/splits/clevr_splits_<model>.json
    The standard data/splits/clevr_splits.json is also updated for
    models whose item_ids match the processed dataset (llava, internvl).
    """
    preds = []
    with open(pred_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                preds.append(json.loads(line))

    by_cat = defaultdict(list)
    for p in preds:
        by_cat[get_category(p)].append(p["item_id"])

    rng = random.Random(SEED)
    cal_ids, eval_ids = [], []
    for cat in sorted(by_cat.keys()):
        ids = list(by_cat[cat])
        rng.shuffle(ids)
        n_cal = max(1, int(round(len(ids) * cal_ratio)))
        cal_ids.extend(ids[:n_cal])
        eval_ids.extend(ids[n_cal:])

    cal_ids.sort()
    eval_ids.sort()

    split_data = {
        "model": model,
        "dataset_file": pred_path,
        "seed": SEED,
        "cal_ratio": cal_ratio,
        "eval_ratio": 1.0 - cal_ratio,
        "total_items": len(preds),
        "calibration_count": len(cal_ids),
        "evaluation_count": len(eval_ids),
        "calibration_ids": cal_ids,
        "evaluation_ids": eval_ids,
    }

    out_path = os.path.join(SPLITS_DIR, "clevr_splits_{}.json".format(
        model.replace("/", "_").replace(".", "-")))

    if dry_run:
        print("  [DRY RUN] Would write split -> {} ({} cal / {} eval)".format(
            out_path, len(cal_ids), len(eval_ids)))
        return

    os.makedirs(SPLITS_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(split_data, fh, indent=2)
    print("  Split -> {} ({} cal / {} eval)".format(
        out_path, len(cal_ids), len(eval_ids)))

    # Also overwrite the shared clevr_splits.json for models that use
    # the processed dataset (llava / internvl whose IDs match)
    if model in ("llava-1.5-7b", "internvl3-8b"):
        shared_path = os.path.join(SPLITS_DIR, "clevr_splits.json")
        shared_data = dict(split_data)
        shared_data["model"] = "shared (llava/internvl)"
        shared_data["dataset_file"] = "data/processed/clevr.jsonl"
        bak = shared_path + ".bak"
        if not os.path.exists(bak) and os.path.exists(shared_path):
            shutil.copy2(shared_path, bak)
        with open(shared_path, "w", encoding="utf-8") as fh:
            json.dump(shared_data, fh, indent=2)
        print("  Shared split updated -> {}".format(shared_path))


def main():
    ap = argparse.ArgumentParser(
        description="Stratified CLEVR prediction downsampler (category from claim.predicate)")
    ap.add_argument("--n", type=int, default=DEFAULT_N_PER_CATEGORY,
                    help="Samples per predicate category (default 250; 3 cats => 750 total)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print statistics only; do not modify any files.")
    ap.add_argument("--models", nargs="+", default=MODELS,
                    help="Model keys to process (default: all three)")
    ap.add_argument("--no-splits", action="store_true",
                    help="Skip regenerating split files.")
    args = ap.parse_args()

    n_cats = 3  # count / exists / attribute (spatial_relation absent in all models)
    print("=" * 65)
    print("  CLEVR Stratified Downsampler")
    print("  Category source : claim.predicate (self-contained, no external data)")
    print("  Active categories: count / exists / attribute  ({} cats)".format(n_cats))
    print("  Target  : {} samples/category  ({} total)".format(args.n, args.n * n_cats))
    print("  Seed    : {}".format(SEED))
    print("  Mode    : {}".format("DRY RUN" if args.dry_run else "WRITE"))
    print("=" * 65)

    for model in args.models:
        pred_path = os.path.join(PREDICTIONS_DIR, "{}_clevr.jsonl".format(model))
        if not os.path.exists(pred_path):
            print("\n  WARNING: {} not found - skipping.".format(pred_path))
            continue

        downsample_file(model, pred_path, args.n, args.dry_run)

        if not args.no_splits:
            regenerate_splits(model, pred_path, args.dry_run)

    print("\n" + "=" * 65)
    if args.dry_run:
        print("  DRY RUN complete. No files were modified.")
    else:
        print("  Downsampling complete.")
        print("  Originals backed up as *.jsonl.bak")
        print("  Per-model split files written to data/splits/")
        print("")
        print("  NOTE: spatial_relation is ABSENT in all model predictions.")
        print("  All 3 models have only 3 categories: count/exists/attribute.")
        print("  This is a data collection gap, not a pipeline bug.")
    print("=" * 65)


if __name__ == "__main__":
    main()



"""
Full orchestration script: run inference + experiment pipeline for
InternVL3-8B and Qwen2.5-VL-7B on all three datasets.

Run from visionresearch/ root:
    python scripts/run_all_models.py

Stages:
  1. Smoke-test each model on 3 MMVP items
  2. Run full inference for each model × dataset
  3. Run experiment pipeline (calibration + solver + tables + figures)
  4. Print cross-model summary
"""

import os
import sys
import json
import traceback
from typing import List, Tuple

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PYTHON_EXEC = sys.executable
DATASETS    = ["mmvp", "clevr", "gqa"]
ALL_MODELS  = ["llava-1.5-7b", "internvl3-8b", "qwen2.5-vl-7b"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def section(title: str):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}\n")


def ok(msg: str):
    print(f"  [OK]   {msg}")


def warn(msg: str):
    print(f"  [WARN] {msg}")


def fail(msg: str):
    print(f"  [FAIL] {msg}")


# --------------------------------------------------------------------------
# Stage 0: Verify Models are ready
# --------------------------------------------------------------------------

def ensure_models():
    section("Stage 0: Verify Models are downloaded")
    qwen_dir = "models/qwen2.5-vl-7b-instruct"
    if os.path.exists(qwen_dir):
        shards = [f for f in os.listdir(qwen_dir) if f.endswith(".safetensors")]
        if shards:
            ok(f"Qwen2.5-VL-7B present ({len(shards)} shards)")
    
    intern_dir = "models/internvl3-8b"
    if os.path.exists(intern_dir):
        shards = [f for f in os.listdir(intern_dir) if f.endswith(".safetensors")]
        if shards:
            ok(f"InternVL3-8B present ({len(shards)} shards)")

    llava_dir = "models/llava-1.5-7b-hf"
    if os.path.exists(llava_dir):
        ok("LLaVA-1.5-7B present")

    return True


# --------------------------------------------------------------------------
# Stage 1: Smoke test
# --------------------------------------------------------------------------

def smoke_test_model(model_key: str, dataset: str = "mmvp", n: int = 3) -> bool:
    """Run inference on n items and verify answers are non-empty."""
    section(f"Stage 1: Smoke test — {model_key} on {n} {dataset} items")

    from src.vlm.evaluate import run_predictions_on_dataset

    tmp_dir = "results/smoke_tests"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_out = os.path.join(tmp_dir, f"{model_key}_{dataset}.jsonl")

    try:
        run_predictions_on_dataset(
            dataset_name=dataset,
            model_key=model_key,
            limit=n,
            mock=False,
            output_dir=tmp_dir,
        )
    except Exception as e:
        fail(f"Smoke test raised exception: {e}")
        traceback.print_exc()
        return False

    if not os.path.exists(tmp_out):
        fail(f"Output file not created: {tmp_out}")
        return False

    preds = []
    with open(tmp_out, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))

    empty = [p for p in preds if not p.get("raw_answer", "").strip()]
    failed_parse = [p for p in preds if p.get("parse_status") == "failed"]

    print(f"  Items tested:     {len(preds)}")
    print(f"  Empty answers:    {len(empty)}")
    print(f"  Parse failures:   {len(failed_parse)}")
    for p in preds:
        status = "OK" if p.get("raw_answer", "").strip() else "EMPTY"
        print(f"    [{status}] {p['item_id']} -> raw='{p['raw_answer'][:60]}' "
              f"conf={p.get('raw_confidence', 0):.3f} parse={p.get('parse_status')}")

    if len(empty) == len(preds):
        fail("ALL answers are empty — model inference is broken.")
        return False
    if len(empty) > len(preds) * 0.5:
        warn(f"{len(empty)}/{len(preds)} answers are empty — may indicate issues.")
    else:
        ok(f"Smoke test passed: {len(preds) - len(empty)}/{len(preds)} non-empty answers.")

    return True


# --------------------------------------------------------------------------
# Stage 2: Full inference
# --------------------------------------------------------------------------

def run_inference_for_model(model_key: str, force: bool = False):
    section(f"Stage 2: Full inference — {model_key}")
    from src.vlm.evaluate import run_predictions_on_dataset

    pred_dir = "results/raw_predictions"
    results = {}

    for ds in DATASETS:
        out_file = os.path.join(pred_dir, f"{model_key}_{ds}.jsonl")
        expected_len = 300 if ds == "mmvp" else 1000

        if not force and os.path.exists(out_file):
            valid = 0
            total = 0
            with open(out_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        total += 1
                        p = json.loads(line)
                        if p.get("raw_answer", "").strip():
                            valid += 1
            if total >= expected_len and valid / total >= 0.8:
                ok(f"{model_key} × {ds}: already exists with {valid}/{total} valid predictions. Skipping.")
                results[ds] = out_file
                continue
            else:
                warn(f"{model_key} × {ds}: existing file has {valid}/{total} valid answers. Re-running.")

        print(f"  Running inference: {model_key} × {ds} ...")
        try:
            run_predictions_on_dataset(
                dataset_name=ds,
                model_key=model_key,
                limit=None,
                mock=False,
                output_dir=pred_dir,
            )
            ok(f"{model_key} × {ds}: inference complete -> {out_file}")
            results[ds] = out_file
        except Exception as e:
            fail(f"{model_key} × {ds}: inference failed: {e}")
            traceback.print_exc()
            results[ds] = None

    return results


# --------------------------------------------------------------------------
# Stage 3: Experiment pipeline
# --------------------------------------------------------------------------

def run_experiment_for_model(model_key: str):
    section(f"Stage 3: Experiment pipeline — {model_key}")
    from src.evaluation.run_experiment import run_experiment_pipeline

    for ds in DATASETS:
        print(f"\n  Running pipeline: {model_key} × {ds} ...")
        try:
            run_experiment_pipeline(
                dataset_name=ds,
                model_key=model_key,
                config_path="configs/experiment.yaml",
                pilot=False,
                mock=False,
            )
            ok(f"{model_key} × {ds}: pipeline complete.")
        except Exception as e:
            fail(f"{model_key} × {ds}: pipeline failed: {e}")
            traceback.print_exc()


# --------------------------------------------------------------------------
# Stage 4: Cross-model summary
# --------------------------------------------------------------------------

def print_summary():
    section("Stage 4: Cross-Model Comparison Summary Across All Datasets")

    metrics_dir = "results/metrics"
    import pandas as pd
    import glob

    master_rows = []

    for ds in DATASETS:
        ds_rows = []
        for model_key in ALL_MODELS:
            t3_file = os.path.join(metrics_dir, f"{model_key}_{ds}_table3_contradiction_metrics.csv")
            t2_file = os.path.join(metrics_dir, f"{model_key}_{ds}_table2_calibration_metrics.csv")

            if os.path.exists(t3_file) and os.path.exists(t2_file):
                t3_df = pd.read_csv(t3_file)
                t2_df = pd.read_csv(t2_file)
                
                # Extract ECEs
                raw_ece = t2_df.loc[t2_df["Method"].str.contains("Raw", case=False), "ECE (10-bin)"].values
                iso_ece = t2_df.loc[t2_df["Method"].str.contains("Isotonic", case=False), "ECE (10-bin)"].values
                
                # Extract Hard SMT / Raw / Isotonic SFAR
                hard_row = t3_df.loc[t3_df["Condition"].str.contains("Hard", case=False)]
                raw_row = t3_df.loc[t3_df["Condition"].str.contains("Raw", case=False)]
                iso_row = t3_df.loc[t3_df["Condition"].str.contains("Isotonic", case=False)]

                accuracy = hard_row["Accuracy"].values[0] if len(hard_row) > 0 else "N/A"
                hard_sfar = hard_row["SFAR"].values[0] if len(hard_row) > 0 else "N/A"
                raw_sfar = raw_row["SFAR"].values[0] if len(raw_row) > 0 else "N/A"
                iso_sfar = iso_row["SFAR"].values[0] if len(iso_row) > 0 else "N/A"
                iso_f1 = iso_row["F1 (95% CI)"].values[0] if len(iso_row) > 0 else "N/A"

                row_dict = {
                    "Dataset": ds.upper(),
                    "Model": model_key,
                    "Accuracy": accuracy,
                    "Raw ECE": raw_ece[0] if len(raw_ece) > 0 else "N/A",
                    "Isotonic ECE": iso_ece[0] if len(iso_ece) > 0 else "N/A",
                    "Hard SMT SFAR": hard_sfar,
                    "Raw SFAR": raw_sfar,
                    "Isotonic SFAR": iso_sfar,
                    "Isotonic F1 (95% CI)": iso_f1
                }
                ds_rows.append(row_dict)
                master_rows.append(row_dict)

        if ds_rows:
            ds_df = pd.DataFrame(ds_rows)
            print(f"\n--- {ds.upper()} Benchmark Results ---")
            print(ds_df.drop(columns=["Dataset"]).to_string(index=False))
            ds_csv = os.path.join(metrics_dir, f"cross_model_{ds}_summary.csv")
            ds_df.to_csv(ds_csv, index=False)
            ok(f"Saved {ds.upper()} summary -> {ds_csv}")

    if master_rows:
        master_df = pd.DataFrame(master_rows)
        print("\n" + "=" * 90)
        print("  MASTER MULTI-MODEL MULTI-DATASET BENCHMARK RESULTS")
        print("=" * 90)
        print(master_df.to_string(index=False))
        print("=" * 90 + "\n")
        
        master_csv = os.path.join(metrics_dir, "master_all_models_all_datasets_summary.csv")
        master_df.to_csv(master_csv, index=False)
        ok(f"Saved Master Summary Table -> {master_csv}")

    print("\n  Full metric tables: results/metrics/")
    print("  Publication figures: results/figures/")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  VisionResearch — Multi-Model Experiment Orchestrator")
    print("=" * 60)

    # Stage 0: Verify models
    ensure_models()

    smoke_results = {}

    for model_key in ALL_MODELS:
        # Stage 1: Smoke test (skip for llava if already cached)
        if model_key != "llava-1.5-7b":
            passed = smoke_test_model(model_key, dataset="mmvp", n=3)
            smoke_results[model_key] = passed
            if not passed:
                fail(f"Smoke test failed for {model_key}. Skipping full run.")
                continue

        # Stage 2: Full inference
        run_inference_for_model(model_key, force=False)

        # Stage 3: Experiment pipeline
        run_experiment_for_model(model_key)

    # Stage 4: Cross-model summary
    print_summary()

    section("Multi-Model Orchestration Complete")


if __name__ == "__main__":
    main()

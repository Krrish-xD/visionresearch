"""
Verification and Test Script for Batched & Sequential VLM Inference Pipeline.

Tests the pipeline through 1,000 images (or user-specified limit):
  1. Batched forward passes (5-15 images per batch)
  2. Sequential model switching with full VRAM cleanup
  3. Live throughput (items/sec) and GPU VRAM tracking
  4. Logging to both console and a dedicated audit log file
  5. Assertion checks on output predictions (JSONL validity, non-empty answers, confidence bounds)

Usage:
    # Run 1000 images with batch size 10 in real mode:
    python scripts/test_batched_pipeline.py --dataset gqa --limit 1000 --batch-size 10

    # Test the pipeline logic immediately in mock simulation mode:
    python scripts/test_batched_pipeline.py --dataset gqa --limit 1000 --batch-size 10 --mock
"""

import os
import sys
import json
import time
import argparse
import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception as e:
        print(f"[Warning] Failed to reconfigure stdout/stderr to UTF-8: {e}", file=sys.stderr)

# Ensure project root is on path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import torch
import collections
import numpy as np
from src.vlm.evaluate import VLMInferenceRunner, run_sequential_models
from src.formalization.validators import validate_prediction
from src.vlm.prompts import format_prompt
from src.vlm.confidence import extract_token_confidence
from src.formalization.parser import parse_vlm_answer_to_claim, normalize_text
from src.calibration.metrics import compute_ece, compute_brier_score

DEFAULT_MODELS = ["qwen2.5-vl-7b", "internvl3-8b", "llava-1.5-7b"]


def _resolve_and_load_image(item: dict):
    """Prefetch and decode a PIL RGB image from disk on background thread."""
    img_path = item.get("image_path")
    if not img_path:
        return None
    if not os.path.isabs(img_path):
        img_path = os.path.join(ROOT_DIR, img_path)
    if not os.path.exists(img_path):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        dir_name = os.path.dirname(img_path)
        for alt_name in [f"{stem}.jpg", f"{int(stem) if stem.isdigit() else stem}.jpg", f"{stem}.png"]:
            cand = os.path.join(dir_name, alt_name)
            if os.path.exists(cand):
                img_path = cand
                break
    if os.path.exists(img_path):
        try:
            with Image.open(img_path) as img:
                return img.convert("RGB")
        except Exception:
            return None
    return None


def _classify_error(ans_type: str, norm_gold: str, norm_pred: str, is_correct: bool, parse_status: str):
    """Classify failure modes for instantaneous zero-pandas error taxonomy."""
    if is_correct:
        return "none", None
    if parse_status != "success":
        return "parser_failure", "Failed to parse model generation into clean claim"
    if ans_type == "yes_no":
        if norm_gold == "no" and (norm_pred == "yes" or "yes" in norm_pred):
            return "hallucination", "False positive: asserted non-existent object or relation"
        else:
            return "missed_detection", "False negative: failed to confirm valid object or relation"
    if ans_type == "count":
        try:
            delta = int(norm_pred) - int(norm_gold)
            direction = "overcount" if delta > 0 else "undercount"
            return "count_discrepancy", f"Off-by-{abs(delta)} ({direction})"
        except Exception:
            return "count_discrepancy", f"Non-numeric count predicted: '{norm_pred}'"
    if ans_type in ["attribute", "relation", "choice"]:
        return "attribute_mismatch", f"Predicted '{norm_pred}' instead of '{norm_gold}'"
    return "other_mismatch", f"Predicted '{norm_pred}' instead of '{norm_gold}'"


def _run_batch_fast(runner, batch_items: list, pil_images: list, max_tokens: int = 4):
    """Execute forward pass with preloaded images and fast constrained tokens."""
    if runner.engine.model is None:
        return [runner.run_inference_on_item(it) for it in batch_items]

    prompts = [
        format_prompt(item["answer_type"], item["question"], item.get("options", ""))
        for item in batch_items
    ]
    batch_results = runner.engine.generate_batch_with_logprobs(
        pil_images, prompts, temperature=0.0, max_tokens=max_tokens
    )
    records = []
    for b, (item, res) in enumerate(zip(batch_items, batch_results)):
        raw_ans = res["full_text"]
        outputs = res["outputs"]
        prompt_len = res["prompt_len"]
        batch_idx = res.get("batch_idx", b)

        raw_conf, logprobs, ans_logits, _ = extract_token_confidence(
            outputs.scores, res["generated_ids"], prompt_len=prompt_len, batch_idx=batch_idx
        )
        claim, norm_ans, parse_status = parse_vlm_answer_to_claim(
            raw_ans, item["answer_type"], question=item["question"],
            gold_facts=item.get("gold_facts", []), options=item.get("options", "")
        )
        norm_gold = normalize_text(item["gold_answer"])
        norm_pred = normalize_text(norm_ans)
        is_correct = (norm_gold == norm_pred) or (norm_gold in norm_pred)

        error_type, error_detail = _classify_error(
            item.get("answer_type", ""), norm_gold, norm_pred, bool(is_correct), parse_status
        )

        conf_val = float(raw_conf)
        bin_start = int(conf_val * 10) / 10.0
        bin_end = min(bin_start + 0.10, 1.0)
        conf_bin = f"{bin_start:.2f}-{bin_end:.2f}"

        records.append({
            "item_id": item["item_id"],
            "model": runner.model_key,
            "category": item.get("category", "unknown"),
            "answer_type": item.get("answer_type", "unknown"),
            "question": item.get("question", ""),
            "image_path": item.get("image_path", ""),
            "gold_answer": item.get("gold_answer", ""),
            "gold_facts": item.get("gold_facts", []),

            "prompt_id": "constrained_v1",
            "raw_answer": raw_ans,
            "normalized_answer": norm_ans,
            "claim": claim,
            "parse_status": parse_status,

            "is_correct": bool(is_correct),
            "error_type": error_type,
            "error_detail": error_detail,

            "raw_confidence": conf_val,
            "confidence_bin": conf_bin,
            "token_logprobs": logprobs,
            "answer_logits": ans_logits,
            "tokens": res.get("tokens", [])
        })
    return records


class Logger:
    """Logs simultaneously to console and an audit file with timestamps."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # Clear / initialize log file
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"=== Batched VLM Pipeline Audit Log started at {datetime.datetime.now()} ===\n\n")

    def log(self, msg: str = ""):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {msg}" if msg else ""
        try:
            print(formatted)
        except UnicodeEncodeError as ue:
            try:
                print(f"[Warning: console encoding fallback: {ue}] {formatted.encode('ascii', errors='replace').decode('ascii')}")
            except Exception as inner_err:
                print(f"[Error] Failed to print to console: {inner_err}", file=sys.stderr)
        except Exception as e:
            print(f"[Error] Failed to print log message: {e}", file=sys.stderr)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")


def get_vram_status() -> Dict[str, float]:
    """Return GPU memory stats in GB."""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "total": 0.0}
    return {
        "allocated": torch.cuda.memory_allocated() / 1e9,
        "reserved": torch.cuda.memory_reserved() / 1e9,
        "total": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }


def audit_prediction_file(pred_file: str, expected_count: int, logger: Logger) -> Dict[str, Any]:
    """Verify that predictions generated by a model are mathematically and structurally sound."""
    if not os.path.exists(pred_file):
        logger.log(f"❌ [AUDIT FAILED] Prediction file not found: {pred_file}")
        return {"passed": False, "count": 0}

    records = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.log(f"❌ [AUDIT FAILED] Corrupt JSON on line {idx} in {pred_file}: {e}")
                    return {"passed": False, "count": len(records)}

    total = len(records)
    empty_answers = 0
    invalid_schemas = 0
    confidence_out_of_bounds = 0
    correct_count = 0

    for r in records:
        # Schema validation
        valid, msg = validate_prediction(r)
        if not valid:
            invalid_schemas += 1

        # Answer check
        ans = r.get("raw_answer", "")
        if not str(ans).strip():
            empty_answers += 1

        # Confidence bounds [0.0, 1.0]
        conf = r.get("raw_confidence", -1.0)
        if conf < 0.0 or conf > 1.0:
            confidence_out_of_bounds += 1

        if r.get("is_correct", False):
            correct_count += 1

    accuracy = (correct_count / total * 100.0) if total > 0 else 0.0
    logger.log(f"  -> Total Records:           {total} (target: {expected_count})")
    logger.log(f"  -> Schema Errors:            {invalid_schemas}")
    logger.log(f"  -> Empty Answers:            {empty_answers}")
    logger.log(f"  -> Out-of-bounds Confidence: {confidence_out_of_bounds}")
    logger.log(f"  -> Accuracy on Ground Truth: {accuracy:.2f}% ({correct_count}/{total})")

    passed = (total >= expected_count) and (empty_answers == 0) and (invalid_schemas == 0) and (confidence_out_of_bounds == 0)
    return {"passed": passed, "count": total, "accuracy": accuracy}


def generate_model_summary_and_slices(
    pred_file: str,
    model_key: str,
    dataset: str,
    output_dir: str,
    inf_throughput: float,
    load_time: float,
    inf_time: float,
    logger: Logger
) -> Dict[str, Any]:
    """Generates an instant, pre-aggregated diagnostic JSON dashboard and category slices without pandas."""
    if not os.path.exists(pred_file):
        return {}

    records = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    total = len(records)
    if total == 0:
        return {}

    confs = [float(r.get("raw_confidence", 0.0)) for r in records]
    labels = [bool(r.get("is_correct", False)) for r in records]
    correct_count = sum(1 for l in labels if l)

    overall_acc = (correct_count / total) * 100.0
    overall_ece, bin_data = compute_ece(confs, labels, num_bins=10)
    overall_brier = compute_brier_score(confs, labels)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    overconfidence_gap = float(mean_conf - (overall_acc / 100.0))

    # Category breakdown & Slicing
    categories = sorted(list(set(r.get("category", "unknown") for r in records)))
    category_breakdown = {}
    slices_dir = os.path.join(output_dir, "slices")

    for cat in categories:
        cat_recs = [r for r in records if r.get("category") == cat]
        cat_total = len(cat_recs)
        cat_correct = sum(1 for r in cat_recs if r.get("is_correct"))
        cat_acc = (cat_correct / cat_total * 100.0) if cat_total > 0 else 0.0
        cat_confs = [float(r.get("raw_confidence", 0.0)) for r in cat_recs]
        cat_labels = [bool(r.get("is_correct", False)) for r in cat_recs]
        cat_ece, _ = compute_ece(cat_confs, cat_labels, num_bins=10) if cat_total > 0 else (0.0, {})
        cat_errors = dict(collections.Counter(r.get("error_type", "unknown") for r in cat_recs))

        category_breakdown[cat] = {
            "total_items": cat_total,
            "correct_items": cat_correct,
            "accuracy_pct": round(cat_acc, 2),
            "mean_confidence": round(float(np.mean(cat_confs)), 4) if cat_confs else 0.0,
            "ece": round(cat_ece, 4),
            "error_breakdown": cat_errors
        }

        # Save sliced JSONL
        cat_slice_dir = os.path.join(slices_dir, cat)
        os.makedirs(cat_slice_dir, exist_ok=True)
        cat_slice_file = os.path.join(cat_slice_dir, f"{model_key}.jsonl")
        with open(cat_slice_file, "w", encoding="utf-8") as f_slice:
            for r in cat_recs:
                f_slice.write(json.dumps(r) + "\n")

    overall_errors = dict(collections.Counter(r.get("error_type", "unknown") for r in records))

    summary = {
        "model": model_key,
        "dataset": dataset,
        "timestamp": datetime.datetime.now().isoformat(),
        "total_items": total,
        "correct_items": correct_count,
        "overall_accuracy_pct": round(overall_acc, 2),
        "overall_ece": round(overall_ece, 4),
        "overall_brier_score": round(overall_brier, 4),
        "mean_confidence": round(mean_conf, 4),
        "overconfidence_gap": round(overconfidence_gap, 4),
        "category_breakdown": category_breakdown,
        "error_distribution": overall_errors,
        "confidence_bins": bin_data,
        "throughput": {
            "avg_img_per_sec": round(inf_throughput, 2),
            "cold_load_time_sec": round(load_time, 2),
            "active_inference_sec": round(inf_time, 2)
        }
    }

    # Save summary JSON
    summaries_dir = os.path.join(output_dir, "summaries")
    os.makedirs(summaries_dir, exist_ok=True)
    summary_path = os.path.join(summaries_dir, f"{model_key}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)

    logger.log(f"  📊 Zero-Pandas Summary Dashboard -> {summary_path}")
    logger.log(f"  📂 Slices partitioned ({len(categories)} categories) -> {slices_dir}/")
    return summary


def generate_cross_model_comparison(output_dir: str, completed_models: List[str], dataset: str, logger: Logger):
    """Generate consolidated leaderboard, category matrix, and disagreement dataset for SMT solver."""
    summaries_dir = os.path.join(output_dir, "summaries")
    comparisons_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comparisons_dir, exist_ok=True)

    loaded_summaries = {}
    for m in completed_models:
        s_path = os.path.join(summaries_dir, f"{m}_summary.json")
        if os.path.exists(s_path):
            try:
                with open(s_path, "r", encoding="utf-8") as f:
                    loaded_summaries[m] = json.load(f)
            except Exception:
                pass

    if not loaded_summaries:
        return

    # Leaderboard table
    leaderboard = []
    for m, s in loaded_summaries.items():
        leaderboard.append({
            "model": m,
            "accuracy_pct": s.get("overall_accuracy_pct", 0.0),
            "ece": s.get("overall_ece", 0.0),
            "brier_score": s.get("overall_brier_score", 0.0),
            "mean_confidence": s.get("mean_confidence", 0.0),
            "throughput_img_s": s.get("throughput", {}).get("avg_img_per_sec", 0.0),
            "categories": {cat: info.get("accuracy_pct", 0.0) for cat, info in s.get("category_breakdown", {}).items()}
        })

    leaderboard.sort(key=lambda x: x["accuracy_pct"], reverse=True)

    lb_path = os.path.join(comparisons_dir, "leaderboard.json")
    with open(lb_path, "w", encoding="utf-8") as f:
        json.dump({"leaderboard": leaderboard}, f, indent=2)
    logger.log(f"  🏆 Cross-Model Leaderboard generated -> {lb_path}")

    # If 2+ models exist, identify disagreements for SMT solver
    if len(completed_models) >= 2:
        model_preds = {}
        for m in completed_models:
            p_path = os.path.join(output_dir, f"{m}_{dataset}.jsonl")
            if not os.path.exists(p_path):
                for f in os.listdir(output_dir):
                    if f.startswith(f"{m}_") and f.endswith(".jsonl"):
                        p_path = os.path.join(output_dir, f)
                        break
            if os.path.exists(p_path):
                m_map = {}
                with open(p_path, "r", encoding="utf-8") as pf:
                    for line in pf:
                        if line.strip():
                            try:
                                d = json.loads(line)
                                m_map[d["item_id"]] = d
                            except Exception:
                                pass
                model_preds[m] = m_map

        # Find overlapping items
        common_ids = set()
        for idx, m in enumerate(completed_models):
            if m in model_preds:
                if idx == 0:
                    common_ids = set(model_preds[m].keys())
                else:
                    common_ids &= set(model_preds[m].keys())

        disagreements = []
        agreed_count = 0
        for item_id in sorted(list(common_ids)):
            answers = {m: model_preds[m][item_id].get("normalized_answer", "") for m in completed_models}
            unique_answers = set(answers.values())
            sample_rec = model_preds[completed_models[0]][item_id]
            if len(unique_answers) > 1:
                disagreements.append({
                    "item_id": item_id,
                    "category": sample_rec.get("category", ""),
                    "question": sample_rec.get("question", ""),
                    "gold_answer": sample_rec.get("gold_answer", ""),
                    "image_path": sample_rec.get("image_path", ""),
                    "predictions": answers,
                    "confidences": {m: model_preds[m][item_id].get("raw_confidence", 0.0) for m in completed_models}
                })
            else:
                agreed_count += 1

        disagree_path = os.path.join(comparisons_dir, "disagreements.jsonl")
        with open(disagree_path, "w", encoding="utf-8") as f_dis:
            for dis in disagreements:
                f_dis.write(json.dumps(dis) + "\n")

        total_compared = len(common_ids)
        consensus_rate = (agreed_count / max(total_compared, 1)) * 100.0
        consensus_path = os.path.join(comparisons_dir, "consensus_summary.json")
        with open(consensus_path, "w", encoding="utf-8") as f_con:
            json.dump({
                "compared_models": completed_models,
                "total_common_items": total_compared,
                "agreed_items": agreed_count,
                "disagreed_items": len(disagreements),
                "consensus_rate_pct": round(consensus_rate, 2),
                "disagreements_file": disagree_path
            }, f_con, indent=2)

        logger.log(f"  ⚔️ Model Disagreements isolated ({len(disagreements)} items) -> {disagree_path}")


def run_pipeline_test(
    models: List[str],
    dataset: str = "gqa",
    limit: int = 1000,
    batch_size: int = 10,
    mock: bool = False,
    force: bool = False,
    dynamic_batch: bool = False,
    output_dir: str = "results/test_batched_predictions",
    log_file: str = "results/test_batched_pipeline.log"
):
    logger = Logger(log_file)
    logger.log("=" * 80)
    logger.log("🧪 BATCHED & SEQUENTIAL VLM PIPELINE VERIFICATION TEST")
    logger.log("=" * 80)
    logger.log(f"Dataset:       {dataset} (Testing up to {limit} items)")
    logger.log(f"Models:        {models}")
    batch_desc = "DYNAMIC AUTO-SIZE (Saturates VRAM)" if (dynamic_batch or batch_size <= 0) else f"{batch_size} images per forward pass"
    logger.log(f"Batch Mode:    {batch_desc}")
    logger.log(f"Mode:          {'SIMULATION / MOCK' if mock else 'REAL VRAM INFERENCE'}")
    logger.log(f"Output Dir:    {output_dir}")
    logger.log(f"Log File:      {log_file}")

    vram = get_vram_status()
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.log(f"Hardware:      {gpu_name} ({vram['total']:.2f} GB VRAM available)")
    else:
        logger.log("Hardware:      CPU only (CUDA not available)")
    logger.log("=" * 80 + "\n")

    # Step 1: Ensure dataset is available
    data_file = os.path.join(ROOT_DIR, "data", "processed", f"{dataset}.jsonl")
    if not os.path.exists(data_file):
        if dataset == "gqa_curated_2500":
            logger.log(f"Dataset {data_file} not found. Auto-generating curated 2,500 balanced subset...")
            from scripts.generate_curated_subset import generate_curated_2500
            try:
                generate_curated_2500(output_path=data_file, per_category_quota=625)
            except Exception as e:
                logger.log(f"❌ Failed to generate curated subset: {e}")
                return False
        else:
            logger.log(f"Dataset {data_file} not found. Preparing via prepare.py...")
            from src.datasets.prepare import prepare_all
            try:
                prepare_all(os.path.join(ROOT_DIR, "configs", "experiment.yaml"))
            except Exception as e:
                logger.log(f"❌ Failed to prepare dataset: {e}")
                return False

    # Count total available items
    available_items = 0
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                available_items += 1

    test_limit = min(limit, available_items) if available_items > 0 else limit
    logger.log(f"Available dataset items in {data_file}: {available_items}")
    logger.log(f"Items to process in this test: {test_limit}\n")

    overall_start = time.time()
    audit_summary = {}

    # Step 2: Run sequential batched inference
    for idx, model_key in enumerate(models, 1):
        logger.log("-" * 80)
        logger.log(f"[{idx}/{len(models)}] Starting Model: {model_key}")
        pred_file = os.path.join(output_dir, f"{model_key}_{dataset}.jsonl")
        if force and os.path.exists(pred_file):
            logger.log(f"  [--force] Removing previous predictions: {pred_file}")
            os.remove(pred_file)

        vram_before = get_vram_status()
        logger.log(f"  VRAM before load: {vram_before['allocated']:.2f} GB allocated / {vram_before['reserved']:.2f} GB reserved")

        # Load dataset items
        items = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
                    if len(items) >= test_limit:
                        break

        # Resume support: skip already processed item IDs
        completed_ids = set()
        if not force and os.path.exists(pred_file):
            with open(pred_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            completed_ids.add(json.loads(line).get("item_id"))
                        except Exception:
                            pass
            if completed_ids:
                logger.log(f"  Resuming: found {len(completed_ids)} existing predictions.")

        remaining_items = [it for it in items if it["item_id"] not in completed_ids]
        if not remaining_items:
            logger.log(f"  All {len(items)} items already processed for {dataset} -> {pred_file}")
            audit_res = audit_prediction_file(pred_file, test_limit, logger)
            audit_summary[model_key] = audit_res
            continue

        # Step A: Load model into VRAM
        t_load_start = time.time()
        runner = VLMInferenceRunner(model_key=model_key)
        if not mock:
            try:
                runner.load_model()
            except Exception as e:
                logger.log(f"❌ [EXCEPTION] Pipeline failed on model {model_key}: {e}")
                import traceback
                logger.log(traceback.format_exc())
                audit_summary[model_key] = {"passed": False, "error": str(e)}
                continue
        load_time = time.time() - t_load_start

        # Step B: Live continuous progress bar across all images
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        pbar = None
        if tqdm is not None:
            pbar = tqdm(
                total=len(remaining_items),
                desc=f"[{model_key}]",
                unit="img",
                dynamic_ncols=True,
                leave=True
            )

        # Step C: Dynamic VRAM auto-queue & batch execution
        current_bs = batch_size if (batch_size > 0 and not dynamic_batch) else 24
        learned_max_bs = 48
        inf_start = time.time()
        processed_count = 0
        idx_cursor = 0

        os.makedirs(output_dir, exist_ok=True)
        with ThreadPoolExecutor(max_workers=2) as executor, open(pred_file, "a", encoding="utf-8") as f_out:
            def _load_slice(slice_items):
                return [_resolve_and_load_image(it) for it in slice_items]

            initial_step = min(current_bs, len(remaining_items) - idx_cursor)
            pref_future = executor.submit(_load_slice, remaining_items[idx_cursor:idx_cursor + initial_step])

            while idx_cursor < len(remaining_items):
                batch_images = pref_future.result()
                step = len(batch_images)
                batch = remaining_items[idx_cursor:idx_cursor + step]

                # Concurrently submit background prefetch for the NEXT batch while GPU works
                next_cursor = idx_cursor + step
                if next_cursor < len(remaining_items):
                    if (dynamic_batch or batch_size <= 0) and torch.cuda.is_available():
                        free_b, total_b = torch.cuda.mem_get_info()
                        used_b = total_b - free_b
                        target_limit_b = total_b * 0.90
                        headroom_b = target_limit_b - used_b
                        if headroom_b > 2.0e9:
                            current_bs = min(current_bs + 4, learned_max_bs)
                        elif headroom_b < 1.0e9 and current_bs > 4:
                            current_bs = max(current_bs - 4, 4)
                        next_step = min(current_bs, len(remaining_items) - next_cursor)
                    else:
                        next_step = min(batch_size if batch_size > 0 else 10, len(remaining_items) - next_cursor)
                    pref_future = executor.submit(_load_slice, remaining_items[next_cursor:next_cursor + next_step])

                try:
                    batch_preds = _run_batch_fast(runner, batch, batch_images, max_tokens=4)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
                    if "out of memory" in str(oom_err).lower() or isinstance(oom_err, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                        learned_max_bs = max(len(batch) - 4, 4)
                        current_bs = max(len(batch) // 2, 4)
                        throttle_msg = f"[Autoscaler] VRAM full at batch {len(batch)} -> throttled to {current_bs}"
                        if pbar is not None:
                            pbar.write(throttle_msg)
                        else:
                            print(throttle_msg)
                        half = max(len(batch) // 2, 1)
                        batch = batch[:half]
                        batch_images = batch_images[:half]
                        batch_preds = _run_batch_fast(runner, batch, batch_images, max_tokens=4)
                    else:
                        raise oom_err

                for pred in batch_preds:
                    f_out.write(json.dumps(pred) + "\n")
                    f_out.flush()

                idx_cursor += len(batch)
                processed_count += len(batch)
                elapsed = time.time() - inf_start
                rate = processed_count / max(elapsed, 0.001)

                if pbar is not None:
                    pbar.update(len(batch))
                    pbar_info = {"speed": f"{rate:.2f} img/s"}
                    if (dynamic_batch or batch_size <= 0):
                        pbar_info["bs"] = len(batch)
                    if torch.cuda.is_available():
                        alloc = torch.cuda.memory_allocated() / 1e9
                        pbar_info["VRAM"] = f"{alloc:.1f}GB"
                    pbar.set_postfix(pbar_info)
                else:
                    pct = (processed_count / len(remaining_items)) * 100.0
                    vram_info = ""
                    if torch.cuda.is_available():
                        alloc = torch.cuda.memory_allocated() / 1e9
                        res = torch.cuda.memory_reserved() / 1e9
                        vram_info = f" | VRAM: {alloc:.2f} GB alloc / {res:.2f} GB res"
                    bs_info = f" (bs={len(batch)})" if (dynamic_batch or batch_size <= 0) else ""
                    print(f"  [{model_key}] [{processed_count}/{len(remaining_items)}] ({pct:.1f}%){bs_info} | {rate:.1f} items/sec{vram_info}")

        if pbar is not None:
            pbar.close()

        # Step D: Unload model and free GPU VRAM
        runner.engine.unload_model()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inf_time = time.time() - inf_start
        model_elapsed = load_time + inf_time
        inf_throughput = processed_count / max(inf_time, 0.001)
        overall_throughput = processed_count / max(model_elapsed, 0.001)

        vram_after = get_vram_status()
        logger.log(f"  Model weights load time: {load_time:.2f}s (one-time cold start)")
        logger.log(f"  Active inference time:   {inf_time:.2f}s")
        logger.log(f"  ⚡ Pure inference speed:   {inf_throughput:.2f} images/sec ({1.0/max(inf_throughput, 0.001):.2f}s per image)")
        logger.log(f"  Overall pipeline speed:  {overall_throughput:.2f} images/sec (including load time)")
        logger.log(f"  VRAM after unload:       {vram_after['allocated']:.2f} GB allocated (Should be ~0.0 GB)")

        # Verify output file
        logger.log(f"  Auditing output predictions: {pred_file}...")
        audit_res = audit_prediction_file(pred_file, test_limit, logger)
        audit_res["inf_throughput"] = inf_throughput
        audit_res["overall_throughput"] = overall_throughput
        audit_res["load_time"] = load_time
        audit_res["inf_time"] = inf_time
        audit_res["runtime_sec"] = model_elapsed

        # Generate zero-pandas summary dashboard and category slices
        summary_res = generate_model_summary_and_slices(
            pred_file, model_key, dataset, output_dir,
            inf_throughput, load_time, inf_time, logger
        )
        audit_res["summary"] = summary_res
        audit_summary[model_key] = audit_res

    overall_elapsed = time.time() - overall_start

    # Generate cross-model leaderboard and disagreement isolation
    generate_cross_model_comparison(output_dir, list(audit_summary.keys()), dataset, logger)

    # Final Dashboard Report
    logger.log("\n" + "=" * 80)
    logger.log("📊 PIPELINE VERIFICATION SUMMARY DASHBOARD")
    logger.log("=" * 80)
    logger.log(f"{'Model Key':<20} {'Status':<10} {'Items':<8} {'Inf Speed (img/s)':<20} {'Load Time':<12} {'Accuracy':<10}")
    logger.log("-" * 80)

    all_passed = True
    for model_key, res in audit_summary.items():
        status = "✅ PASS" if res.get("passed") else "❌ FAIL"
        if not res.get("passed"):
            all_passed = False
        count = res.get("count", 0)
        speed = f"{res.get('inf_throughput', 0.0):.2f}"
        load_t = f"{res.get('load_time', 0.0):.1f}s"
        acc = f"{res.get('accuracy', 0.0):.1f}%" if "accuracy" in res else "N/A"
        logger.log(f"{model_key:<20} {status:<10} {count:<8} {speed:<20} {load_t:<12} {acc:<10}")

    logger.log("-" * 80)
    logger.log(f"Total Test Elapsed: {overall_elapsed:.2f} seconds")
    logger.log(f"Audit Result:       {'🎉 ALL TESTS PASSED' if all_passed else '⚠️ ISSUES DETECTED'}")
    logger.log("=" * 80 + "\n")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test batched and sequential VLM inference pipeline.")
    parser.add_argument("--dataset", default="gqa", help="Dataset name to test (default: gqa)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="List of models to test")
    parser.add_argument("--limit", type=int, default=1000, help="Number of items to test (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size (e.g. 5, 10, 15)")
    parser.add_argument("--mock", action="store_true", help="Run simulated inference without downloading weights")
    parser.add_argument("--force", action="store_true", help="Force re-run from scratch (do not resume previous test run)")
    parser.add_argument("--output-dir", default="results/test_batched_predictions", help="Output directory")
    parser.add_argument("--dynamic-batch", action="store_true", help="Enable dynamic VRAM auto-queue to continuously saturate GPU memory")
    parser.add_argument("--log-file", default="results/test_batched_pipeline.log", help="Log file path")

    args = parser.parse_args()

    success = run_pipeline_test(
        models=args.models,
        dataset=args.dataset,
        limit=args.limit,
        batch_size=args.batch_size,
        mock=args.mock,
        force=args.force,
        dynamic_batch=args.dynamic_batch,
        output_dir=args.output_dir,
        log_file=args.log_file
    )
    sys.exit(0 if success else 1)

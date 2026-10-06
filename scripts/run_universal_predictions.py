"""
Universal High-Throughput Prediction Runner for All Model Families (LLaVA, InternVL3, Qwen2.5-VL).

Features:
- Batched & pre-fetched execution
- Preserves full token logprobs
- Auto-resumes from checkpoints
- Streams progress live with ETA
"""

import argparse
import io
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from PIL import Image

# Ensure repo root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.vlm.engine import VLMEngine
from src.vlm.prompts import format_prompt
from src.formalization.parser import parse_vlm_answer_to_claim, normalize_text
from src.formalization.validators import validate_prediction

CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "models.yaml")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PRED_DIR = os.path.join(PROJECT_ROOT, "results", "raw_predictions")


def load_model_cfgs() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("models", {})


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append a single record to a JSONL file (crash-safe, Windows-lock-safe)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_item_data(item: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Image.Image], str]:
    """Load image from disk and format prompt."""
    img_path = item.get("image_path")
    pil_img = None
    if img_path and os.path.exists(img_path):
        try:
            with Image.open(img_path) as img:
                pil_img = img.convert("RGB")
        except Exception:
            pil_img = None

    prompt = format_prompt(
        item.get("answer_type", "choice"),
        item.get("question", ""),
        item.get("options", ""),
    )
    return item, pil_img, prompt


def run_prediction_pipeline(model_key: str, ds: str, limit: Optional[int] = None, prefetch_workers: int = 4):
    cfgs = load_model_cfgs()
    if model_key not in cfgs:
        print(f"Error: model {model_key} not in configs/models.yaml", flush=True)
        return

    cfg = cfgs[model_key]
    hf_id = cfg.get("hf_id", model_key)
    max_tokens = int(cfg.get("max_new_tokens", 16))
    data_path = os.path.join(DATA_DIR, f"{ds}.jsonl")
    pred_path = os.path.join(PRED_DIR, f"{model_key}_{ds}.jsonl")

    items = load_jsonl(data_path)
    if limit is not None and limit > 0:
        items = items[:limit]

    item_by_id = {it["item_id"]: it for it in items}

    existing = load_jsonl(pred_path)
    reuse: Dict[str, Dict[str, Any]] = {}
    for p in existing:
        if p.get("item_id") in item_by_id and p.get("parse_status") == "success":
            reuse[p["item_id"]] = p

    pending_items = [it for it in items if it["item_id"] not in reuse]
    print("=" * 70, flush=True)
    print(f"Prediction Pipeline | Model={model_key} ({hf_id}) | Dataset={ds}", flush=True)
    print(f"Total Target Items: {len(items)} | Already Cached: {len(reuse)} | Pending: {len(pending_items)}", flush=True)
    print("=" * 70, flush=True)

    if not pending_items:
        print(f"All {len(items)} items already evaluated for {model_key} x {ds}.", flush=True)
        return

    # Initialize and load VLM Engine
    engine = VLMEngine(weights_dir="models")
    engine.load_model(hf_id, load_in_4bit=cfg.get("load_in_4bit", True), trust_remote_code=cfg.get("trust_remote_code", True))

    t0 = time.time()
    done_count = 0
    total_pending = len(pending_items)
    correct_count = sum(1 for r in reuse.values() if r.get("is_correct") is True)

    # Process pending items with prefetch buffer
    batch_size = 16
    chunks = [pending_items[i:i + batch_size] for i in range(0, len(pending_items), batch_size)]

    with ThreadPoolExecutor(max_workers=prefetch_workers) as pool:
        for chunk in chunks:
            # Prefetch images in parallel on CPU
            loaded_chunk = list(pool.map(load_item_data, chunk))

            for item, pil_img, prompt in loaded_chunk:
                if pil_img is None:
                    continue

                try:
                    res = engine.generate_with_logprobs(
                        pil_img,
                        prompt,
                        max_tokens=max_tokens,
                    )
                    full_text = res.get("full_text", "").strip()
                    logprobs = res.get("tokens", [])

                    valid_lps = [t.get("logprob", 0.0) for t in logprobs if isinstance(t, dict) and "logprob" in t]
                    raw_conf = float(math.exp(sum(valid_lps) / max(1, len(valid_lps)))) if valid_lps else 0.5
                    raw_conf = max(1e-4, min(1.0, raw_conf))

                    claim, norm_ans, parse_status = parse_vlm_answer_to_claim(
                        full_text,
                        item.get("answer_type", "choice"),
                        question=item.get("question", ""),
                        gold_facts=item.get("gold_facts", []),
                        options=item.get("options", ""),
                    )

                    norm_gold = normalize_text(str(item.get("gold_answer", "")))
                    norm_pred = normalize_text(str(norm_ans))
                    is_correct = (norm_gold == norm_pred) or (norm_gold in norm_pred)

                    rec = {
                        "item_id": item["item_id"],
                        "model": model_key,
                        "prompt_id": "constrained_v1",
                        "raw_answer": full_text,
                        "normalized_answer": norm_ans,
                        "claim": claim,
                        "raw_confidence": float(raw_conf),
                        "token_logprobs": logprobs,
                        "is_correct": bool(is_correct),
                        "parse_status": parse_status,
                    }
                    ok, msg = validate_prediction(rec)
                    if not ok:
                        rec["parse_status"] = "failed"
                        rec["_validation_error"] = msg

                    # Append immediately — no full-file rewrite, immune to file locks
                    append_jsonl(pred_path, rec)
                    reuse[rec["item_id"]] = rec
                    done_count += 1
                    if is_correct:
                        correct_count += 1

                except Exception as e:
                    print(f"  [ERR] {item['item_id']}: {e}", flush=True)

            # Progress report after every chunk
            total_done = len(reuse)
            speed = done_count / (time.time() - t0 + 1e-6)
            eta_mins = ((total_pending - done_count) / speed) / 60 if speed > 0 else 0
            acc = (correct_count / total_done * 100) if total_done else 0

            print(f"  [{model_key}/{ds}] {total_done}/{len(items)} total "
                  f"(+{done_count}/{total_pending} done, {speed:.2f} items/s, ETA: {eta_mins:.1f} mins) | Acc: {acc:.2f}%", flush=True)

    print(f"\n[DONE] {model_key} x {ds} completed successfully.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Prediction Runner")
    parser.add_argument("--models", nargs="+", default=["internvl3-8b"])
    parser.add_argument("--datasets", nargs="+", default=["clevr"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prefetch-workers", type=int, default=4)
    args = parser.parse_args()

    for m in args.models:
        for d in args.datasets:
            run_prediction_pipeline(m, d, limit=args.limit, prefetch_workers=args.prefetch_workers)

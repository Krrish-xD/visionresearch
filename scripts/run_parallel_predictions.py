"""
Cache-aware parallel prediction client over the VLM HTTP server.

Assumes backend/main.py (uvicorn) is already running, e.g.:
    .venv\\Scripts\\python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

For each model x dataset it:
  1. Loads current data items      data/processed/{ds}.jsonl
  2. Loads existing predictions    results/raw_predictions/{model}_{ds}.jsonl
  3. Reuses valid records (parse_status == "success") whose item_id is in the data.
  4. Drops stale records (item_id no longer present in the data file).
  5. Regenerates missing / failed / mismatched items in parallel via POST /api/generate.
  6. Writes an aligned, all-valid prediction JSONL and reports a per-item audit.

Usage (run from the project root):
    .venv\\Scripts\\python scripts\\run_parallel_predictions.py                       # all models/datasets
    .venv\\Scripts\\python scripts\\run_parallel_predictions.py --dry-run             # plan only
    .venv\\Scripts\\python scripts\\run_parallel_predictions.py --models llava-1.5-7b --datasets mmvp
    .venv\\Scripts\\python scripts\\run_parallel_predictions.py --workers 4 --server http://localhost:8000
"""

import argparse
import io
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from PIL import Image

# Make the project root importable so `src.*` resolves.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.vlm.prompts import format_prompt
from src.formalization.parser import parse_vlm_answer_to_claim, normalize_text
from src.formalization.validators import validate_prediction

CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "models.yaml")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PRED_DIR = os.path.join(PROJECT_ROOT, "results", "raw_predictions")
DATASETS = ["mmvp", "clevr", "gqa"]
ALL_MODELS = ["llava-1.5-7b", "internvl3-8b", "qwen2.5-vl-7b"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model_cfgs() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("models", {})


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    return out


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def failed_placeholder(item_id: str, model_key: str, reason: str) -> Dict[str, Any]:
    return {
        "item_id": item_id,
        "model": model_key,
        "prompt_id": "constrained_v1",
        "raw_answer": "",
        "normalized_answer": "",
        "claim": None,
        "raw_confidence": 0.0,
        "token_logprobs": [],
        "is_correct": False,
        "parse_status": "failed",
        "_server_error": reason,
    }


def build_record(full_text: str, tokens: List[Dict[str, Any]], item: Dict[str, Any],
                 model_key: str) -> Dict[str, Any]:
    """Mirror src/vlm/evaluate.py run_inference_on_item post-processing."""
    logprobs = [float(t.get("logprob", 0.0)) for t in tokens]
    if logprobs:
        mean_lp = float(sum(logprobs) / len(logprobs))
        raw_conf = float(math.exp(mean_lp))
    else:
        raw_conf = 0.5
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

    # answer_logits are intentionally omitted: the /api/generate contract returns
    # per-token logprobs only (not full-vocab logits). run_experiment_pipeline
    # falls back to a logit vector derived from raw_confidence when answer_logits
    # is absent, so this is safe.
    return {
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


def server_generate(server_url: str, hf_id: str, item: Dict[str, Any],
                    max_tokens: int, timeout: int = 300) -> Tuple[str, List[Dict[str, Any]]]:
    """POST /api/generate for a single item; return (full_text, tokens)."""
    image_path = item.get("image_path")
    prompt = format_prompt(
        item.get("answer_type", "choice"),
        item.get("question", ""),
        item.get("options", ""),
    )

    files = None
    if image_path and os.path.exists(image_path):
        files = {"image": ("img.png", open(image_path, "rb"), "image/png")}
    else:
        raise FileNotFoundError(f"missing image for {item['item_id']}: {image_path}")

    form = {
        "model_id": hf_id,
        "prompt": prompt,
        "temperature": "0.0",
        "top_p": "1.0",
        "top_k": "50",
        "max_tokens": str(max_tokens),
    }
    try:
        resp = requests.post(
            f"{server_url}/api/generate",
            data=form,
            files=files,
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        return body.get("full_text", ""), body.get("tokens", [])
    finally:
        if files is not None:
            files["image"][1].close()


# ---------------------------------------------------------------------------
# Per-model/dataset orchestration
# ---------------------------------------------------------------------------
def process_model_dataset(model_key: str, ds: str, cfg: Dict[str, Any],
                          server_url: str, workers: int, dry_run: bool) -> Dict[str, int]:
    hf_id = cfg.get("hf_id", model_key)
    max_tokens = int(cfg.get("max_new_tokens", 16))
    data_path = os.path.join(DATA_DIR, f"{ds}.jsonl")
    pred_path = os.path.join(PRED_DIR, f"{model_key}_{ds}.jsonl")

    items = load_jsonl(data_path)
    item_by_id = {it["item_id"]: it for it in items}

    existing = load_jsonl(pred_path)
    existing_by_id: Dict[str, Dict[str, Any]] = {}
    for p in existing:
        existing_by_id[p["item_id"]] = p  # last occurrence wins

    # Reuse valid records present in the current data; everything else is pending.
    reuse: Dict[str, Dict[str, Any]] = {}
    for iid, p in existing_by_id.items():
        if iid in item_by_id and p.get("parse_status") == "success":
            reuse[iid] = p

    pending_items = [it for it in items if it["item_id"] not in reuse]
    stale_drop = len([p for p in existing if p.get("item_id") not in item_by_id])

    stats = {
        "data": len(items),
        "reuse": len(reuse),
        "regen": len(pending_items),
        "stale_drop": stale_drop,
    }

    print(f"  [{model_key} x {ds}] data={stats['data']} reuse={stats['reuse']} "
          f"regen={stats['regen']} stale_drop={stale_drop}")

    if dry_run or stats["regen"] == 0:
        if not dry_run:
            aligned = [reuse.get(it["item_id"], failed_placeholder(it["item_id"], model_key, "missing"))
                       for it in items]
            write_jsonl(pred_path, aligned)
            _report(pred_path, model_key, ds)
        return stats

    # Ensure the requested model is resident on the server before fan-out.
    try:
        r = requests.post(f"{server_url}/api/model/load",
                          json={"model_id": hf_id}, timeout=900)
        if r.status_code != 200:
            print(f"    [WARN] load {hf_id}: HTTP {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"    [WARN] load {hf_id}: {e}")

    results: Dict[str, Dict[str, Any]] = {}
    t0 = time.time()
    done = 0

    def gen_one(it: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        full_text, tokens = server_generate(server_url, hf_id, it, max_tokens)
        rec = build_record(full_text, tokens, it, model_key)
        ok, msg = validate_prediction(rec)
        if not ok:
            rec["parse_status"] = "failed"
            rec["_validation_error"] = msg
        return it["item_id"], rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(gen_one, it): it["item_id"] for it in pending_items}
        for fut in as_completed(futures):
            iid = futures[fut]
            done += 1
            try:
                _, rec = fut.result()
                results[iid] = rec
            except Exception as e:
                print(f"    [ERR] {iid}: {e}", flush=True)
                results[iid] = failed_placeholder(iid, model_key, str(e)[:200])
            if done % 25 == 0 or done == len(pending_items):
                speed_str = f" ({done/(time.time()-t0+1e-9):.2f}/s)" if time.time() - t0 > 0.1 else ""
                print(f"    {model_key}/{ds}: {done}/{len(pending_items)} done{speed_str}", flush=True)
                # Incremental checkpointing every 50 items
                if done % 50 == 0 or done == len(pending_items):
                    cur_aligned = []
                    for it in items:
                        if it["item_id"] in reuse:
                            cur_aligned.append(reuse[it["item_id"]])
                        elif it["item_id"] in results:
                            cur_aligned.append(results[it["item_id"]])
                    if cur_aligned:
                        write_jsonl(pred_path, cur_aligned)

    # Final assemble: current data order, reusing valid records, filling regen.
    aligned: List[Dict[str, Any]] = []
    for it in items:
        if it["item_id"] in reuse:
            aligned.append(reuse[it["item_id"]])
        elif it["item_id"] in results:
            aligned.append(results[it["item_id"]])
        else:
            aligned.append(failed_placeholder(it["item_id"], model_key, "regen_failed"))

    write_jsonl(pred_path, aligned)
    _report(pred_path, model_key, ds)
    return stats


def _report(pred_path: str, model_key: str, ds: str) -> None:
    recs = load_jsonl(pred_path)
    succ = sum(1 for r in recs if r.get("parse_status") == "success")
    correct = sum(1 for r in recs if r.get("is_correct") is True)
    acc = (correct / len(recs) * 100) if recs else 0.0
    print(f"  -> {os.path.basename(pred_path)}: {succ}/{len(recs)} valid | Accuracy: {correct}/{len(recs)} ({acc:.2f}%)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Cache-aware parallel VLM prediction client (HTTP server)")
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfgs = load_model_cfgs()
    missing = [m for m in args.models if m not in cfgs]
    if missing:
        print(f"[FATAL] model config(s) missing in configs/models.yaml: {missing}")
        sys.exit(2)

    print("=" * 70)
    print(f" Parallel VLM prediction client  | server={args.server} | workers={args.workers}")
    print(f" models={args.models}  datasets={args.datasets}  dry_run={args.dry_run}")
    print("=" * 70)

    for model_key in args.models:
        cfg = cfgs[model_key]
        for ds in args.datasets:
            try:
                process_model_dataset(model_key, ds, cfg, args.server, args.workers, args.dry_run)
            except Exception as e:
                print(f"  [ERROR] {model_key} x {ds}: {e}")
                traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
High-Throughput Batched GPU Prediction Engine for VisionResearch.

Direct in-process batch inference (Batch Size = 8/16) with multi-threaded prefetching.
Reuses existing checkpoints from results/raw_predictions/{model}_{ds}.jsonl.
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
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM

# Ensure repo root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

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


def append_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    os.replace(temp, path)


def load_image_and_prompt(item: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Image.Image], str]:
    """Load image from disk and build formatted prompt."""
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


class BatchedInferenceEngine:
    """Direct in-process GPU batched generator with per-token logprob extraction."""

    def __init__(self, model_key: str, cfg: Dict[str, Any], device: str = "cuda"):
        self.model_key = model_key
        self.cfg = cfg
        self.device = device
        self.hf_id = cfg.get("hf_id", model_key)
        self.max_new_tokens = int(cfg.get("max_new_tokens", 16))
        self.model = None
        self.processor = None
        self._load()

    def _load(self):
        print(f"Loading {self.model_key} ({self.hf_id}) into GPU...", flush=True)
        from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.processor = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            self.processor.tokenizer.padding_side = "left"

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.hf_id,
            quantization_config=bnb if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"Model {self.model_key} loaded successfully.", flush=True)

    @torch.inference_mode()
    def generate_batch(self, batch: List[Tuple[Dict[str, Any], Image.Image, str]]) -> List[Dict[str, Any]]:
        """Run batched forward pass and return validated records."""
        items = [b[0] for b in batch]
        images = [b[1] for b in batch]
        prompts = [b[2] for b in batch]

        # Format prompts for LLaVA
        formatted_prompts = [f"USER: <image>\n{p}\nASSISTANT:" for p in prompts]
        
        inputs = self.processor(
            text=formatted_prompts,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        seqs = outputs.sequences  # shape: (B, prompt_len + gen_len)
        scores = outputs.scores   # tuple of gen_len tensors, each (B, vocab)
        gen_tokens = seqs[:, prompt_len:]
        tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor

        batch_records = []
        for i, item in enumerate(items):
            gen_ids = gen_tokens[i]
            # Decode generated text
            full_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # Extract logprobs
            logprobs = []
            valid_lps = []
            for t_step, score_tensor in enumerate(scores):
                if t_step >= len(gen_ids):
                    break
                tid = gen_ids[t_step].item()
                if tid in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                    break
                tok_str = tokenizer.decode([tid])
                step_logits = score_tensor[i]
                step_lps = torch.log_softmax(step_logits, dim=-1)
                lp_val = float(step_lps[tid].item())
                valid_lps.append(lp_val)
                logprobs.append({"token": tok_str, "logprob": lp_val})

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
                "model": self.model_key,
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

            batch_records.append(rec)

        return batch_records


def run_model_dataset_batched(model_key: str, ds: str, batch_size: int = 8, prefetch_workers: int = 4):
    cfgs = load_model_cfgs()
    if model_key not in cfgs:
        print(f"Error: model {model_key} not in configs/models.yaml", flush=True)
        return

    cfg = cfgs[model_key]
    data_path = os.path.join(DATA_DIR, f"{ds}.jsonl")
    pred_path = os.path.join(PRED_DIR, f"{model_key}_{ds}.jsonl")

    items = load_jsonl(data_path)
    item_by_id = {it["item_id"]: it for it in items}

    existing = load_jsonl(pred_path)
    reuse: Dict[str, Dict[str, Any]] = {}
    for p in existing:
        if p.get("item_id") in item_by_id and p.get("parse_status") == "success":
            reuse[p["item_id"]] = p

    pending_items = [it for it in items if it["item_id"] not in reuse]
    print("=" * 70, flush=True)
    print(f"Batched GPU Inference | Model={model_key} | Dataset={ds} | Batch Size={batch_size}", flush=True)
    print(f"Total: {len(items)} | Already Done: {len(reuse)} | Pending: {len(pending_items)}", flush=True)
    print("=" * 70, flush=True)

    if not pending_items:
        print(f"All {len(items)} items already evaluated for {model_key} x {ds}.", flush=True)
        return

    engine = BatchedInferenceEngine(model_key, cfg)

    # Chunk pending items into batches
    batches = [pending_items[i:i + batch_size] for i in range(0, len(pending_items), batch_size)]
    t0 = time.time()
    done_count = 0
    total_pending = len(pending_items)

    with ThreadPoolExecutor(max_workers=prefetch_workers) as pool:
        for b_idx, batch_items in enumerate(batches):
            # Pre-load images in parallel
            loaded_batch = list(pool.map(load_image_and_prompt, batch_items))
            # Filter out any missing images
            valid_batch = [b for b in loaded_batch if b[1] is not None]
            if not valid_batch:
                continue

            # Run GPU batched forward pass
            batch_recs = engine.generate_batch(valid_batch)
            
            # Save batch immediately to disk
            for r in batch_recs:
                reuse[r["item_id"]] = r
            
            # Incremental flush to file
            aligned = [reuse.get(it["item_id"]) for it in items if it["item_id"] in reuse]
            write_jsonl(pred_path, aligned)

            done_count += len(batch_recs)
            speed = done_count / (time.time() - t0 + 1e-6)
            eta_mins = ((total_pending - done_count) / speed) / 60 if speed > 0 else 0
            correct = sum(1 for r in aligned if r and r.get("is_correct") is True)
            acc = (correct / len(aligned) * 100) if aligned else 0

            print(f"  [{model_key}/{ds}] {len(aligned)}/{len(items)} total "
                  f"(+{done_count}/{total_pending} done, {speed:.2f} items/s, ETA: {eta_mins:.1f} mins) | Acc: {acc:.2f}%", flush=True)

    print(f"\n[DONE] {model_key} x {ds} completed successfully.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batched VLM inference")
    parser.add_argument("--models", nargs="+", default=["llava-1.5-7b"])
    parser.add_argument("--datasets", nargs="+", default=["clevr"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prefetch-workers", type=int, default=4)
    args = parser.parse_args()

    for m in args.models:
        for d in args.datasets:
            run_model_dataset_batched(m, d, batch_size=args.batch_size, prefetch_workers=args.prefetch_workers)

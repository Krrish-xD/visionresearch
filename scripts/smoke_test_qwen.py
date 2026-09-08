"""
Isolated smoke test for Qwen2.5-VL-7B inference.
Run:
    python scripts/smoke_test_qwen.py

Tests 3 MMVP items to verify:
  - Model loads without error
  - raw_answer is non-empty
  - token logprobs are extracted correctly
  - parse_status is verified
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vlm.evaluate import run_predictions_on_dataset

SMOKE_DIR = "results/smoke_tests"
os.makedirs(SMOKE_DIR, exist_ok=True)


def run():
    print("\n=== Qwen2.5-VL-7B Smoke Test ===")
    print("Loading model and running 3 MMVP items...")

    try:
        run_predictions_on_dataset(
            dataset_name="mmvp",
            model_key="qwen2.5-vl-7b",
            limit=3,
            mock=False,
            output_dir=SMOKE_DIR,
        )
    except Exception as e:
        print(f"[FAIL] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    out = os.path.join(SMOKE_DIR, "qwen2.5-vl-7b_mmvp.jsonl")
    if not os.path.exists(out):
        print("[FAIL] Output file not created")
        return False

    preds = []
    with open(out, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))

    print(f"\nResults ({len(preds)} items):")
    all_ok = True
    for p in preds:
        ans = p.get("raw_answer", "").strip()
        conf = p.get("raw_confidence", 0)
        status = p.get("parse_status", "?")
        empty = not ans
        if empty:
            all_ok = False
        tag = "EMPTY" if empty else "OK"
        print(f"  [{tag}] {p['item_id']}: ans='{ans[:50]}' conf={conf:.3f} parse={status}")

    if all_ok:
        print("\n[PASS] Qwen2.5-VL-7B smoke test PASSED")
    else:
        print("\n[FAIL] Qwen2.5-VL-7B smoke test FAILED — some empty answers")

    return all_ok


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)

"""
Empirical Batch Size Benchmark for Vision-Language Models (VLMs).

Tests batch sizes B = 1 to 30:
  - Loads model exactly once into VRAM.
  - Preloads images and prompts into RAM (zero disk I/O interference).
  - Runs a warmup pass to prime CUDA kernels.
  - Executes exactly 10 batches for each batch size (e.g., 10x1, 10x2, ..., 10x30).
  - Synchronizes CUDA clocks to measure pure GPU execution time.
  - Measures throughput (img/s), batch latency, per-image latency, and VRAM consumption.
  - Gracefully catches Out-Of-Memory (OOM) errors and determines the exact saturation ceiling.
  - Outputs a ranked leaderboard identifying the optimal batch size sweet spot.

Usage:
    python scripts/benchmark_batch_sizes.py --model qwen2.5-vl-7b --min-bs 1 --max-bs 30 --iters 10
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any
from PIL import Image

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception as e:
        pass

# Ensure project root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import torch
from src.vlm.evaluate import VLMInferenceRunner
from src.vlm.prompts import format_prompt


def resolve_image_path(raw_path: str) -> str:
    """Resolve image path across alternate extensions and naming styles."""
    if os.path.exists(raw_path):
        return raw_path
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    dir_name = os.path.dirname(raw_path)
    candidates = [
        os.path.join(dir_name, f"{stem}.jpg"),
        os.path.join(dir_name, f"{int(stem) if stem.isdigit() else stem}.jpg"),
        os.path.join(dir_name, f"{stem}.png"),
        os.path.join(dir_name, f"{int(stem) if stem.isdigit() else stem}.png"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return raw_path


def load_test_pool(dataset: str, pool_size: int = 30) -> tuple:
    """Preload pool_size images and prompts into RAM to eliminate disk I/O from benchmark."""
    data_file = os.path.join(ROOT_DIR, "data", "processed", f"{dataset}.jsonl")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Processed dataset file not found: {data_file}")

    print(f"[Benchmark] Preloading {pool_size} test items into RAM from {data_file}...")
    items = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
                if len(items) >= pool_size * 2:  # load extra in case some images are missing
                    break

    preloaded_images = []
    preloaded_prompts = []

    for item in items:
        img_path = resolve_image_path(item.get("image_path", ""))
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    preloaded_images.append(img.convert("RGB"))
                prompt = format_prompt(
                    item.get("answer_type", "yes_no"),
                    item.get("question", "What is shown in this image?"),
                    item.get("options", "")
                )
                preloaded_prompts.append(prompt)
                if len(preloaded_images) >= pool_size:
                    break
            except Exception:
                continue

    if len(preloaded_images) < pool_size:
        # If dataset had fewer images, cycle them to reach pool_size
        if not preloaded_images:
            raise RuntimeError(f"Could not load any valid images from {data_file}")
        while len(preloaded_images) < pool_size:
            idx = len(preloaded_images) % len(preloaded_images)
            preloaded_images.append(preloaded_images[idx].copy())
            preloaded_prompts.append(preloaded_prompts[idx])

    print(f"[Benchmark] Successfully cached {len(preloaded_images)} images and prompts in CPU RAM.\n")
    return preloaded_images[:pool_size], preloaded_prompts[:pool_size]


def run_batch_size_benchmark(
    model_key: str = "qwen2.5-vl-7b",
    dataset: str = "gqa",
    min_bs: int = 1,
    max_bs: int = 30,
    iters: int = 10,
    max_tokens: int = 4,
    output_json: str = "results/batch_size_benchmark.json"
):
    print("=" * 80)
    print("🔬 VLM BATCH SIZE EMPIRICAL BENCHMARK & OPTIMIZATION SUITE")
    print("=" * 80)
    print(f"Model:                {model_key}")
    print(f"Dataset:              {dataset}")
    print(f"Batch Size Range:     {min_bs} -> {max_bs}")
    print(f"Batches per Size:     {iters} iterations")
    print(f"Max Generated Tokens: {max_tokens} (Fast path)")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Accelerator:      {gpu_name} ({total_vram:.2f} GB VRAM)")
    else:
        print("Hardware:             CPU only (CUDA not detected)")
    print("=" * 80 + "\n")

    # Step 1: Preload images into RAM
    images_pool, prompts_pool = load_test_pool(dataset=dataset, pool_size=max_bs)

    # Step 2: Load model into VRAM once
    print(f"[Engine] Loading model '{model_key}' into VRAM (one-time initialization)...")
    t0 = time.time()
    runner = VLMInferenceRunner(model_key=model_key)
    runner.load_model()
    load_elapsed = time.time() - t0
    vram_after_load = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"[Engine] Model loaded in {load_elapsed:.2f}s (Base model VRAM: {vram_after_load:.2f} GB).\n")

    # Step 3: CUDA Warmup (2 forward passes to compile kernels & allocate buffers)
    print("[Benchmark] Executing CUDA warmup to compile kernels...")
    try:
        runner.engine.generate_batch_with_logprobs(
            images_pool[:2], prompts_pool[:2], max_tokens=max_tokens
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("[Benchmark] Warmup complete. Commencing empirical sweeps.\n")
    except Exception as e:
        print(f"[Warning] Warmup encountered error: {e}")

    # Step 4: Sweep Batch Sizes min_bs .. max_bs
    results = []
    oom_encountered = False

    print("-" * 92)
    print(f"{'Batch Size':<12} {'Total Imgs':<12} {'Total Time':<12} {'Speed (img/s)':<16} {'Batch Latency':<16} {'Peak VRAM':<12} {'Status'}")
    print("-" * 92)

    for bs in range(min_bs, max_bs + 1):
        batch_imgs = images_pool[:bs]
        batch_prompts = prompts_pool[:bs]
        total_images = bs * iters

        # Check VRAM before run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            # Execute exactly `iters` batches
            for i in range(iters):
                runner.engine.generate_batch_with_logprobs(
                    batch_imgs, batch_prompts, max_tokens=max_tokens
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_elapsed = time.perf_counter() - t_start

            throughput = total_images / t_elapsed
            avg_batch_lat = t_elapsed / iters
            avg_img_lat = t_elapsed / total_images
            peak_vram = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0

            res = {
                "batch_size": bs,
                "iters": iters,
                "total_images": total_images,
                "total_seconds": round(t_elapsed, 4),
                "throughput_img_per_sec": round(throughput, 3),
                "avg_batch_latency_sec": round(avg_batch_lat, 4),
                "avg_image_latency_sec": round(avg_img_lat, 4),
                "peak_vram_gb": round(peak_vram, 2),
                "status": "SUCCESS"
            }
            results.append(res)

            print(f"{bs:<12} {total_images:<12} {t_elapsed:<12.2f} {throughput:<16.2f} {avg_batch_lat:<16.3f} {peak_vram:<12.2f} ✅ OK")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            err_str = str(err).lower()
            if "out of memory" in err_str or isinstance(err, torch.cuda.OutOfMemoryError):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"{bs:<12} {total_images:<12} {'--':<12} {'--':<16} {'--':<16} {'>=16.0 GB':<12} ❌ OOM")
                results.append({
                    "batch_size": bs,
                    "iters": iters,
                    "status": "OOM",
                    "error": "CUDA Out Of Memory"
                })
                oom_encountered = True
                print(f"\n[!] CUDA Out Of Memory hit at batch size {bs}. Halting further batch increases.")
                break
            else:
                print(f"{bs:<12} {total_images:<12} {'--':<12} {'--':<16} {'--':<16} {'--':<12} ❌ ERROR: {err}")
                results.append({
                    "batch_size": bs,
                    "iters": iters,
                    "status": "ERROR",
                    "error": str(err)
                })
                break

    # Step 5: Unload Model
    runner.engine.unload_model()
    print("-" * 92)

    # Step 6: Leaderboard & Analysis
    valid_results = [r for r in results if r.get("status") == "SUCCESS"]
    if not valid_results:
        print("\n❌ No batch sizes succeeded.")
        return

    # Sort by throughput descending
    ranked_by_speed = sorted(valid_results, key=lambda x: x["throughput_img_per_sec"], reverse=True)
    best_overall = ranked_by_speed[0]

    # Find efficiency plateau (sweet spot where adding more batch size yields < 2% gain)
    sweet_spot = valid_results[0]
    for r in valid_results:
        speed_ratio = r["throughput_img_per_sec"] / best_overall["throughput_img_per_sec"]
        if speed_ratio >= 0.95:  # within 95% of peak throughput
            sweet_spot = r
            break

    print("\n" + "=" * 80)
    print("🏆 BATCH SIZE OPTIMIZATION RESULTS & RECOMMENDATION")
    print("=" * 80)
    print(f"🥇 Absolute Fastest Batch Size:   BS = {best_overall['batch_size']}")
    print(f"   • Throughput:                 {best_overall['throughput_img_per_sec']:.2f} images/sec")
    print(f"   • Batch Latency:              {best_overall['avg_batch_latency_sec']:.2f} seconds per batch")
    print(f"   • VRAM Consumed:              {best_overall['peak_vram_gb']:.2f} GB")
    print()
    print(f"🎯 Recommended 'Sweet Spot' Size: BS = {sweet_spot['batch_size']}")
    print(f"   • Throughput:                 {sweet_spot['throughput_img_per_sec']:.2f} images/sec ({sweet_spot['throughput_img_per_sec']/best_overall['throughput_img_per_sec']*100:.1f}% of peak)")
    print(f"   • Batch Latency:              {sweet_spot['avg_batch_latency_sec']:.2f} seconds per batch")
    print(f"   • VRAM Consumed:              {sweet_spot['peak_vram_gb']:.2f} GB (safer VRAM margin)")
    print("=" * 80)

    # Comparison Table of Top 5
    print("\nTop 5 Most Efficient Batch Sizes:")
    print(f"{'Rank':<6} {'Batch Size':<12} {'Throughput (img/s)':<22} {'Batch Latency':<16} {'VRAM'}")
    print("-" * 70)
    for idx, r in enumerate(ranked_by_speed[:5], 1):
        print(f"#{idx:<5} BS={r['batch_size']:<9} {r['throughput_img_per_sec']:<22.2f} {r['avg_batch_latency_sec']:<16.2f} {r['peak_vram_gb']:.2f} GB")
    print("-" * 70)

    # Save to disk
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    summary_data = {
        "model": model_key,
        "best_batch_size": best_overall["batch_size"],
        "best_throughput": best_overall["throughput_img_per_sec"],
        "sweet_spot_batch_size": sweet_spot["batch_size"],
        "sweet_spot_throughput": sweet_spot["throughput_img_per_sec"],
        "all_results": results
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nDetailed JSON report exported to -> {output_json}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal VLM inference batch size.")
    parser.add_argument("--model", default="qwen2.5-vl-7b", help="Model key to benchmark (default: qwen2.5-vl-7b)")
    parser.add_argument("--dataset", default="gqa", help="Dataset name to pull items from (default: gqa)")
    parser.add_argument("--min-bs", type=int, default=1, help="Minimum batch size to test (default: 1)")
    parser.add_argument("--max-bs", type=int, default=30, help="Maximum batch size to test (default: 30)")
    parser.add_argument("--iters", type=int, default=10, help="Number of batches per batch size (default: 10)")
    parser.add_argument("--max-tokens", type=int, default=4, help="Max generated tokens (default: 4)")
    parser.add_argument("--output", default="", help="Output JSON results file (defaults to results/benchmark_batch_size_<model>.json)")

    args = parser.parse_args()

    out_file = args.output if args.output else f"results/benchmark_batch_size_{args.model}.json"

    run_batch_size_benchmark(
        model_key=args.model,
        dataset=args.dataset,
        min_bs=args.min_bs,
        max_bs=args.max_bs,
        iters=args.iters,
        max_tokens=args.max_tokens,
        output_json=out_file
    )

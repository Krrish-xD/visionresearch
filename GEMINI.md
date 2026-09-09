# GEMINI Context - VisionResearch: Calibrated Confidence for Symbolic Error Detection

## 1. Project Overview & Scientific Motivation
This repository is an empirical research and benchmarking study (targeting top-tier conferences such as CVPR / NeurIPS / ECCV) investigating neuro-symbolic error detection and recovery in Vision-Language Models (VLMs).

- **The Problem**: Modern VLMs (LLaVA, InternVL3, Qwen2.5-VL) hallucinate visual relations, objects, counts, and subtle visual contrasts, accompanied by **pathological overconfidence** (raw Expected Calibration Error > 50%–60%; model confidences cluster at 90%–99% even when wrong).
- **The Neuro-Symbolic Bottleneck**:
  1. *Unsound Soft-Weighting*: When ground-truth scene facts are weighted softly alongside VLM claims in MaxSMT optimization, overconfident hallucinations overpower ground truth, causing **Solver Over-Override (SOOR)** events in 91%–100% of error cases.
  2. *Single-Claim Degeneracy*: When ground truth is hard/immutable, single scalar predictions trivially conflict or pass, making confidence weighting redundant unless candidate hypotheses compete or selective verification is applied.
- **The Solution**:
  1. Recalibrate VLM confidence using post-hoc methods (notably **Isotonic Regression**, which collapses raw ECE from ~60% down to 0.3%–7.3%).
  2. Enforce **Hard Ground Truth** in Z3 SMT verification to prevent SOOR and bound the Solver False-Accept Rate (SFAR).
  3. Deploy **Multi-Hypothesis $N$-Best MaxSMT** to actively recover from base VLM misclassifications (recovering up to 92.55% of errors).
  4. Use calibrated conformal risk thresholds for **Selective Abstention**, establishing a monotonic Coverage-Risk (SFAR) frontier.

## 2. Integrated Architecture (Merged from 'shubh' & 'krrish')
The codebase on `main` integrates the optimal features from both feature branches:
- **Backend & API (`src/api/routes.py`, `backend/main.py`)**:
  - Non-blocking async execution: heavy model loading and inference calls are offloaded via `fastapi.concurrency.run_in_threadpool`.
  - Concurrency serialization: `_model_load_lock = asyncio.Lock()` prevents simultaneous model load/unload collisions.
  - Payload pre-validation: image format validation runs prior to model invocation, rejecting malformed requests immediately.
- **VLM Engine (`src/vlm/engine.py`, `src/vlm/confidence.py`, `src/vlm/internvl_inference.py`)**:
  - Native in-engine batching: `generate_batch_with_logprobs()` for Qwen (`_generate_qwen_batch`) and LLaVA/OneVision/NeXT (`_generate_generic_batch`).
  - PyTorch SDPA: `attn_implementation="sdpa"` support for high-throughput attention.
  - Safe lifecycle: explicit `unload_model()` resets `self.model` and `self.processor` to `None` with `gc.collect()` and `torch.cuda.empty_cache()`.
  - Batch-aware confidence: `extract_token_confidence(..., batch_idx=...)` correctly indexes batched score tensors.
  - Bugfix: safe `eos_token_id` retrieval in InternVL inference.
- **Evaluation Pipeline (`src/vlm/evaluate.py`)**:
  - Batched dataset processing: `run_inference_on_batch()` with `batch_size: int = 8`.
  - Checkpoint resumption: `resume=True` skips already computed items in output JSONL and streams new predictions with `f_out.flush()`.
  - Sequential model execution: `run_sequential_models()` executes models sequentially with strict VRAM cleanup between models, adhering to a 16 GB VRAM budget.
- **Datasets & Formalization (`src/datasets/`, `src/formalization/schema.py`)**:
  - CLEVR v1.0 AST parser: `src/datasets/clevr_real.py` derives answer types/categories from functional execution programs (terminal operators and `relate` spatial relations).
  - CLEVR 1k Splits: `data/splits/clevr_splits.json` updated to 1,000 stratified items (400 calibration / 600 evaluation).
  - GQA real image resolution: `src/datasets/gqa.py` dynamically links to authentic extracted images.
  - Enriched prediction schema: `PREDICTION_SCHEMA` supports optional diagnostic fields without schema failure.
- **Models Configuration (`configs/models.yaml`)**:
  - Standardized local `weights/<model>` paths.
  - Added `llava-1.5-7b` entry.
  - Configured `attn_implementation: "sdpa"` and `max_new_tokens: 4` for short-answer tasks.
- **Benchmarking & Utility Scripts (`scripts/`)**:
  - `run_batch_predictions.py`: Direct GPU batch generation with prefetch threadpool.
  - `run_parallel_predictions.py`: Cache-aware parallel HTTP client over `/api/generate`.
  - `unpack_gqa_images.py`: Extraction tool for Hugging Face Parquet GQA shards.
  - `benchmark_batch_sizes.py`: Empirical saturation benchmark (testing batch sizes 1 to 30 with OOM recovery).
  - `generate_curated_subset.py`: Balanced 2,500-item benchmark generator across the 4 cognitive pillars.
  - `test_batched_pipeline.py`: Comprehensive test and verification script for batched and sequential pipelines.

## 3. Current State
- Branch: `main`
- All changes from `shubh` and `krrish` are merged, verified for syntax, and staged.
- Working tree is clean and ready for commit.

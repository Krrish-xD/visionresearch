# VisionResearch: Calibrated Confidence for Symbolic Error Detection in Visual Grounding

Reproducible empirical evaluation testing whether post-hoc calibration of Vision-Language Model (VLM) confidence improves symbolic contradiction detection and reduces the **Solver False-Accept Rate (SFAR)** when VLM claims are checked via **Z3 / MaxSMT**.

---

## 🛠️ Environment & Prerequisites

- **Python**: 3.11.x (managed via `uv`)
- **GPU**: NVIDIA RTX 2000 Ada Generation (16 GB VRAM, CUDA 12.x)
- **Virtual Environment**: `venv`

To setup the virtual environment, install dependencies, and launch the Backend + Web UI servers:
```bash
python start.py
```

---

## 🚀 Quickstart Commands

### 1. Prepare Datasets (MMVP, CLEVR, GQA)
```bash
python -m src.datasets.prepare --config configs/experiment.yaml
```

### 2. Generate Stratified Splits (40% Calibration / 60% Evaluation)
```bash
python -m src.evaluation.make_splits --config configs/experiment.yaml
```

### 3. Run Unit Tests (139 tests)
```bash
venv/bin/pytest tests/ -v
```

### 4. Run Experiment Pipeline
```bash
# Run Pilot (50 items)
python -m src.evaluation.run_experiment --config configs/experiment.yaml --dataset mmvp --model llava-1.5-7b --pilot

# Run Full Benchmark
python -m src.evaluation.run_experiment --config configs/experiment.yaml --dataset mmvp --model llava-1.5-7b
```

---

## 📊 Experimental Results & Outputs

- **Per-Item Audit Log**: `results/metrics/{model}_{dataset}_per_item_audit.csv`
- **Tables (CSV, MD, LaTeX)**:
  - `results/metrics/table1_dataset_composition.*`
  - `results/metrics/table2_calibration_metrics.*`
  - `results/metrics/table3_contradiction_metrics.*`
  - `results/metrics/table4_category_breakdown.*`
  - `results/metrics/table5_solver_runtime.*`
- **Figures (300 DPI)**:
  - `results/figures/fig2_reliability_diagrams.png`
  - `results/figures/fig3_f1_bootstrap_ci.png`
  - `results/figures/fig4_sfar_by_category.png`
  - `results/figures/fig5_ece_vs_sfar.png`

---

## 📐 Project Architecture

```text
visionresearch/
├── backend/                 # FastAPI application, REST endpoints, and router modules
├── frontend/                # Vite/React Web UI
├── configs/
│   ├── experiment.yaml          # Dataset paths, split ratios, seeds, solver timeouts
│   └── models.yaml              # Model IDs, quantization settings, prompt templates
├── data/
│   ├── raw/                     # Raw downloaded images and metadata
│   ├── processed/               # Standardized JSONL items (2,300 items)
│   └── splits/                  # Stratified 40/60 splits
├── src/
│   ├── datasets/                # MMVP, CLEVR, GQA loaders and standardizers
│   ├── vlm/                     # VLM model engine (engine.py), evaluation script (evaluate.py), pipeline utilities
│   ├── formalization/           # JSON schemas, deterministic parser, and validators
│   ├── solver/                  # Z3 AST encoder, MaxSMT verifier, SFAR diagnostics
│   ├── calibration/             # Temperature scaling, Isotonic, Conformal risk & ECE metrics
│   └── evaluation/              # Experiment runner, statistical tests, tables & plots
└── tests/                       # Pytest suite covering all modules (139 tests)
```

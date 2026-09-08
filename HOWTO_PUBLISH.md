# HOWTO_PUBLISH.md — Submission Checklist & Guide

> **Authored for**: VisionResearch Neuro-Symbolic Calibration Paper  
> **Target Venue**: International Conference (e.g., CVPR / ECCV / NeurIPS / AAAI / ICCV)  
> **Manuscript**: `paper/main.tex` + `paper/references.bib`

---

## 1. What Has Been Completed

### Research & Experiments
| Item | Status |
|------|--------|
| Inference pipeline (LLaVA-1.5-7B, InternVL3-8B, Qwen2.5-VL-7B) | DONE |
| Data splits (train/calibration/evaluation) for MMVP, CLEVR, GQA | DONE |
| Raw prediction JSONL files (9 conditions) | DONE |
| Normalisation / text parsing (4 answer-extraction variants) | DONE |
| Temperature Scaling (multinomial NLL minimisation per split) | DONE |
| Isotonic Regression (PAVA) (non-parametric monotonic calibration) | DONE |
| APS Conformal Risk Control (distribution-free coverage guarantee) | DONE |
| All 4 calibration methods tested on all 9 conditions | DONE |
| N-Best Hypothesis MaxSMT Selection | DONE |
| MMVP Paired Visual Consistency Analysis | DONE |
| Baseline comparison (Greedy / Thresholding / Self-Consistency) | DONE |
| Selective Abstention Controller (coverage-risk frontier) | DONE |
| Full test suite: 165/165 tests PASS | DONE |

### Paper Content
| Item | Status |
|------|--------|
| Abstract | DONE |
| Introduction + 3 main contributions | DONE |
| Related Work (VLM hallucination, calibration, neuro-symbolic) | DONE |
| Problem formulation & soundness theory (SOOR definition) | DONE |
| Methods section (full pipeline description) | DONE |
| Results: Table 2 (Calibration ECE/Brier/NLL), Table 3 (SFAR), Table 4 (Category) | DONE |
| Results: Table 6 (Multi-hypothesis MaxSMT), Table 7 (Selective Abstention) | DONE |
| Results: Table 8 (MMVP Paired Consistency), Table 9 (Baseline Comparison) | DONE |
| Qualitative Case Studies section (4 cases) | DONE |
| Discussion & Threats to Validity | DONE |
| Conclusion | DONE |
| BibTeX references (references.bib) | DONE |

### Figures
| Figure | Description | File |
|--------|-------------|------|
| Fig 2 | Reliability diagrams (before/after calibration) | results/figures/fig2_reliability_diagrams.png |
| Fig 3 | F1 Bootstrap CI | results/figures/fig3_f1_bootstrap_ci.png |
| Fig 4 | SFAR by category | results/figures/fig4_sfar_by_category.png |
| Fig 5 | ECE vs SFAR scatter | results/figures/fig5_ece_vs_sfar.png |
| Fig 6 | Coverage vs SFAR (conformal frontier) | results/figures/fig6_coverage_vs_sfar.png |
| Fig 7 | Multi-hypothesis recovery waterfall | results/figures/fig7_multi_hypothesis_recovery.png |
| Fig 8 | Qualitative Case Studies (2x2 panel) | paper/figures/fig8_case_studies.pdf + .png |
| Fig 9 | Calibration Heatmap (4 methods x 9 conditions) | paper/figures/fig9_calibration_heatmap.pdf + .png |

---

## 2. Steps To Submit

### Step 1 — Install a LaTeX distribution
If you do not already have LaTeX:
- Windows: Install MiKTeX (https://miktex.org/) or TeX Live (https://tug.org/texlive/)
- Linux/macOS: sudo apt install texlive-full or MacTeX

### Step 2 — Compile the manuscript
```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex   # run twice to resolve cross-references
```

### Step 3 — Copy figures into paper/figures/
The paper already includes fig8 and fig9 in paper/figures/.
Copy earlier result figures if you want to include them:
```bash
copy results\figures\fig2_reliability_diagrams.png  paper\figures\
copy results\figures\fig3_f1_bootstrap_ci.png        paper\figures\
copy results\figures\fig4_sfar_by_category.png       paper\figures\
copy results\figures\fig5_ece_vs_sfar.png            paper\figures\
copy results\figures\fig6_coverage_vs_sfar.png       paper\figures\
copy results\figures\fig7_multi_hypothesis_recovery.png paper\figures\
```

### Step 4 — Personalise author information
Edit lines 29-32 in paper/main.tex:
```latex
\author{
  \textbf{Your Name} \\
  Institution / University Name \\
  \texttt{youremail@institution.edu}
}
```

### Step 5 — Anonymise for blind review (if required)
Most CVPR/NeurIPS submissions require anonymised initial submissions:
- Replace author name/institution with "Anonymous"
- Remove any self-citations that could de-anonymise
- Use venue-specific class file with blind review option

### Step 6 — Format per venue requirements
| Venue | Page Limit | Notes |
|-------|-----------|-------|
| CVPR/ICCV | 8 pages + refs | Portrait, two-column |
| NeurIPS | 9 pages + refs | Single-column |
| AAAI | 7 pages + refs | Two-column |
| ECCV | 14 pages + refs | Springer LNCS |

Replace the documentclass line with the venue-specific class file.

### Step 7 — Final proof checks
- All tables numbered sequentially and referenced in text
- All figures referenced in text
- Abstract is under 250 words (current: ~190 words)
- Page count within limits for chosen venue
- Run chktex main.tex for LaTeX linting

### Step 8 — Package for submission
```bash
zip visionresearch_submission.zip paper/main.tex paper/references.bib paper/figures/*.png paper/figures/*.pdf
```

---

## 3. Code & Reproducibility

```bash
# Clone and install
pip install -r requirements.txt

# Run all calibration experiments
python -m scripts.validate_calibration_all_methods

# Run evaluations
python -m src.evaluation.evaluate_advanced --model llava-1.5-7b --dataset mmvp
python -m src.evaluation.evaluate_paired --model llava-1.5-7b
python -m src.evaluation.evaluate_baselines --model llava-1.5-7b

# Verify test suite
pytest tests/ -v
```

All results are written to results/metrics/ as .csv, .md, and .tex files.

---

## 4. Key Numbers to Cite in Paper

| Metric | Value | Table |
|--------|-------|-------|
| ECE reduction (MMVP, LLaVA) | 0.5893 -> 0.0178 (x33) | Table 2 |
| Max Recovery Rate | 92.55% (InternVL3-8B MMVP) | Table 6 |
| MMVP Paired Acc gain | 0.0% -> 71.33% (InternVL3-8B) | Table 8 |
| SFAR bound (Hard-GT) | 15.2%–23.5% | Table 3 |
| Calibrated MaxSMT vs Greedy | 87.01% vs 68.93% (+18.08%) | Table 9 |
| Zero-SFAR coverage threshold | >= 80% | Figure 6 |
| Test pass rate | 165/165 = 100% | N/A |
| Temperature T range | 0.751–2.163 | Table 2 |
| Conformal qhat range | 0.930–1.000 | Table 7 |

---

## 5. Suggested Venue Fit

Given the combination of visual hallucination evaluation, calibration methodology, and neuro-symbolic integration:

Primary recommendation: CVPR 2026 (Computer Vision focus, high impact)
Alternative: ICLR 2026 (reliable ML and calibration focus)
Workshop option: NeurIPS 2025 "Reliable & Responsible Foundation Models" workshop

---

*Last updated: 2026-09-08. All 165 tests passing. All 9 calibration conditions validated.*

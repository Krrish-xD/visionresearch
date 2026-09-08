Searched for files: *.md
Searched for files: *
Viewed README.md:1-87
Viewed PROJECT.md:1-34
Viewed RESEARCH_SUMMARY_AND_ROADMAP.md:1-87
Searched for files: *
Viewed main.tex:1-100
Viewed main.tex:101-250
Viewed main.tex:250-294
Viewed RESEARCH_GAPS_AND_NEXT_STEPS.md:1-191
Viewed RESEARCH_EXECUTION_PLAN.md:1-120
Searched for files: *.py
Searched for files: *.py
Searched for files: *.yaml
Viewed experiment.yaml:1-51
Viewed models.yaml:1-52
Viewed verifier.py:1-100
Viewed metrics.py:1-60
Viewed selective.py:1-60
Viewed evaluate_advanced.py:1-80
Viewed evaluate_paired.py:1-60
Searched for files: *
Listed directory data
Searched for files: *
Listed directory visionresearch
Viewed HOWTO_PUBLISH.md:1-175
Searched for files: *
Viewed validate_calibration_all_methods.py:1-60
Searched for files: *
Searched for files: *
Ran command: `git log -n 5 --oneline`
Viewed .gitignore:1-52
Ran command: `git status`
Ran command: `git show --stat 2c64b75`
Viewed test_solver.py:1-60
Viewed test_multi_hypothesis.py:1-60

### 1. Executive Problem Statement

* **The Core Failure Mode:** Modern Vision-Language Models (VLMs) frequently hallucinate visual details—misidentifying spatial relations, hallucinating non-existent objects, miscounting, and failing subtle visual contrasts. Crucially, these perceptual errors are accompanied by **pathological overconfidence**: even when generating false assertions, models assign internal probability estimates clustering between 90% and 99% (yielding Expected Calibration Errors exceeding 50%–60%). They are systematically and confidently wrong.
* **The Neuro-Symbolic Dilemma:** To curb hallucinations, automated reasoning via Satisfiability Modulo Theories (SMT/MaxSMT) solvers (e.g., Z3) can check VLM claims against physical scene knowledge. However, combining deep learning with symbolic logic introduces an architectural bottleneck:
  1. *Unsound Soft-Weighting*: If physical scene facts are weighted softly alongside VLM claims in an optimization solver (MaxSMT), an overconfident hallucination will overpower the ground-truth facts. The solver will discard physical reality to satisfy the model’s error—a failure mode formalized in this project as a **Solver Over-Override (SOOR)** event.
  2. *Single-Claim Degeneracy*: If ground truth is asserted as immutable hard axioms, a single scalar prediction either trivially conflicts or satisfies the solver, rendering post-hoc confidence calibration meaningless unless candidate hypotheses compete or selective verification is applied.
* **Why the Problem Exists:** VLMs are optimized via token-level maximum likelihood over massive web-scraped corpora. Consequently, they lean on linguistic priors and memorized associations rather than genuine visual grounding. Because their raw softmax outputs do not reflect true posterior correctness, downstream logic engines cannot distinguish between a grounded perception and a confident hallucination.

---

### 2. Primary Objective & End Goal

* **Overarching Goal:** To establish a controlled, mathematically sound neuro-symbolic framework that tests whether **post-hoc confidence calibration** enables formal SMT solvers to detect visual contradictions, eliminate solver over-override, and actively recover from perceptual failures.
* **Nature of the Project:** This is an **empirical scientific research study and benchmarking suite** prepared for submission to a top-tier peer-reviewed venue (e.g., CVPR, NeurIPS, ECCV). 
* **Role of the Artifacts:** The accompanying codebase, FastAPI server, and React UI serve as a reproducible experimental harness, verification pipeline, and interactive audit tool rather than a standalone commercial product.

---

### 3. Core Hypotheses & Central Questions

* **Central Research Questions:**
  * **RQ1 (Error Detection):** Does recalibrating VLM confidence improve precision, recall, and F1 when identifying contradictions between model predictions and symbolic scene facts?
  * **RQ2 (False Acceptance):** Does post-hoc calibration systematically reduce the **Solver False-Accept Rate (SFAR)**—the rate at which false claims escape detection—especially in failure-prone domains like counting and spatial relations?
  * **RQ3 (Calibration Regimes):** How do parametric methods (Temperature Scaling), non-parametric methods (Isotonic Regression), and distribution-free methods (Conformal Risk Control / Adaptive Prediction Sets) compare in their utility for logic solver weighting?
  * **RQ4 (Active Error Recovery):** Can an SMT solver advance from a passive verification gate into an *active error corrector* when candidate alternatives ($N$-best hypotheses) are weighted by calibrated probabilities?
  * **RQ5 (Selective Verification):** Does calibrated confidence enable monotonic risk-coverage trade-offs, allowing systems to safely abstain from high-risk queries?

* **Primary Claims Proven / Disproven by the Project:**
  * **Soft MaxSMT is Scientifically Unsound:** Allowing uncalibrated VLM confidence to compete against soft ground-truth facts triggers SOOR events in 91%–100% of error cases. Benchmark reality must be enforced as immutable hard constraints.
  * **Isotonic Calibration Eliminates Overconfidence:** Non-parametric Isotonic Regression reliably collapses raw ECE from up to 61.3% down to 0.3%–7.3% across model families and datasets.
  * **Multi-Hypothesis MaxSMT Enables Active Recovery:** When framed as competing candidate claims under hard constraints, MaxSMT recovers up to 92.55% of VLM errors on challenging visual tasks, restoring paired visual consistency from 0.0% to over 71.3%.
  * **Selective Abstention Provides Strict Risk Bounds:** Calibrated confidence yields a monotonic Coverage vs. SFAR frontier, driving the false-acceptance rate to zero at actionable coverage thresholds (e.g., 80% coverage).

---

### 4. Success Criteria & Target Outcomes

* **Quantitative Success Metrics:**
  * **Calibration Quality:** Sharp reduction in Expected Calibration Error (ECE-10), Brier Score, and Negative Log-Likelihood (NLL).
  * **Verification Reliability:** Bounding SFAR under sound hard ground truth (empirically kept between 15.2% and 23.5%) and driving SOOR to 0% in verified operating modes.
  * **Error Recovery Rate:** Percentage of base VLM misclassifications corrected by $N$-best MaxSMT optimization.
  * **Paired Visual Consistency:** Performance on MMVP contrasting image pairs (measuring whether a model correctly answers *both* opposing visual conditions rather than exploiting text bias).
  * **Selective Coverage-Risk Frontier:** Monotonic decrease in SFAR as coverage is reduced, guaranteeing zero false acceptance at high coverage.
* **Final Deliverables:**
  * A camera-ready academic manuscript ([`paper/main.tex`](file:///c:/Users/KRRISHMOHTA-24589129/projects/visionresearch/paper/main.tex), BibTeX references, vector figures, and qualitative case studies).
  * An automated, multi-model evaluation pipeline and comprehensive unit test suite (165/165 passing tests).
  * Standardized empirical metric tables (CSV, Markdown, LaTeX) across 9 model $\times$ dataset conditions.
  * An interactive demo application (FastAPI backend + React frontend) demonstrating real-time visual claim formalization, confidence calibration, and SMT verification.

---

### 5. Scope & Target Domain

* **In Scope:**
  * **Model Families:** Modern open-weight Vision-Language Models capable of running within an edge/workstation 16 GB VRAM budget: [LLaVA-1.5-7B](file:///c:/Users/KRRISHMOHTA-24589129/projects/visionresearch/configs/models.yaml), [InternVL3-8B](file:///c:/Users/KRRISHMOHTA-24589129/projects/visionresearch/configs/models.yaml), and [Qwen2.5-VL-7B](file:///c:/Users/KRRISHMOHTA-24589129/projects/visionresearch/configs/models.yaml).
  * **Datasets & Tasks:** Authentic fine-grained visual discrimination on physical images ([MMVP](file:///c:/Users/KRRISHMOHTA-24589129/projects/visionresearch/configs/experiment.yaml)), compositional scene logic ([CLEVR](file:///c:/Users/KRRISHMOHTA-24589129/projects/visionresearch/configs/experiment.yaml)), and spatial visual question answering ([GQA](file:///c:/Users/KRRISHMOHTA-24589129/projects/visionresearch/configs/experiment.yaml)). Task types include multiple-choice selection, object counting, relational positioning, and binary existence queries.
  * **Calibration Methods:** Temperature Scaling, Isotonic Regression (PAVA), and Adaptive Prediction Sets (APS) Conformal Risk Control.
  * **Verification Logic:** First-order logic and MaxSMT constraint satisfaction executed via Microsoft Z3.
* **Explicitly Out of Scope:**
  * Pretraining or fine-tuning underlying vision encoders or language decoders (models are treated as frozen inference backbones).
  * Open-domain, unconstrained natural language dialog or text generation tasks without clear symbolic ground truth.
  * Novel SMT solver algorithm development (the project treats Z3 as a fixed black-box solver).
  * End-to-end unguided scene graph generation from scratch (the formalization layer operates over predefined predicate schemas and known benchmark ontologies).
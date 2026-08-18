# Calibrated Confidence for Symbolic Error Detection in Visual Grounding

## Objective

Build a reproducible empirical study that tests whether post-hoc calibration of VLM confidence improves symbolic detection of visual-grounding contradictions when VLM claims are checked with Z3/MaxSMT.

The publishable claim must be narrow:

> Given fixed VLMs, fixed benchmarks, fixed claim extraction, and fixed symbolic encoding, does calibrated confidence produce better contradiction detection and fewer solver-side false negatives than raw confidence or uniform weighting?

Do not claim a new VLM, a new solver, or a solved hallucination detector. The contribution is the controlled evaluation protocol, the calibration-to-weight analysis, and the resulting empirical evidence.

## Proposal-to-Paper Rule

Your proposal is not the paper. The proposal describes what you intend to do; the paper must report what you actually did.

| Proposal | Paper |
|---|---|
| Future tense: "we will test" | Past tense: "we tested" |
| No results required | Real numbers, tables, figures, and error analysis required |
| Can be broad and persuasive | Must be tight, factual, and reproducible |
| Conceptual figures are acceptable | Every figure must come from experiment logs |
| Contribution is the study design | Contribution is the result and what it proves or disproves |

Do not write the final introduction first. Build the pipeline, run the pilot, generate real tables, then write Methods, Experimental Setup, Results, Discussion, Introduction, and Conclusion in that order.

## Critical Correction to the Proposal

The proposal says an incorrect high-confidence VLM claim may be satisfied by the solver "at the cost of discarding a correct ground-truth fact." That is only possible if ground-truth facts are encoded as soft constraints.

For a sound verifier, benchmark ground truth must be hard. Z3 cannot drop a hard constraint. Therefore:

- Hard constraints: ontology, type rules, benchmark scene facts, benchmark answer facts.
- Soft constraints: VLM-generated claims and optional auxiliary model claims.
- Solver verdict: a VLM claim is contradicted if adding it conflicts with the hard fact base or if Optimize must leave the VLM soft claim unsatisfied.

Replace the original Solver Over-Override Rate with this measurable diagnostic:

## Solver False-Accept Rate

Solver False-Accept Rate (SFAR):

```text
SFAR = count(false VLM claim accepted by verifier) / count(false VLM claim)
```

A false claim is "accepted" when the verifier returns `consistent` or keeps the VLM claim satisfied despite benchmark ground truth showing it is wrong.

If you want to study over-override specifically, add a separate experimental ablation where ground-truth facts are intentionally encoded as soft anchor constraints. That ablation should be labeled unsafe and diagnostic only, not the main verifier.

## Research Questions

RQ1. Does post-hoc calibration improve contradiction-detection precision, recall, and F1 compared with raw confidence weighting?

RQ2. Does calibration reduce SFAR, especially on counting and spatial-relation failures?

RQ3. Which calibration method is most useful for solver weighting: temperature scaling, isotonic regression, or conformal risk weighting?

RQ4. Is better calibration, measured by ECE and Brier score, actually correlated with better symbolic verification?

## Hypotheses

H1. Raw VLM confidence will be miscalibrated and overconfident on visually difficult categories.

H2. Calibrated confidence will reduce SFAR relative to raw confidence.

H3. Improvements will be strongest on counting, orientation, and spatial-relation categories, where VLMs often answer confidently but incorrectly.

H0. Calibration will not significantly change downstream symbolic verification performance. This is still publishable if the experiment is rigorous, because it shows MaxSMT verification is insensitive to confidence calibration under the tested encoding.

## Experimental Conditions

Run every dataset item under the same extraction and solver pipeline, changing only the weight function.

| Condition | Weight | Purpose |
|---|---:|---|
| `hard_claim_check` | VLM claim temporarily asserted hard | Pure logical contradiction oracle |
| `uniform_soft` | `w = 1.0` | MaxSMT baseline without confidence |
| `raw_confidence` | `w = p_raw` | Direct confidence weighting |
| `temperature_scaled` | `w = p_temp` | Parametric calibration |
| `isotonic` | `w = p_iso` | Non-parametric calibration |
| `conformal_risk` | `w = 1 - risk_bound` | Distribution-free risk weighting |

The primary comparison is `raw_confidence` vs each calibrated condition.

## What to Use

### Language and Runtime

- Python 3.10 or 3.11.
- CUDA-enabled PyTorch for VLM inference.
- One RTX-class GPU is enough for 7B models with 4-bit quantization. Use cloud GPU only if local VRAM is below 8 GB.

### Core Libraries

- `torch`
- `transformers`
- `accelerate`
- `bitsandbytes`
- `z3-solver`
- `scikit-learn`
- `netcal`
- `numpy`
- `pandas`
- `scipy`
- `statsmodels`
- `datasets`
- `opencv-python`
- `pillow`
- `tqdm`
- `pyyaml`
- `matplotlib`
- `seaborn`

### Models

Start with one primary modern model, then add one stable baseline family for comparison.

1. `OpenGVLab/InternVL3-8B` as the primary model for the main paper.
2. `llava-hf/llava-1.5-7b-hf` as the stable baseline for reproducibility and hardware accessibility.
3. If hardware and dependency stability allow, replace or supplement the LLaVA baseline with a LLaVA-NeXT checkpoint such as `llava-hf/llava-v1.6-mistral-7b-hf`.

Do not use proprietary APIs for the main result because you need token logits and reproducibility.

Do not use Ollama for the main experiments. Ollama is useful for chatting with local models, but it does not reliably expose the token-level logits needed for calibration. Run VLMs directly through Hugging Face Transformers with `return_dict_in_generate=True` and `output_scores=True`.

Important compatibility note:

- `InternVL3` requires `trust_remote_code=True` and is optimized for visual grounding tasks with improved reasoning capabilities. Confirm the model loads cleanly before freezing the environment for the team.
- `LLaVA-1.5` is older, but it remains useful as a strong reproducible baseline because the tooling is mature and many peers can run it locally.

### Datasets

Use datasets with machine-checkable answers.

1. MMVP: main stress test for CLIP-blind visual grounding.
2. CLEVR validation subset: clean symbolic scene graphs and functional programs.
3. GQA balanced subset: real-image scene graphs and spatial/object relations.

Recommended first publishable scope:

- MMVP full set.
- CLEVR 1,000-item subset filtered to counting, existence, attribute, and spatial questions.
- GQA 1,000-item subset filtered to binary spatial/object-relation questions.

Scale only after the pipeline is correct.

## Repository Structure

Create this structure:

```text
visionresearch/
  README.md
  RESEARCH_EXECUTION_PLAN.md
  requirements.txt
  configs/
    experiment.yaml
    models.yaml
  data/
    raw/
    processed/
    splits/
  src/
    datasets/
      mmvp.py
      clevr.py
      gqa.py
    vlm/
      inference.py
      confidence.py
      prompts.py
    formalization/
      schema.py
      parser.py
      validators.py
    solver/
      z3_encoder.py
      verifier.py
      diagnostics.py
    calibration/
      temperature.py
      isotonic.py
      conformal.py
      metrics.py
    evaluation/
      run_experiment.py
      stats.py
      tables.py
      plots.py
  experiments/
    pilot/
    main/
  results/
    raw_predictions/
    solver_outputs/
    metrics/
    figures/
  paper/
    manuscript.md
    figures/
    tables/
```

## Data Schema

Every benchmark item must be converted into one JSONL row:

```json
{
  "item_id": "mmvp_0001",
  "dataset": "mmvp",
  "image_path": "data/raw/mmvp/images/0001.jpg",
  "question": "How many chair legs are visible?",
  "answer_type": "count",
  "gold_answer": "3",
  "gold_facts": [
    {"predicate": "count", "subject": "chair_leg", "value": 3}
  ],
  "category": "counting"
}
```

VLM outputs must be stored separately:

```json
{
  "item_id": "mmvp_0001",
  "model": "llava-1.5-7b",
  "prompt_id": "closed_answer_v1",
  "raw_answer": "4",
  "claim": {"predicate": "count", "subject": "chair_leg", "value": 4},
  "raw_confidence": 0.93,
  "token_logprobs": [-0.072],
  "is_correct": false
}
```

Never overwrite raw model outputs. All calibration and solver results should be derived artifacts.

The final per-item result table must include enough detail to audit every decision:

```csv
item_id,dataset,category,question,gold_answer,raw_answer,normalized_answer,
raw_confidence,temp_confidence,isotonic_confidence,conformal_weight,
hard_claim_verdict,uniform_verdict,raw_verdict,temp_verdict,isotonic_verdict,conformal_verdict,
hard_claim_correct,uniform_correct,raw_correct,temp_correct,isotonic_correct,conformal_correct,
sfar_uniform,sfar_raw,sfar_temp,sfar_isotonic,sfar_conformal,
parse_status,solver_status,solve_time_ms,seed,model,prompt_id
```

Every crash, timeout, parse failure, skipped item, and normalization decision must be logged. Silent dropping is not allowed.

## Prompting Protocol

Use constrained prompts so parsing is reliable.

For count:

```text
Answer with only one integer.
Question: {question}
```

For yes/no relation:

```text
Answer with only yes or no.
Question: {question}
```

For attribute:

```text
Answer with only the attribute word.
Question: {question}
```

Use temperature `0`, greedy decoding, and fixed image preprocessing. Save exact prompts in `src/vlm/prompts.py`.

## Confidence Extraction

For single-token answers:

```text
p_raw = softmax(logits_t)[generated_token]
```

For multi-token answers:

```text
p_raw = exp(mean(log p(token_i | previous tokens)))
```

Use Hugging Face generation with `return_dict_in_generate=True`, `output_scores=True`, and, where available, `output_logits=True`.

Important:

- Use the probability of the answer token(s), not the model's verbalized confidence.
- For yes/no tasks, force the answer vocabulary to `yes` and `no` if the model backend allows it.
- Record whether an answer required normalization, such as `"four"` to `4`.

## Symbolic Claim Schema

Support only these predicates in version 1:

```text
exists(object)
count(object) = integer
attribute(object, attribute_type) = value
relation(object_a, relation_type, object_b)
```

Do not attempt unrestricted natural-language autoformalization in the first paper. A constrained schema makes the result defensible.

## Z3 Encoding

Use hard benchmark facts and soft VLM claims.

Example:

```python
from z3 import Int, Optimize

chair_legs = Int("count_chair_leg")
opt = Optimize()
opt.add(chair_legs == 3)                         # hard benchmark fact
opt.add_soft(chair_legs == 4, weight="0.93")      # VLM claim
status = opt.check()
model = opt.model()
claim_satisfied = model.eval(chair_legs == 4)
```

For direct contradiction testing, also run a hard-claim check:

```python
from z3 import Solver

s = Solver()
s.add(chair_legs == 3)
s.add(chair_legs == 4)
verdict = s.check()  # unsat means contradicted
```

Use the hard-claim check as the logical correctness oracle. Use MaxSMT to test whether confidence-weighted claim sets improve final verification behavior when multiple claims or auxiliary facts compete.

## Calibration

Split each dataset by item:

- 40% calibration
- 60% evaluation

Keep splits fixed across all methods and models. Stratify by dataset and category.

Reason: MMVP is small. A 20% calibration split gives only about 60 examples, which is weak for isotonic regression and category-level analysis. Use 40/60 for the first paper, then report a sensitivity check with 20/80 or cross-validation if time allows.

### Temperature Scaling

Fit scalar `T > 0` on calibration items by minimizing NLL.

```text
p_temp = softmax(logits / T)
```

Use this first because it is simple and low variance.

### Isotonic Regression

Fit monotonic mapping:

```text
p_iso = g(p_raw)
```

Use `sklearn.isotonic.IsotonicRegression` or `netcal.binning.IsotonicRegression`. Require at least 200 calibration examples before trusting it.

### Conformal Risk Weighting

Use split conformal calibration with nonconformity:

```text
score = 1 - p_true
```

At miscoverage `alpha = 0.1`, estimate a risk threshold on the calibration split. Convert uncertainty into weight:

```text
w = max(0.05, 1 - conformal_risk)
```

Report conformal results separately from probability calibration. They answer different uncertainty questions.

## Metrics

Primary verification metrics:

- Contradiction precision
- Contradiction recall
- Contradiction F1
- SFAR

Calibration metrics:

- Expected Calibration Error with 10 and 15 bins
- Adaptive Calibration Error
- Brier score
- Negative log-likelihood

Solver metrics:

- Mean solve time
- p95 solve time
- Number of hard constraints
- Number of soft constraints
- Number of unsatisfied VLM claims

Category metrics:

- Count
- Existence
- Attribute
- Spatial relation
- MMVP category, where available

Baseline metrics:

- Raw VLM answer accuracy with no symbolic verifier.
- Hard-claim logical contradiction oracle.
- Optional external non-Z3 baseline: a lightweight self-consistency verifier using 3 to 5 answer samples per item, reported separately from the main MaxSMT comparison.

The non-Z3 baseline is useful for reviewers. It shows whether the symbolic pipeline adds value beyond simply asking the model multiple times.

## Statistical Analysis

Use paired tests because every method is evaluated on the same items.

- McNemar's test for paired contradiction-detection success/failure.
- Stratified bootstrap with 10,000 resamples for F1 confidence intervals.
- Wilcoxon signed-rank test for paired confidence/calibration score differences.
- Benjamini-Hochberg correction for category-level comparisons.
- Report effect sizes, not only p-values.

Minimum publishable reporting:

```text
metric ± 95% CI
p-value after correction
effect size
number of examples
model
dataset
category
```

## Implementation Phases

### Phase 1: Reproducible Skeleton

Deliverables:

- `requirements.txt`
- config files
- dataset loaders
- JSONL schemas
- deterministic split generator

Success check:

- Running one command creates `data/processed/*.jsonl` and `data/splits/*.json`.

### Phase 2: VLM Inference

Deliverables:

- Qwen2.5-VL inference script
- one baseline inference script from the LLaVA family
- confidence extraction
- raw prediction cache
- answer normalization

Success check:

- Run inference on 20 MMVP examples.
- Each row has answer, normalized answer, token logprobs, raw confidence, and correctness.

### Phase 3: Formalization

Deliverables:

- deterministic claim parser for count, yes/no, attribute, relation
- schema validator
- parse-failure report

Success check:

- At least 95% parse success on constrained benchmark questions.
- Failed parses are logged, not silently dropped.

### Phase 4: Z3 Verifier

Deliverables:

- hard contradiction oracle
- MaxSMT weighted verifier
- solver diagnostics

Success check:

- Unit tests prove that contradictory count, attribute, and relation claims return expected verdicts.

### Phase 5: Calibration

Deliverables:

- temperature scaling
- isotonic regression
- conformal risk weighting
- ECE/Brier/NLL metrics

Success check:

- Reliability plots show raw vs calibrated confidence on calibration and evaluation splits.

### Phase 6: Pilot Experiment

Deliverables:

- 50 MMVP items
- all conditions
- one model
- first tables and figures

Success check:

- End-to-end command produces `results/metrics/pilot_summary.csv`.
- Manually inspect 20 randomly selected solver decisions.

### Phase 7: Main Experiment

Deliverables:

- MMVP full set
- CLEVR subset
- GQA subset
- one or two VLMs
- final metrics, plots, logs

Success check:

- All results are reproducible from configs and cached raw predictions.

### Phase 8: Analysis and Figures

Deliverables:

- reliability diagrams
- main result table
- SFAR bar chart
- per-category heatmap
- ECE-vs-SFAR scatter plot
- two to four case studies, including at least one failure where calibration did not help

Success check:

- Every figure is generated by a script from logged CSV/JSONL files.
- No figure is illustrative in the final paper.

### Phase 9: Paper

Deliverables:

- abstract
- introduction
- related work
- method
- experiments
- results
- limitations
- reproducibility appendix

Success check:

- Every table is generated from `results/metrics/*.csv`.
- Every claim in the paper points to a logged experiment.

## Commands to Build Toward

Target final workflow:

```bash
python -m src.datasets.prepare --config configs/experiment.yaml
python -m src.evaluation.make_splits --config configs/experiment.yaml
python -m src.vlm.inference --config configs/experiment.yaml --model qwen2.5-vl-7b
python -m src.calibration.fit --config configs/experiment.yaml --model qwen2.5-vl-7b
python -m src.evaluation.run_experiment --config configs/experiment.yaml --model qwen2.5-vl-7b
python -m src.evaluation.tables --results results/metrics/main.csv
python -m src.evaluation.plots --results results/metrics/main.csv
```

## Expected Tables

Table 1. Dataset composition by dataset, category, and split.

Table 2. Calibration metrics by model and method.

Table 3. Contradiction detection metrics by model and method.

Table 4. Category-level SFAR and F1.

Table 5. Solver runtime and constraint scaling.

## Expected Figures

Figure 1. Pipeline diagram.

Figure 2. Reliability diagrams: raw vs temperature vs isotonic.

Figure 3. F1 difference with 95% bootstrap CI.

Figure 4. SFAR by category.

Figure 5. Calibration error vs solver false-accept rate.

Figure 6. Case studies showing image, question, VLM answer, raw confidence, calibrated confidence, Z3 facts, and solver verdict.

## Paper Writing Plan

Use this structure once real results exist:

1. Abstract: problem, gap, method, datasets/models, strongest numerical result, implication.
2. Introduction: VLM hallucination, solver verification, confidence miscalibration, gap, contributions.
3. Related Work: visual grounding failures, calibration, neuro-symbolic verification, conformal uncertainty.
4. Method: formal problem statement, confidence extraction, calibration methods, Z3 encoding, SFAR, algorithm.
5. Experimental Setup: datasets, models, prompts, splits, conditions, metrics, statistics, hardware.
6. Results: main table, calibration quality, SFAR, per-category effects, runtime, case studies.
7. Discussion: why the result happened, where calibration helps, where it fails, what this means for verifier design.
8. Threats to Validity: extraction errors, benchmark limits, SFAR construct validity, sample size.
9. Conclusion: one paragraph restating the empirical answer.

Writing rules:

- State numbers before interpretations.
- Do not write "calibration helped"; write the exact delta, confidence interval, and p-value.
- Show at least one negative or ambiguous result.
- Never say "guaranteed hallucination removal" or "trustworthy VLM" without qualification.
- Use past tense after experiments are complete.

## Publication Strategy

Best first target:

- ACL Student Research Workshop
- EMNLP Student Research Workshop
- NeurIPS Datasets and Benchmarks workshop
- CVPR/ICCV/ECCV workshop on trustworthy multimodal AI
- AAAI student abstract or workshop

A full main-conference submission is possible only if the results are strong and the protocol includes at least two models and multiple datasets.

## Twelve-Week Execution Timeline

| Week | Target |
|---:|---|
| 1 | Environment, repo skeleton, configs, logging format |
| 2 | MMVP loader, pilot split, schema validators |
| 3 | VLM inference through Transformers, confidence extraction |
| 4 | Claim parser, normalization, hard Z3 contradiction oracle |
| 5 | MaxSMT verifier, solver diagnostics, unit tests |
| 6 | Calibration methods and calibration metrics |
| 7 | 50-item pilot, manual audit, fix parser/solver failures |
| 8 | Full MMVP run and first results table |
| 9 | CLEVR/GQA subset run and statistical analysis |
| 10 | Figures, case studies, ablations, reproducibility checks |
| 11 | Draft Methods, Setup, Results, Discussion |
| 12 | Draft Introduction/Related Work/Conclusion, advisor review, venue formatting |

If hardware or dataset downloads slow things down, preserve the pilot and MMVP full run first. A clean single-dataset workshop paper is better than a half-broken three-dataset paper.

## Risk Register

| Risk | Why it matters | Mitigation |
|---|---|---|
| Claim parsing fails | Bad parser can dominate results | Use constrained prompts and limited schema |
| Confidence extraction differs by model | Invalid cross-model comparison | Store raw logits/scores and document extraction |
| MaxSMT weighting has no effect with one claim | Calibration cannot help if there is no choice | Include multi-claim outputs or auxiliary candidate claims for MaxSMT experiments |
| MMVP is small | Low statistical power | Add CLEVR/GQA subsets and report confidence intervals |
| Ground truth accidentally encoded soft | Unsound verifier | Unit-test hard fact immutability |
| Isotonic overfits | Non-parametric calibration needs data | Compare on held-out eval only and report calibration-set size |
| Results are null | May feel disappointing | Frame as evidence against a common assumption |
| Ollama hides logits | Calibration cannot be computed correctly | Use Hugging Face Transformers for main experiments |
| Missing non-symbolic baseline | Reviewers may say Z3 adds no value | Add raw VLM accuracy and optional self-consistency |
| Figures are hand-made | Reproducibility risk | Generate every table/figure from scripts |

## Key Engineering Rule

Separate these four layers completely:

1. Raw VLM prediction.
2. Parsed symbolic claim.
3. Calibrated confidence.
4. Solver verdict.

Never regenerate predictions during solver experiments. This makes every comparison paired, reproducible, and publishable.

## First Task List

1. Create repository skeleton.
2. Add `requirements.txt`.
3. Implement JSONL schema and validators.
4. Implement MMVP loader.
5. Implement 20-item pilot split.
6. Implement LLaVA inference with token confidence extraction.
7. Implement hard Z3 contradiction oracle.
8. Implement MaxSMT soft-claim verifier.
9. Add unit tests for count, attribute, existence, and relation claims.
10. Run pilot and inspect results manually.

## Examiner-Proof Checklist

Before submission, every answer must be yes:

| Check | Required answer |
|---|---|
| Can someone reproduce the experiment from Method and configs? | Yes |
| Are calibration and evaluation splits completely separate? | Yes |
| Are prompts identical across conditions? | Yes |
| Are raw predictions cached and reused across solver conditions? | Yes |
| Are all random seeds logged? | Yes |
| Are confidence intervals reported? | Yes |
| Are effect sizes reported? | Yes |
| Is there a raw VLM baseline and at least one non-Z3 comparison or ablation? | Yes |
| Are parse failures, solver timeouts, and skipped examples counted? | Yes |
| Are negative results reported honestly? | Yes |
| Are all figures generated from real experiment outputs? | Yes |
| Does the paper explain that deployment lacks benchmark ground truth? | Yes |

## Sources Checked

- Z3 Optimize API and `add_soft`: https://z3prover.github.io/api/html/classz3py_1_1_optimize.html
- Z3 soft constraints guide: https://microsoft.github.io/z3guide/docs/optimization/softconstraints/
- `z3-solver` package: https://pypi.org/project/z3-solver/
- Hugging Face generation scores/logits: https://huggingface.co/docs/transformers/main/internal/generation_utils
- netcal calibration framework: https://efs-opensource.github.io/calibration-framework/
- MMVP / Eyes Wide Shut summary and code pointer: https://paperswithcode.com/paper/eyes-wide-shut-exploring-the-visual
- CLEVR official dataset page: https://web.eecs.umich.edu/~justincj/clevr/
- GQA official dataset page: https://cs.stanford.edu/people/dorarad/gqa/
- LLaVA 1.5 model card: https://huggingface.co/llava-hf/llava-1.5-7b-hf
- Qwen2-VL record: https://dblp.org/rec/journals/corr/abs-2409-12191
- Qwen2.5-VL model card: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- LLaVA-NeXT documentation: https://huggingface.co/docs/transformers/main/model_doc/llava_next

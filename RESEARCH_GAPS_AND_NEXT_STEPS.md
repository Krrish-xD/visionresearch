# Research gaps, validity issues, and recommended next steps

Written after a full read of the execution plan, pipeline code, datasets, cached predictions, and current results. This is a working research notebook, not a paper draft.

## Where the work is actually stuck

The repository looks like Phase 7–8 (full experiments, tables, figures). Scientifically it is stuck at the **end of Phase 1 / broken Phase 2**.

| Layer | Status | Why it matters |
|---|---|---|
| Repo skeleton, schemas, splits, unit tests | Built | Fine as engineering |
| MMVP images + processed JSONL | Present (300 items) | Only real visual benchmark currently in the repo |
| CLEVR / GQA processed JSONL | **Synthetic** questions and gold facts; **no real images** | Any accuracy/SFAR on these datasets is not a visual-grounding result |
| LLaVA-1.5-7B weights | Downloaded | Usable baseline |
| InternVL3-8B weights | Downloaded (~16 GB, 4 shards) | Primary model in the plan |
| Qwen2.5-VL-7B weights | Downloaded (~16 GB, 5 shards) | Planned second modern model; **no predictions yet** |
| InternVL MMVP predictions | 3 rows, **all empty answers**, `raw_confidence=1.0`, parse failed | Inference path is broken; do not run the full benchmark |
| LLaVA MMVP predictions | 300 real generations | Usable once parsing and solver encoding are fixed |
| LLaVA CLEVR/GQA predictions | 1000+1000 rows, high accuracy (~74–77%) | Almost certainly answering **text-only** because images are missing |
| Tables/figures in `results/` | Exist | Last write is from LLaVA; **table3 is overwritten per run**, so it is not a multi-model paper table |
| `GEMINI.md` / `PROJECT.md` | Mark the project “complete” | That was a **backend/mock audit**, not the empirical study |

**Do not treat existing Table 2–5 or figures as paper results.** They mix (a) a scientifically unsound MaxSMT encoding, (b) MMVP parse failures, and (c) synthetic CLEVR/GQA with no images.

---

## Exact place to start (ordered)

Do this before any more full-dataset sweeps.

### Step 0 — Freeze the scientific contract (one meeting / one decision)

The execution plan and the live solver **disagree**.

- Plan (`RESEARCH_EXECUTION_PLAN.md`): ground-truth facts are **hard**. Soft constraints are VLM claims only. SFAR is the main diagnostic. SOOR (dropping GT) is an **unsafe ablation**, not the main verifier.
- Code (`src/solver/verifier.py`): GT is **soft** with `gt_weight=500`, VLM weight is `round(p * 1000)`. Overconfident false claims **override GT**. Tests even **require** this SOOR behavior.

Until this is decided, every new table is uninterpretable.

**Recommended default (defensible for a workshop paper):**

1. Main experiment: GT hard, VLM claim soft (or temporarily hard in `hard_claim_check`).
2. Keep current soft-GT MaxSMT as a clearly labeled **unsafe ablation**.
3. Report SFAR on the main (hard-GT) verifier.

If we keep soft-GT as the *main* result, reviewers who know SMT will correctly say the verifier is unsound: the “world” can be rewritten to match a confident hallucination.

### Step 1 — Make InternVL3 produce non-empty answers + real logprobs (3–20 MMVP items)

This is the current engineering blocker for the **primary** model.

Evidence: `results/raw_predictions/internvl3-8b_mmvp.jsonl` has empty `raw_answer` and dummy confidence `1.0` (the fallback in `extract_token_confidence` when `scores` is empty).

Likely failure modes to debug first (do not rewrite the whole engine until one of these is confirmed):

- Vision tokens not actually written into `inputs_embeds` (boolean-index assignment on a flattened tensor is easy to get wrong in PyTorch).
- Conversation template / `<IMG_CONTEXT>` count mismatch with `num_image_token * num_patches`.
- `language_model.generate(inputs_embeds=...)` returning empty or EOS-only sequences.
- `evaluate.py` catching a load error and silently falling back to mock — but empty answers look more like a live generate path that produced no tokens, not the mock path (mock always emits an answer).

**Success check:** 20 MMVP items with non-empty answers, token logprobs, `raw_confidence ∈ (0,1)`, and parse rate logged. Then stop.

### Step 2 — 50-item MMVP pilot on **one** working model (LLaVA first if InternVL is still broken)

LLaVA already has 300 cached MMVP predictions. Do **not** regenerate them. Fix parser + solver, then re-run **only** calibration/solver/tables on the cache.

Pilot audit: 20 random solver decisions by hand (image, question, gold, claim, weights, Z3 verdict).

### Step 3 — Full MMVP only, then decide whether CLEVR/GQA enter the first paper

A clean MMVP workshop paper is stronger than three datasets of which two are fake scenes.

---

## Validity bugs that will invalidate the paper if left in

### 1. Soft ground truth in the main MaxSMT loop (soundness)

See `verify_with_maxsmt`. This is the largest scientific defect. Calibration “helping” in current Table 3 is largely **changing whether a false claim out-weights GT (500 vs p×1000)**. That is not “symbolic error detection of a hallucinated claim against a fact base”; it is **priority between two soft assertions**.

Also: with a **single** VLM claim, MaxSMT has almost no interesting choice once GT is hard. The plan already flags this risk. The honest main result for one-claim items is close to the hard oracle; calibration can only matter if we add **competing claims** (n-best, yes/no both, or an auxiliary captioner).

### 2. CLEVR and GQA are not the official datasets

`src/datasets/clevr.py` and `gqa.py` sample random questions and gold answers. Image paths point at files that are not in `data/raw/` (only `data/raw/mmvp` exists). LLaVA “accuracy” ~77% / ~74% is therefore **not** visual grounding.

If we want those datasets: download official CLEVR val + GQA balanced questions, filter to checkable types, and bind `gold_facts` to **scene graphs**, not RNG.

### 3. MMVP answer-type / schema mismatch (parser contamination)

Example (`mmvp_0001`): options `(a) Open (b) Closed`, `answer_type: "relation"` because the question contains “closer”, gold fact is a relation with `value: "(a)"`. LLaVA answers `"Open"` → parse **failed**. Constrained choice prompts are not used.

112 / 300 LLaVA MMVP items failed to parse. Failed parses are treated as contradicted in the experiment loop. That inflates contradiction metrics and mixes **extraction error** with **solver error**.

Fix: if `options` is non-empty, force `answer_type: choice` and parse letters **or** option text (`open`/`closed`). Do not copy gold subject as `item_N` unless that is the intended symbolic object.

### 4. Temperature scaling is not scaling the answer-token softmax

`extract_token_confidence` stores **top-k logits**, not the full vocab distribution. `TemperatureScaling` then treats index 0 as “correct class” and index 1 as “wrong class”. That is not Guo et al. temperature scaling on the predictive distribution. ECE/Brier numbers in Table 2 are therefore **not comparable to the calibration literature** until we either:

- store the chosen-token logit plus a proper binary construction `logit_yes = log p_raw`, or
- store the full (or answer-vocab) logit vector and the actual generated token index.

Isotonic on `p_raw` vs correctness is still valid as a **binary** calibrator. Prefer it as the nonparametric method; fix temperature to a binary NLL on `logit(p_raw)`.

### 5. Cross-condition tables overwrite

`table2_*.csv` / `table3_*.csv` are global filenames. The last pipeline run wins. Need `results/metrics/{model}_{dataset}/` or prefixed files, plus a combiner for the paper.

### 6. Silent mock fallback

`run_predictions_on_dataset` prints a warning and continues if weights fail to load. A paper run must **abort** if `mock=False` and the model did not load.

### 7. InternVL custom generate vs logits

InternVL’s public `generate` does not expose scores. The custom `inputs_embeds` path is the right idea for this paper (we need logits). It is also the highest-risk code. Until smoke tests pass, InternVL results must not enter any table.

---

## Research gaps worth pursuing (beyond the current plan)

These are optional follow-ups. They would make a stronger paper **after** the main protocol is valid. They are not excuses to expand scope this week.

### G1. Single-claim MaxSMT is almost degenerate (high priority scientifically)

With one soft claim and hard GT, calibrated weight cannot change SAT/UNSAT of a **logical contradiction**. Calibration can only change behavior if:

- multiple mutually inconsistent VLM claims compete, or
- GT is soft (unsound), or
- we use weights for **selective verification** (abstain / don’t even query the solver below a threshold).

**Better main RQ variant:** does calibration improve *which claims we submit* or *how we rank n-best candidates*, not “does T change Z3 on one equality.”

A clean extra condition: **selective SFAR** — only verify claims with calibrated p ≥ τ; report coverage vs SFAR. That is closer to how a real system would use calibration.

### G2. Token likelihood ≠ visual-grounding correctness (known, still under-used here)

VLM `p(token)` is often high for frequent words (`yes`, `(a)`, counts). MMVP is designed so CLIP-like models fail on subtle visual pairs. Expect **severe overconfidence**; H1 should be tested with **pair-consistency**: same question, two paired images, conflicting gold answers. Calibration that ignores pairing leaves a lot on the table.

**Possible extra analysis:** ECE and SFAR on MMVP **pairs**, not i.i.d. items. If the model answers `(a)` for both images with p>0.8, that is a structured failure the current i.i.d. split hides.

### G3. Claim extraction is the hidden bottleneck

Neuro-symbolic papers often fail at autoformalization, not at Z3. We currently leak gold structure into the claim (`subject` copied from `gold_facts`). That makes the solver look better than a deployed system that cannot see gold.

**Honest limitation to write later:** this paper studies *confidence weighting given an almost gold-aligned schema*, not open-domain autoformalization.

**Optional stronger experiment:** parse claims **without** reading gold facts (subject/relation from question only). Report the delta. If F1 collapses, that is a publishable negative about formalization, not a failed project.

### G4. Binary vs multiclass calibration for constrained decoding

Yes/no and (a)/(b) should use **forced decoding over the answer vocabulary** (plan already says this). Geometric mean of all generated tokens (including EOS) dilutes p_raw. This is a methods fix and a small ablation: `p_first_content_token` vs `exp(mean log p)`.

### G5. Conformal “risk weighting” vs conformal **decision**

Current conformal path turns a nonconformity score into a MaxSMT weight. Reviewers from uncertainty quantification will ask for **coverage**: does the prediction set at α=0.1 contain the true answer on the eval split? If we cannot show coverage, do not call it conformal calibration in the paper title; call it a heuristic transform of nonconformity scores.

A higher-value paper add-on: **conformal abstention** (answer set empty or size>1 → do not assert a hard/soft claim). That is closer to Angelopoulos/Romano-style risk control than reweighting Z3.

### G6. Deployment threat the plan already names

At test time there is **no gold fact base**. The current pipeline is an **oracle-in-the-loop diagnostic**: “if we had GT, would calibrated weights help the solver catch errors?” That is still publishable if we say so in the introduction and limitations.

**Follow-up research (next paper, not this one):** replace GT hard facts with an independent perception module (detector + scene graph) and measure SFAR against **human** labels. Different problem.

### G7. Self-consistency / n-best as the missing non-Z3 baseline

The plan asks for this; it is not implemented. Cheap and reviewer-friendly: 5 samples at T>0 (separate from greedy main run), majority vote, compare F1/SFAR to Z3. Keep greedy+logits as the calibration path so we do not mix stochastic decoding into the main table.

---

## Recommended scope for the first paper

Keep the claim narrow:

> Given fixed VLMs, fixed MMVP items, fixed constrained extraction, and a fixed Z3 encoding with **immutable benchmark facts**, does post-hoc calibration of token-level confidence change contradiction detection and SFAR relative to raw confidence and uniform weights?

Drop CLEVR/GQA from v1 unless official data is wired. Add InternVL and/or Qwen only after smoke tests. Treat LLaVA as the reproducibility baseline.

If the main (hard-GT, one-claim) result is **null**, that is still a paper: it falsifies the assumption that better ECE automatically yields better MaxSMT verification. Current Table 3 must not be used to claim the opposite, because of the soft-GT confound.

---

## Questions that change the start date

1. First submission: **MMVP-only** vs wait for official CLEVR+GQA?
2. Confirm main verifier: **hard GT** (plan) vs keep SOOR-as-main (current code)?
3. Must InternVL3-8B appear in v1, or is LLaVA + Qwen enough if InternVL generate stays fragile?

Until (1) and (2) are answered, the only code that should run is InternVL/Qwen smoke tests and parser/encoding fixes — not another full 2300-item sweep.

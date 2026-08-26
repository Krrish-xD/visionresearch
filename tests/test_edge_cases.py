"""
Comprehensive Edge-Case Unit Tests for VisionResearch.

Covers:
  1. Unparseable & Malformed VLM String Outputs (empty, gibberish, code blocks, essays).
  2. Z3 SMT Edge Cases (empty facts, timeouts, contradictory constraints).
  3. Calibration Split Edge Cases (0% errors, 100% errors, 2 samples, identical confidences).
  4. Statistical Functions Edge Cases (McNemar with 0 discordant, Cohen's h edge values, FDR).
"""

import unittest
import numpy as np

from src.formalization.parser import parse_vlm_answer_to_claim, normalize_text
from src.calibration.temperature import TemperatureScaling, softmax
from src.calibration.isotonic import IsotonicCalibrator
from src.calibration.conformal import ConformalRiskWeighting, AdaptivePredictionSets
from src.calibration.metrics import compute_ece, compute_brier_score, compute_nll
from src.solver.verifier import check_hard_contradiction, verify_with_maxsmt
from src.solver.z3_encoder import encode_fact_to_z3
from src.evaluation.stats import (
    mcnemar_test,
    compute_paired_contingency,
    cohens_h,
    wilcoxon_paired_test,
    apply_benjamini_hochberg
)

class TestEdgeCases(unittest.TestCase):

    # -------------------------------------------------------------------------
    # 1. Unparseable and Malformed VLM Outputs
    # -------------------------------------------------------------------------
    def test_unparseable_empty_and_gibberish(self):
        gold_facts = [{"predicate": "count", "subject": "chair", "value": 3}]
        malformed_answers = [
            "",
            "   ",
            "###!@#$$%^&*",
            "I am an AI assistant and I cannot view images.",
            "```python\nprint('hello')\n```",
            "There are perhaps roughly 3 or maybe 4 or 5 chairs depending on perspective."
        ]

        for raw_ans in malformed_answers:
            claim, norm_ans, status = parse_vlm_answer_to_claim(
                raw_answer=raw_ans,
                answer_type="count",
                question="How many chairs are there?",
                gold_facts=gold_facts
            )
            # Must handle gracefully without crashing
            self.assertIn(status, ["success", "failed", "unparsed"])
            self.assertIsInstance(norm_ans, str)

    def test_malformed_choice_and_yes_no(self):
        # Invalid multiple choice options
        claim_c, norm_c, status_c = parse_vlm_answer_to_claim(
            raw_answer="Option (z) None of the above",
            answer_type="choice",
            question="Which option?",
            gold_facts=[{"predicate": "choice", "subject": "q1", "value": "(a)"}],
            options="(a) First (b) Second"
        )
        self.assertIn(status_c, ["success", "failed", "unparsed"])

        # Ambiguous yes/no
        claim_yn, norm_yn, status_yn = parse_vlm_answer_to_claim(
            raw_answer="It is somewhat possible, but I cannot be sure.",
            answer_type="yes_no",
            question="Is there a table?",
            gold_facts=[{"predicate": "exists", "subject": "table", "value": True}]
        )
        self.assertIn(status_yn, ["success", "failed", "unparsed"])

    # -------------------------------------------------------------------------
    # 2. Z3 SMT Edge Cases
    # -------------------------------------------------------------------------
    def test_z3_empty_claims_and_facts(self):
        # Empty claims and facts
        status, claim_sat, gt_sat, soor_list, t_ms = verify_with_maxsmt([], [], [], gt_weight=500)
        self.assertEqual(status, "sat")
        self.assertEqual(len(claim_sat), 0)
        self.assertEqual(len(gt_sat), 0)
        self.assertEqual(len(soor_list), 0)

    def test_z3_contradiction_oracle_empty(self):
        status, is_contra, t_ms = check_hard_contradiction([], None)
        self.assertEqual(status, "sat")
        self.assertFalse(is_contra)

    def test_z3_complex_contradiction(self):
        # Multiple gold facts with contradictory claims
        gt_facts = [
            {"predicate": "count", "subject": "red_sphere", "value": 2},
            {"predicate": "relation", "subject": "red_sphere", "relation_type": "left", "object": "blue_cube", "value": True}
        ]
        claims = [
            {"predicate": "count", "subject": "red_sphere", "value": 5},
            {"predicate": "relation", "subject": "red_sphere", "relation_type": "left", "object": "blue_cube", "value": False}
        ]
        # Low weight -> drops both claims, preserves both GT facts
        status, claim_sat, gt_sat, soor_list, t_ms = verify_with_maxsmt(gt_facts, claims, [0.3, 0.3], gt_weight=500)
        self.assertEqual(status, "sat")
        self.assertFalse(claim_sat[0])
        self.assertFalse(claim_sat[1])
        self.assertTrue(gt_sat[0])
        self.assertTrue(gt_sat[1])
        self.assertFalse(soor_list[0])
        self.assertFalse(soor_list[1])

    # -------------------------------------------------------------------------
    # 3. Calibration Extremes (0% errors, 100% errors, 2 samples, identical confs)
    # -------------------------------------------------------------------------
    def test_calibration_zero_percent_errors(self):
        # 100% accuracy in calibration split
        logits = [[4.0, 1.0, 0.0] for _ in range(10)]
        labels = [True] * 10
        confs = [0.95] * 10

        temp = TemperatureScaling()
        temp.fit(logits, labels)
        self.assertFalse(np.isnan(temp.temperature))
        self.assertGreater(temp.temperature, 0.0)

        iso = IsotonicCalibrator()
        iso.fit(confs, labels)
        iso_out = iso.transform([0.95])
        self.assertFalse(np.isnan(iso_out[0]))

        aps = AdaptivePredictionSets(alpha=0.10)
        aps.fit(logits, labels)
        self.assertLessEqual(aps.q_hat, 1.0)
        aps_out = aps.transform([[4.0, 1.0, 0.0]])
        self.assertGreaterEqual(aps_out[0], 0.05)

    def test_calibration_hundred_percent_errors(self):
        # 0% accuracy in calibration split (severe overconfidence)
        logits = [[4.5, 1.0, 0.0] for _ in range(10)]
        labels = [False] * 10
        confs = [0.95] * 10

        temp = TemperatureScaling()
        temp.fit(logits, labels)
        self.assertGreater(temp.temperature, 1.0) # Should increase T significantly

        iso = IsotonicCalibrator()
        iso.fit(confs, labels)
        iso_out = iso.transform([0.95])
        self.assertLessEqual(iso_out[0], 0.5)

        aps = AdaptivePredictionSets(alpha=0.10)
        aps.fit(logits, labels)
        self.assertLessEqual(aps.q_hat, 1.0)
        self.assertGreater(aps.q_hat, 0.0)

    def test_calibration_minimal_two_samples(self):
        logits = [[3.0, 1.0], [4.0, 1.0]]
        labels = [True, False]
        confs = [0.88, 0.95]

        temp = TemperatureScaling()
        temp.fit(logits, labels)
        self.assertFalse(np.isnan(temp.temperature))

        iso = IsotonicCalibrator()
        iso.fit(confs, labels)
        self.assertEqual(len(iso.transform([0.90])), 1)

        aps = AdaptivePredictionSets(alpha=0.10)
        aps.fit(logits, labels)
        self.assertLessEqual(aps.q_hat, 1.0)

    def test_calibration_identical_confidences(self):
        confs = [0.80] * 10
        labels = [True] * 5 + [False] * 5

        ece, _ = compute_ece(confs, labels, num_bins=5)
        self.assertFalse(np.isnan(ece))
        brier = compute_brier_score(confs, labels)
        self.assertFalse(np.isnan(brier))
        nll = compute_nll(confs, labels)
        self.assertFalse(np.isnan(nll))

    # -------------------------------------------------------------------------
    # 4. Statistical Methods Edge Cases
    # -------------------------------------------------------------------------
    def test_mcnemar_zero_discordant_pairs(self):
        # Perfectly identical predictions
        flags_a = [True, False, True, False]
        flags_b = [True, False, True, False]
        table = compute_paired_contingency(flags_a, flags_b)
        stat, p_val = mcnemar_test(table)
        self.assertEqual(stat, 0.0)
        self.assertEqual(p_val, 1.0)

    def test_cohens_h_effect_sizes(self):
        # Identical proportions
        h_zero = cohens_h(0.80, 0.80)
        self.assertAlmostEqual(h_zero, 0.0, places=5)

        # Large shift (0.90 vs 0.30)
        h_large = cohens_h(0.90, 0.30)
        self.assertGreater(abs(h_large), 0.8)

    def test_benjamini_hochberg_fdr(self):
        # Mixed p-values
        p_vals = [0.001, 0.01, 0.04, 0.15, 0.50]
        rejected, p_adj = apply_benjamini_hochberg(p_vals, alpha=0.05)
        self.assertEqual(len(rejected), len(p_vals))
        self.assertTrue(rejected[0]) # 0.001 must be rejected
        self.assertFalse(rejected[-1]) # 0.50 must not be rejected

if __name__ == "__main__":
    unittest.main()

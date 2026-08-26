"""Unit tests for Z3 SMT & MaxSMT verification and SOOR tracking."""

import unittest
from src.solver.verifier import check_hard_contradiction, verify_with_maxsmt
from src.solver.diagnostics import compute_contradiction_metrics, compute_sfar, compute_soor

class TestZ3Verifier(unittest.TestCase):

    def test_hard_contradiction_oracle_conflict(self):
        # Ground truth: 3 chair legs
        gt = [{"predicate": "count", "subject": "chair_leg", "value": 3}]
        # VLM claim: 4 chair legs (contradiction)
        claim = {"predicate": "count", "subject": "chair_leg", "value": 4}

        status, is_contra, t_ms = check_hard_contradiction(gt, claim)
        self.assertEqual(status, "unsat")
        self.assertTrue(is_contra)
        self.assertGreater(t_ms, 0.0)

    def test_hard_contradiction_oracle_consistent(self):
        gt = [{"predicate": "count", "subject": "chair_leg", "value": 3}]
        claim = {"predicate": "count", "subject": "chair_leg", "value": 3}

        status, is_contra, t_ms = check_hard_contradiction(gt, claim)
        self.assertEqual(status, "sat")
        self.assertFalse(is_contra)

    def test_maxsmt_soft_verification(self):
        gt = [{"predicate": "count", "subject": "chair_leg", "value": 3}]
        # Conflicting claim with low weight (0.4 < W_gt=500 -> W_vlm=400)
        claims = [
            {"predicate": "count", "subject": "chair_leg", "value": 5}
        ]
        weights = [0.4]

        status, claim_sat, gt_sat, soor_list, t_ms = verify_with_maxsmt(gt, claims, weights, gt_weight=500)
        self.assertEqual(status, "sat")
        self.assertFalse(claim_sat[0])  # conflicting claim dropped
        self.assertTrue(gt_sat[0])      # GT preserved
        self.assertFalse(soor_list[0])  # No SOOR event

    def test_maxsmt_soor_override(self):
        gt = [{"predicate": "count", "subject": "chair_leg", "value": 3}]
        # Conflicting claim with high weight (0.92 > W_gt=500 -> W_vlm=920)
        claims = [
            {"predicate": "count", "subject": "chair_leg", "value": 5}
        ]
        weights = [0.92]

        status, claim_sat, gt_sat, soor_list, t_ms = verify_with_maxsmt(gt, claims, weights, gt_weight=500)
        self.assertEqual(status, "sat")
        self.assertTrue(claim_sat[0])   # VLM claim satisfied
        self.assertFalse(gt_sat[0])     # GT dropped
        self.assertTrue(soor_list[0])   # SOOR triggered!

    def test_sfar_computation(self):
        # 4 items:
        # Item 0: correct, accepted
        # Item 1: wrong, accepted (false accept!)
        # Item 2: wrong, rejected (caught!)
        # Item 3: wrong, accepted (false accept!)
        is_correct = [True, False, False, False]
        is_accepted = [True, True, False, True]

        sfar = compute_sfar(is_correct, is_accepted)
        # 2 false accepts out of 3 false claims = 2/3 = 0.6667
        self.assertAlmostEqual(sfar, 2.0 / 3.0, places=3)

    def test_soor_computation(self):
        is_correct = [True, False, False, False]
        soor_flags = [False, True, False, False]
        soor = compute_soor(is_correct, soor_flags)
        # 1 SOOR out of 3 false claims = 1/3 = 0.3333
        self.assertAlmostEqual(soor, 1.0 / 3.0, places=3)

if __name__ == "__main__":
    unittest.main()


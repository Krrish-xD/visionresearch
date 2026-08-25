"""Unit tests for Z3 SMT & MaxSMT verification."""

import unittest
from src.solver.verifier import check_hard_contradiction, verify_with_maxsmt
from src.solver.diagnostics import compute_contradiction_metrics, compute_sfar

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
        # Consistent claim + inconsistent claim
        claims = [
            {"predicate": "count", "subject": "chair_leg", "value": 3},
            {"predicate": "count", "subject": "chair_leg", "value": 5}
        ]
        weights = [0.9, 0.4]

        status, sat_list, t_ms = verify_with_maxsmt(gt, claims, weights)
        self.assertEqual(status, "sat")
        self.assertTrue(sat_list[0])   # consistent claim satisfied
        self.assertFalse(sat_list[1])  # conflicting claim dropped

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

if __name__ == "__main__":
    unittest.main()

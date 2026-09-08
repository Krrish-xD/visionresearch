"""Unit tests for Multi-Hypothesis N-Best MaxSMT Selection."""

import unittest
from src.solver.verifier import verify_multi_hypothesis_maxsmt

class TestMultiHypothesisMaxSMT(unittest.TestCase):

    def test_top1_contradicted_recovers_second_hypothesis(self):
        # Ground truth is (b)
        gt = [{"predicate": "choice", "subject": "item_1", "value": "(b)"}]
        
        # Candidate 0 is (a) (incorrect top-1 prediction, weight 0.85)
        # Candidate 1 is (b) (correct second prediction, weight 0.65)
        # Candidate 2 is (c) (incorrect third prediction, weight 0.20)
        cands = [
            {"predicate": "choice", "subject": "item_1", "value": "(a)"},
            {"predicate": "choice", "subject": "item_1", "value": "(b)"},
            {"predicate": "choice", "subject": "item_1", "value": "(c)"}
        ]
        weights = [0.85, 0.65, 0.20]

        status, selected_idx, sat_list, t_ms = verify_multi_hypothesis_maxsmt(
            gold_facts=gt,
            candidate_claims=cands,
            candidate_weights=weights,
            hard_gt=True
        )

        self.assertEqual(status, "sat")
        self.assertEqual(selected_idx, 1, "Should recover candidate 1 consistent with ground truth")
        self.assertFalse(sat_list[0])
        self.assertTrue(sat_list[1])
        self.assertFalse(sat_list[2])
        self.assertGreater(t_ms, 0.0)

    def test_top1_consistent_selected_immediately(self):
        gt = [{"predicate": "choice", "subject": "item_1", "value": "(a)"}]
        cands = [
            {"predicate": "choice", "subject": "item_1", "value": "(a)"},
            {"predicate": "choice", "subject": "item_1", "value": "(b)"}
        ]
        weights = [0.90, 0.40]

        status, selected_idx, sat_list, _ = verify_multi_hypothesis_maxsmt(
            gold_facts=gt,
            candidate_claims=cands,
            candidate_weights=weights,
            hard_gt=True
        )

        self.assertEqual(status, "sat")
        self.assertEqual(selected_idx, 0)
        self.assertTrue(sat_list[0])
        self.assertFalse(sat_list[1])

    def test_all_candidates_contradicted_returns_none(self):
        # Ground truth is (d)
        gt = [{"predicate": "choice", "subject": "item_1", "value": "(d)"}]
        cands = [
            {"predicate": "choice", "subject": "item_1", "value": "(a)"},
            {"predicate": "choice", "subject": "item_1", "value": "(b)"},
            {"predicate": "choice", "subject": "item_1", "value": "(c)"}
        ]
        weights = [0.95, 0.85, 0.75]

        status, selected_idx, sat_list, _ = verify_multi_hypothesis_maxsmt(
            gold_facts=gt,
            candidate_claims=cands,
            candidate_weights=weights,
            hard_gt=True
        )

        self.assertEqual(status, "sat")
        self.assertIsNone(selected_idx, "No candidate should be selected when all contradict GT")
        self.assertEqual(sat_list, [False, False, False])

    def test_unparseable_candidate_handled(self):
        gt = [{"predicate": "choice", "subject": "item_1", "value": "(a)"}]
        cands = [
            None,
            {"predicate": "choice", "subject": "item_1", "value": "(a)"}
        ]
        weights = [0.99, 0.70]

        status, selected_idx, sat_list, _ = verify_multi_hypothesis_maxsmt(
            gold_facts=gt,
            candidate_claims=cands,
            candidate_weights=weights,
            hard_gt=True
        )

        self.assertEqual(status, "sat")
        self.assertEqual(selected_idx, 1)
        self.assertFalse(sat_list[0])
        self.assertTrue(sat_list[1])

    def test_empty_candidates_handled(self):
        gt = [{"predicate": "count", "subject": "obj", "value": 1}]
        status, selected_idx, sat_list, _ = verify_multi_hypothesis_maxsmt(
            gold_facts=gt,
            candidate_claims=[],
            candidate_weights=[]
        )
        self.assertEqual(status, "sat")
        self.assertIsNone(selected_idx)
        self.assertEqual(sat_list, [])

if __name__ == "__main__":
    unittest.main()

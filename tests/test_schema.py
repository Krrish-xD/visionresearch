"""Unit tests for schema contracts and validation."""

import unittest
from src.formalization.validators import validate_item, validate_prediction, validate_solver_result

class TestSchemaValidation(unittest.TestCase):

    def test_valid_item(self):
        item = {
            "item_id": "mmvp_0001",
            "dataset": "mmvp",
            "question": "How many chair legs are visible?",
            "answer_type": "count",
            "gold_answer": "3",
            "gold_facts": [{"predicate": "count", "subject": "chair_leg", "value": 3}],
            "category": "Quantity and Count"
        }
        valid, msg = validate_item(item)
        self.assertTrue(valid, msg)

    def test_invalid_item_missing_required(self):
        item = {
            "item_id": "mmvp_0001",
            "dataset": "mmvp"
        }
        valid, msg = validate_item(item)
        self.assertFalse(valid)

    def test_valid_prediction(self):
        pred = {
            "item_id": "mmvp_0001",
            "model": "llava-1.5-7b",
            "prompt_id": "constrained_v1",
            "raw_answer": "4",
            "normalized_answer": "4",
            "claim": {"predicate": "count", "subject": "chair_leg", "value": 4},
            "raw_confidence": 0.92,
            "token_logprobs": [-0.08],
            "is_correct": False,
            "parse_status": "success"
        }
        valid, msg = validate_prediction(pred)
        self.assertTrue(valid, msg)

    def test_valid_solver_result(self):
        res = {
            "item_id": "mmvp_0001",
            "condition": "temperature_scaled",
            "weight": 0.76,
            "solver_status": "sat",
            "is_satisfied": True,
            "is_contradicted": False,
            "solve_time_ms": 1.45
        }
        valid, msg = validate_solver_result(res)
        self.assertTrue(valid, msg)

if __name__ == "__main__":
    unittest.main()

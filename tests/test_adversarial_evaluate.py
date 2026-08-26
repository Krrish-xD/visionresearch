"""
Adversarial stress tests and empirical verification for VLM evaluation pipeline and schema validators.
Author: Challenger 2 (Milestone 4)
"""

import os
import json
import tempfile
import pytest
import torch
import numpy as np

from src.vlm.evaluate import VLMInferenceRunner, run_predictions_on_dataset
from src.vlm.confidence import extract_token_confidence
from src.formalization.validators import validate_prediction, validate_item, validate_solver_result
from src.formalization.schema import PREDICTION_SCHEMA, ITEM_SCHEMA
from src.formalization.parser import parse_vlm_answer_to_claim, normalize_text


class TestBoundaryLimits:
    """Boundary stress tests on dataset limits and slicing."""

    def test_limit_zero(self, tmp_path):
        out_file = run_predictions_on_dataset(
            dataset_name="mmvp",
            model_key="llava-1.5-7b",
            limit=0,
            mock=True,
            output_dir=str(tmp_path)
        )
        assert os.path.exists(out_file)
        with open(out_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 0

    def test_limit_one(self, tmp_path):
        out_file = run_predictions_on_dataset(
            dataset_name="mmvp",
            model_key="llava-1.5-7b",
            limit=1,
            mock=True,
            output_dir=str(tmp_path)
        )
        assert os.path.exists(out_file)
        with open(out_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 1
        valid, msg = validate_prediction(lines[0])
        assert valid is True, f"Validation failed: {msg}"

    def test_limit_exceeding_dataset_length(self, tmp_path):
        # mmvp has 300 items (301 lines with trailing)
        out_file = run_predictions_on_dataset(
            dataset_name="mmvp",
            model_key="llava-1.5-7b",
            limit=10000,
            mock=True,
            output_dir=str(tmp_path)
        )
        assert os.path.exists(out_file)
        with open(out_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 300
        for pred in lines:
            valid, msg = validate_prediction(pred)
            assert valid is True, f"Validation failed for {pred['item_id']}: {msg}"

    def test_limit_none_runs_all(self, tmp_path):
        out_file = run_predictions_on_dataset(
            dataset_name="mmvp",
            model_key="llava-1.5-7b",
            limit=None,
            mock=True,
            output_dir=str(tmp_path)
        )
        assert os.path.exists(out_file)
        with open(out_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 300


class TestMultiDatasetEvaluation:
    """Stress tests across all supported datasets: mmvp, clevr, gqa."""

    @pytest.mark.parametrize("dataset_name,sample_size", [
        ("mmvp", 50),
        ("clevr", 50),
        ("gqa", 50),
    ])
    def test_mock_evaluation_across_datasets(self, dataset_name, sample_size, tmp_path):
        out_file = run_predictions_on_dataset(
            dataset_name=dataset_name,
            model_key="llava-1.5-7b",
            limit=sample_size,
            mock=True,
            output_dir=str(tmp_path)
        )
        assert os.path.exists(out_file)
        with open(out_file, "r", encoding="utf-8") as f:
            predictions = [json.loads(line) for line in f if line.strip()]

        assert len(predictions) == sample_size
        for pred in predictions:
            valid, msg = validate_prediction(pred)
            assert valid is True, f"Schema validation error on {dataset_name} ({pred['item_id']}): {msg}"
            assert pred["parse_status"] in ["success", "failed", "unsupported"]
            assert 0.0 <= pred["raw_confidence"] <= 1.0
            assert isinstance(pred["token_logprobs"], list)
            assert len(pred["token_logprobs"]) > 0

    def test_all_answer_types_handling(self):
        runner = VLMInferenceRunner(model_key="llava-1.5-7b")
        test_items = [
            {
                "item_id": "test_count",
                "question": "How many cars are there?",
                "answer_type": "count",
                "gold_answer": "3",
                "options": "",
                "gold_facts": [{"predicate": "count", "subject": "car", "value": 3}]
            },
            {
                "item_id": "test_yes_no",
                "question": "Is there a dog?",
                "answer_type": "yes_no",
                "gold_answer": "yes",
                "options": "",
                "gold_facts": [{"predicate": "exists", "subject": "dog", "value": True}]
            },
            {
                "item_id": "test_attribute",
                "question": "What color is the block?",
                "answer_type": "attribute",
                "gold_answer": "red",
                "options": "",
                "gold_facts": [{"predicate": "attribute", "subject": "block", "attribute_type": "color", "value": "red"}]
            },
            {
                "item_id": "test_choice",
                "question": "Which animal is bigger?",
                "answer_type": "choice",
                "gold_answer": "(a)",
                "options": "(a) Elephant (b) Mouse",
                "gold_facts": [{"predicate": "choice", "subject": "animal", "value": "(a)", "attribute_type": "option"}]
            },
            {
                "item_id": "test_relation",
                "question": "Is the cat on top of the mat?",
                "answer_type": "relation",
                "gold_answer": "above",
                "options": "",
                "gold_facts": [{"predicate": "relation", "subject": "cat", "relation_type": "above", "object": "mat"}]
            }
        ]

        for item in test_items:
            pred = runner.run_inference_on_item(item)
            valid, msg = validate_prediction(pred)
            assert valid is True, f"Failed schema validation for {item['answer_type']}: {msg}"
            assert pred["parse_status"] in ["success", "failed", "unsupported"]
            if pred["parse_status"] == "success":
                assert pred["claim"] is not None
            else:
                assert pred["claim"] is None

    def test_unparseable_answers_fallback(self):
        """Verify that completely unparseable responses fail gracefully to claim=None without schema violation."""
        claim, norm_ans, status = parse_vlm_answer_to_claim(
            "I cannot determine the answer from this image.",
            "count",
            question="How many dogs?"
        )
        assert status == "failed"
        assert claim is None

        # Build prediction record manually and validate against schema
        pred = {
            "item_id": "unparseable_01",
            "model": "llava-1.5-7b",
            "prompt_id": "constrained_v1",
            "raw_answer": "I cannot determine the answer from this image.",
            "normalized_answer": norm_ans,
            "claim": claim,
            "raw_confidence": 0.5,
            "token_logprobs": [-0.693],
            "is_correct": False,
            "parse_status": status
        }
        valid, msg = validate_prediction(pred)
        assert valid is True, f"Schema validation failed on unparseable prediction: {msg}"

    def test_missing_optional_fields_resilience(self):
        """Test item with minimal required fields (missing options, gold_facts, image_path)."""
        runner = VLMInferenceRunner(model_key="llava-1.5-7b")
        minimal_item = {
            "item_id": "min_001",
            "question": "How many circles?",
            "answer_type": "count",
            "gold_answer": "4"
        }
        pred = runner.run_inference_on_item(minimal_item)
        valid, msg = validate_prediction(pred)
        assert valid is True, f"Failed for minimal item: {msg}"
        assert pred["item_id"] == "min_001"


class TestDeterminismAndNumericalInvariants:
    """Stress tests on determinism, confidence ranges, and logprob consistency."""

    def test_mock_inference_determinism_same_process(self):
        runner = VLMInferenceRunner(model_key="llava-1.5-7b")
        item = {
            "item_id": "det_item_42",
            "question": "How many balls?",
            "answer_type": "count",
            "gold_answer": "2",
            "options": "",
            "gold_facts": [{"predicate": "count", "subject": "ball", "value": 2}]
        }
        pred1 = runner.run_inference_on_item(item)
        pred2 = runner.run_inference_on_item(item)

        assert pred1 == pred2
        assert pred1["raw_confidence"] == pred2["raw_confidence"]
        assert pred1["token_logprobs"] == pred2["token_logprobs"]
        assert pred1["is_correct"] == pred2["is_correct"]
        assert pred1["claim"] == pred2["claim"]

    def test_mock_logprob_confidence_consistency(self):
        runner = VLMInferenceRunner(model_key="llava-1.5-7b")
        for i in range(20):
            item = {
                "item_id": f"consistency_item_{i}",
                "question": "Is the box open?",
                "answer_type": "yes_no",
                "gold_answer": "yes",
                "options": "",
                "gold_facts": [{"predicate": "exists", "subject": "box", "value": True}]
            }
            pred = runner.run_inference_on_item(item)
            raw_conf = pred["raw_confidence"]
            logprobs = pred["token_logprobs"]
            
            assert 0.0 <= raw_conf <= 1.0
            assert len(logprobs) == 1
            # Check logprob == ln(raw_confidence) within floating point tolerance
            expected_logprob = float(np.log(raw_conf))
            assert np.isclose(logprobs[0], expected_logprob, atol=1e-5)

    def test_extract_token_confidence_empty_scores(self):
        conf, logprobs, ans_logits, cand_ids = extract_token_confidence((), torch.tensor([1, 2, 3]), prompt_len=0)
        assert conf == 1.0
        assert logprobs == [0.0]

    def test_extract_token_confidence_zero_generated_tokens(self):
        scores = (torch.randn(1, 100),)
        gen_ids = torch.tensor([[10, 20]])
        # prompt_len equals sequence length => 0 new tokens
        conf, logprobs, ans_logits, cand_ids = extract_token_confidence(scores, gen_ids, prompt_len=2)
        assert conf == 1.0
        assert logprobs == [0.0]

    def test_extract_token_confidence_normal_generation(self):
        vocab_size = 50
        # 3 generated tokens
        scores = (
            torch.randn(1, vocab_size),
            torch.randn(1, vocab_size),
            torch.randn(1, vocab_size)
        )
        gen_ids = torch.tensor([[1, 2, 5, 12, 25]])
        prompt_len = 2

        conf, logprobs, ans_logits, cand_ids = extract_token_confidence(scores, gen_ids, prompt_len=prompt_len)
        assert len(logprobs) == 3
        assert 0.0001 <= conf <= 1.0
        # Check geometric mean property
        expected_conf = float(np.exp(np.mean(logprobs)))
        expected_conf = max(0.0001, min(1.0, expected_conf))
        assert np.isclose(conf, expected_conf, atol=1e-5)

    def test_extract_token_confidence_numerical_extremes(self):
        vocab_size = 10
        # Extreme logits
        large_logits = torch.zeros(1, vocab_size)
        large_logits[0, 3] = 1000.0  # Dominant token with near 1.0 prob
        scores = (large_logits,)
        gen_ids = torch.tensor([[3]])

        conf, logprobs, ans_logits, cand_ids = extract_token_confidence(scores, gen_ids, prompt_len=0)
        assert not np.isnan(conf)
        assert not np.isinf(conf)
        assert 0.0001 <= conf <= 1.0
        assert not np.isnan(logprobs[0])


class TestSchemaValidationAdversarialCorruption:
    """Adversarial testing of PREDICTION_SCHEMA validation."""

    @pytest.fixture
    def valid_pred(self):
        return {
            "item_id": "test_valid_01",
            "model": "llava-1.5-7b",
            "prompt_id": "constrained_v1",
            "raw_answer": "yes",
            "normalized_answer": "yes",
            "claim": {"predicate": "exists", "subject": "dog", "value": True},
            "raw_confidence": 0.95,
            "token_logprobs": [-0.051293],
            "is_correct": True,
            "parse_status": "success"
        }

    def test_valid_prediction_passes(self, valid_pred):
        valid, msg = validate_prediction(valid_pred)
        assert valid is True
        assert msg == "valid"

    def test_null_claim_allowed(self, valid_pred):
        valid_pred["claim"] = None
        valid, msg = validate_prediction(valid_pred)
        assert valid is True

    @pytest.mark.parametrize("missing_field", [
        "item_id", "model", "prompt_id", "raw_answer",
        "normalized_answer", "raw_confidence", "is_correct"
    ])
    def test_missing_required_fields_rejected(self, valid_pred, missing_field):
        corrupted = dict(valid_pred)
        del corrupted[missing_field]
        valid, msg = validate_prediction(corrupted)
        assert valid is False
        assert f"'{missing_field}' is a required property" in msg

    @pytest.mark.parametrize("bad_conf", [-0.001, -1.0, 1.001, 2.0, "0.95", None])
    def test_invalid_confidence_rejected(self, valid_pred, bad_conf):
        corrupted = dict(valid_pred)
        corrupted["raw_confidence"] = bad_conf
        valid, msg = validate_prediction(corrupted)
        assert valid is False

    @pytest.mark.parametrize("boundary_conf", [0.0, 0.5, 1.0])
    def test_boundary_confidence_accepted(self, valid_pred, boundary_conf):
        pred = dict(valid_pred)
        pred["raw_confidence"] = boundary_conf
        valid, msg = validate_prediction(pred)
        assert valid is True

    @pytest.mark.parametrize("bad_status", ["partial", "ok", "error", 123, True])
    def test_invalid_parse_status_rejected(self, valid_pred, bad_status):
        corrupted = dict(valid_pred)
        corrupted["parse_status"] = bad_status
        valid, msg = validate_prediction(corrupted)
        assert valid is False

    @pytest.mark.parametrize("bad_claim", ["just a string", 12345, ["list", "of", "items"]])
    def test_invalid_claim_types_rejected(self, valid_pred, bad_claim):
        corrupted = dict(valid_pred)
        corrupted["claim"] = bad_claim
        valid, msg = validate_prediction(corrupted)
        assert valid is False

    @pytest.mark.parametrize("bad_logprobs", ["not a list", [1.0, "invalid_str"], None, {"a": 1}])
    def test_invalid_token_logprobs_rejected(self, valid_pred, bad_logprobs):
        corrupted = dict(valid_pred)
        corrupted["token_logprobs"] = bad_logprobs
        valid, msg = validate_prediction(corrupted)
        assert valid is False

    @pytest.mark.parametrize("bad_is_correct", ["true", "false", 1, 0, None, [True]])
    def test_invalid_is_correct_type_rejected(self, valid_pred, bad_is_correct):
        corrupted = dict(valid_pred)
        corrupted["is_correct"] = bad_is_correct
        valid, msg = validate_prediction(corrupted)
        assert valid is False

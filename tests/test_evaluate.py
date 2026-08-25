"""
Tests for the offline VLM evaluation pipeline.
"""
import pytest
import os
from unittest.mock import patch, MagicMock, mock_open

from src.vlm.evaluate import VLMInferenceRunner, run_predictions_on_dataset
from src.formalization.validators import validate_prediction

@pytest.fixture
def mock_item():
    return {
        "item_id": "test_item_001",
        "question": "Is there a dog in the image?",
        "answer_type": "yes_no",
        "gold_answer": "yes",
        "options": "",
        "gold_facts": ["A dog is present."]
    }

class TestVLMInferenceRunner:
    def test_mock_inference_mode(self, mock_item):
        """Test that mock inference returns a well-formed prediction record."""
        # Using a dummy model key; config will fallback if not found, or we patch the open call
        with patch('builtins.open', mock_open(read_data='models:\n  dummy: {hf_id: "test/dummy"}')):
            import yaml
            # We mock yaml.safe_load since we mock open
            with patch('yaml.safe_load', return_value={"models": {"dummy": {"hf_id": "test/dummy"}}}):
                runner = VLMInferenceRunner(model_key="dummy")
        
        # We don't call runner.load_model(), so engine.model remains None -> triggers mock mode
        
        prediction = runner.run_inference_on_item(mock_item, prompt_id="test_prompt")
        
        # Verify the structure of the returned dictionary
        assert prediction["item_id"] == "test_item_001"
        assert prediction["model"] == "dummy"
        assert prediction["prompt_id"] == "test_prompt"
        
        # Verify parsing and logic occurred
        assert "raw_answer" in prediction
        assert "normalized_answer" in prediction
        assert "claim" in prediction
        assert "raw_confidence" in prediction
        assert "token_logprobs" in prediction
        assert "is_correct" in prediction
        
        # Validate against the official schema
        is_valid, msg = validate_prediction(prediction)
        assert is_valid is True, f"Prediction schema validation failed: {msg}"

    @patch("src.vlm.evaluate.VLMEngine")
    def test_load_model_calls_engine(self, mock_engine_cls, mock_item):
        """Test that VLMInferenceRunner delegates loading to the unified engine."""
        with patch('builtins.open', mock_open()):
            with patch('yaml.safe_load', return_value={"models": {"dummy": {"hf_id": "test/dummy", "load_in_4bit": False}}}):
                runner = VLMInferenceRunner(model_key="dummy")
                
                # Mock the engine instance inside the runner
                mock_engine_instance = MagicMock()
                runner.engine = mock_engine_instance
                
                runner.load_model()
                
                # Verify the engine was asked to load the HF ID with the correct config
                mock_engine_instance.load_model.assert_called_once_with(
                    "test/dummy", load_in_4bit=False, trust_remote_code=False
                )

    def test_config_fallback_empty(self):
        """Test fallback when config file has empty or unexpected format."""
        with patch('builtins.open', mock_open()):
            with patch('yaml.safe_load', return_value={}):
                runner = VLMInferenceRunner(model_key="unknown_model")
                assert runner.cfg == {}


class TestRunPredictionsOnDataset:
    def test_run_predictions_mock_with_limit(self, tmp_path):
        """Test running mock predictions on existing dataset with limit."""
        out_dir = str(tmp_path / "predictions")
        out_file = run_predictions_on_dataset(
            dataset_name="mmvp",
            model_key="llava-1.5-7b",
            limit=2,
            mock=True,
            output_dir=out_dir
        )
        assert os.path.exists(out_file)
        
        import json
        with open(out_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        
        assert len(lines) == 2
        for pred in lines:
            valid, msg = validate_prediction(pred)
            assert valid is True, f"Schema validation error: {msg}"
            assert pred["model"] == "llava-1.5-7b"

    def test_run_predictions_nonexistent_dataset(self, tmp_path):
        """Test FileNotFoundError when dataset does not exist."""
        with pytest.raises(FileNotFoundError):
            run_predictions_on_dataset(
                dataset_name="nonexistent_dataset_xyz",
                limit=1,
                mock=True,
                output_dir=str(tmp_path)
            )

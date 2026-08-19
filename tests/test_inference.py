"""
Tests for VLMEngine model loading heuristics.

These tests mock the heavy HuggingFace from_pretrained calls so they run
instantly without downloading any model weights or requiring a GPU.
"""
import pytest
from unittest.mock import patch, MagicMock
import os
import tempfile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def weights_dir(tmp_path):
    """Create a temporary weights directory with fake model folders."""
    for name in ["llava-1.5-7b", "Qwen2-VL-2B", "some-generic-model"]:
        (tmp_path / name).mkdir()
    return str(tmp_path)


@pytest.fixture
def engine(weights_dir):
    """Return a fresh VLMEngine pointing at the temp weights dir."""
    from src.vlm.inference import VLMEngine
    return VLMEngine(weights_dir=weights_dir)


# ---------------------------------------------------------------------------
# list_available_models
# ---------------------------------------------------------------------------

class TestListAvailableModels:
    def test_returns_all_subdirectories(self, engine, weights_dir):
        models = engine.list_available_models()
        ids = {m["id"] for m in models}
        assert ids == {"llava-1.5-7b", "Qwen2-VL-2B", "some-generic-model"}

    def test_includes_full_paths(self, engine, weights_dir):
        models = engine.list_available_models()
        for m in models:
            assert m["path"].startswith(weights_dir)
            assert os.path.isdir(m["path"])

    def test_empty_weights_dir(self, tmp_path):
        from src.vlm.inference import VLMEngine
        engine = VLMEngine(weights_dir=str(tmp_path))
        assert engine.list_available_models() == []

    def test_nonexistent_weights_dir(self):
        from src.vlm.inference import VLMEngine
        engine = VLMEngine(weights_dir="/nonexistent/path/that/does/not/exist")
        assert engine.list_available_models() == []

    def test_ignores_files_in_weights_dir(self, weights_dir):
        """Only directories should be listed, not loose files."""
        # Create a stray file inside weights/
        with open(os.path.join(weights_dir, "README.md"), "w") as f:
            f.write("ignore me")
        from src.vlm.inference import VLMEngine
        engine = VLMEngine(weights_dir=weights_dir)
        ids = {m["id"] for m in engine.list_available_models()}
        assert "README.md" not in ids


# ---------------------------------------------------------------------------
# load_model – routing heuristic
# ---------------------------------------------------------------------------

class TestLoadModelRouting:
    """Verify that load_model picks the correct model class based on the path."""

    @patch("src.vlm.inference.AutoProcessor")
    @patch("src.vlm.inference.LlavaForConditionalGeneration")
    def test_llava_path_uses_llava_class(self, mock_llava_cls, mock_proc, engine):
        mock_proc.from_pretrained.return_value = MagicMock()
        mock_llava_cls.from_pretrained.return_value = MagicMock()

        engine.load_model("/weights/llava-1.5-7b")

        mock_llava_cls.from_pretrained.assert_called_once()
        assert engine.current_model_id == "/weights/llava-1.5-7b"

    @patch("src.vlm.inference.AutoProcessor")
    @patch("src.vlm.inference.Qwen2VLForConditionalGeneration")
    def test_qwen_path_uses_qwen_class(self, mock_qwen_cls, mock_proc, engine):
        mock_proc.from_pretrained.return_value = MagicMock()
        mock_qwen_cls.from_pretrained.return_value = MagicMock()

        engine.load_model("/weights/Qwen2-VL-2B")

        mock_qwen_cls.from_pretrained.assert_called_once()
        assert engine.current_model_id == "/weights/Qwen2-VL-2B"

    @patch("src.vlm.inference.AutoProcessor")
    @patch("src.vlm.inference.AutoModelForCausalLM")
    def test_generic_path_uses_auto_class(self, mock_auto_cls, mock_proc, engine):
        mock_proc.from_pretrained.return_value = MagicMock()
        mock_auto_cls.from_pretrained.return_value = MagicMock()

        engine.load_model("/weights/some-generic-model")

        mock_auto_cls.from_pretrained.assert_called_once()
        assert engine.current_model_id == "/weights/some-generic-model"

    @patch("src.vlm.inference.AutoProcessor")
    @patch("src.vlm.inference.AutoModelForCausalLM")
    def test_skip_reload_if_same_model(self, mock_auto_cls, mock_proc, engine):
        """If the same model ID is requested twice, don't reload."""
        mock_proc.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_auto_cls.from_pretrained.return_value = mock_model

        engine.load_model("/weights/some-generic-model")
        engine.load_model("/weights/some-generic-model")

        # Should only have been called once
        assert mock_auto_cls.from_pretrained.call_count == 1

    @patch("src.vlm.inference.torch")
    @patch("src.vlm.inference.AutoProcessor")
    @patch("src.vlm.inference.AutoModelForCausalLM")
    def test_unloads_previous_model(self, mock_auto_cls, mock_proc, mock_torch, engine):
        """Loading a new model should free the old one's VRAM."""
        mock_proc.from_pretrained.return_value = MagicMock()
        mock_auto_cls.from_pretrained.return_value = MagicMock()

        engine.load_model("/weights/model-a")
        engine.load_model("/weights/model-b")

        # torch.cuda.empty_cache should have been called during unload
        mock_torch.cuda.empty_cache.assert_called()

    @patch("src.vlm.inference.AutoProcessor")
    @patch("src.vlm.inference.AutoModelForCausalLM")
    def test_case_insensitive_routing(self, mock_auto_cls, mock_proc, engine):
        """Model class routing should be case-insensitive."""
        mock_proc.from_pretrained.return_value = MagicMock()

        # Patch Qwen class at module level
        with patch("src.vlm.inference.Qwen2VLForConditionalGeneration") as mock_qwen:
            mock_qwen.from_pretrained.return_value = MagicMock()
            engine.load_model("/weights/QWEN2-VL-LARGE")
            mock_qwen.from_pretrained.assert_called_once()

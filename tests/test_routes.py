"""
Tests for FastAPI API routes.

Uses FastAPI's TestClient with mocked VLMEngine so no GPU or model
weights are required.
"""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from fastapi.testclient import TestClient
import io
from PIL import Image


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_engine():
    """Provide a fully mocked VLMEngine and patch it into the routes module."""
    engine = MagicMock()
    engine.model = None
    engine.current_model_id = None
    engine.processor = None
    
    import src.api.routes
    with patch("src.api.routes.vlm_engine", engine):
        # Import app AFTER patching so the router binds to our mock
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        from backend.main import app
        yield engine, TestClient(app)


# ---------------------------------------------------------------------------
# GET /api/models
# ---------------------------------------------------------------------------

class TestListModelsEndpoint:
    def test_returns_models(self, mock_engine):
        engine, client = mock_engine
        engine.list_available_models.return_value = [
            {"id": "llava-1.5-7b", "path": "/weights/llava-1.5-7b"},
            {"id": "Qwen2-VL-2B", "path": "/weights/Qwen2-VL-2B"},
        ]

        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["id"] == "llava-1.5-7b"
        assert data[1]["id"] == "Qwen2-VL-2B"

    def test_returns_empty_list(self, mock_engine):
        engine, client = mock_engine
        engine.list_available_models.return_value = []

        resp = client.get("/api/models")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# GET /api/model/status
# ---------------------------------------------------------------------------

class TestModelStatusEndpoint:
    def test_no_model_loaded(self, mock_engine):
        engine, client = mock_engine
        engine.model = None
        engine.current_model_id = None

        resp = client.get("/api/model/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_loaded"] is False
        assert data["active_model_id"] is None

    def test_model_loaded(self, mock_engine):
        engine, client = mock_engine
        engine.model = MagicMock()  # Truthy – a model is loaded
        engine.current_model_id = "Qwen2-VL-2B"

        resp = client.get("/api/model/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_loaded"] is True
        assert data["active_model_id"] == "Qwen2-VL-2B"


# ---------------------------------------------------------------------------
# POST /api/model/load
# ---------------------------------------------------------------------------

class TestModelLoadEndpoint:
    def test_successful_load(self, mock_engine):
        engine, client = mock_engine
        engine.list_available_models.return_value = [
            {"id": "Qwen2-VL-2B", "path": "/weights/Qwen2-VL-2B"}
        ]
        engine.load_model.return_value = None  # success

        resp = client.post("/api/model/load", json={"model_id": "Qwen2-VL-2B"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        engine.load_model.assert_called_once_with("/weights/Qwen2-VL-2B")

    def test_load_resolves_id_to_path(self, mock_engine):
        """The route should map the short ID to the full local path."""
        engine, client = mock_engine
        engine.list_available_models.return_value = [
            {"id": "llava-1.5-7b", "path": "/home/user/weights/llava-1.5-7b"}
        ]

        client.post("/api/model/load", json={"model_id": "llava-1.5-7b"})
        engine.load_model.assert_called_once_with("/home/user/weights/llava-1.5-7b")

    def test_load_unknown_model_falls_through(self, mock_engine):
        """If the ID isn't found locally, it passes the raw ID to load_model."""
        engine, client = mock_engine
        engine.list_available_models.return_value = []
        engine.load_model.side_effect = Exception("Model not found")

        resp = client.post("/api/model/load", json={"model_id": "nonexistent"})
        assert resp.status_code == 500
        assert "Failed to load model" in resp.json()["detail"]

    def test_load_internal_error(self, mock_engine):
        engine, client = mock_engine
        engine.list_available_models.return_value = [
            {"id": "bad-model", "path": "/weights/bad-model"}
        ]
        engine.load_model.side_effect = RuntimeError("CUDA OOM")

        resp = client.post("/api/model/load", json={"model_id": "bad-model"})
        assert resp.status_code == 500
        assert "CUDA OOM" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/generate
# ---------------------------------------------------------------------------

def _make_test_image_bytes():
    """Create a tiny 10x10 red PNG in memory."""
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


class TestGenerateEndpoint:
    def test_successful_inference(self, mock_engine):
        engine, client = mock_engine
        engine.list_available_models.return_value = [
            {"id": "Qwen2-VL-2B", "path": "/weights/Qwen2-VL-2B"}
        ]
        engine.load_model.return_value = None
        engine.generate_with_logprobs.return_value = {
            "full_text": "A red square",
            "tokens": [
                {"token_id": 1, "text": "A", "logprob": -0.1, "prob_percent": 90.5},
                {"token_id": 2, "text": " red", "logprob": -0.5, "prob_percent": 60.7},
                {"token_id": 3, "text": " square", "logprob": -1.2, "prob_percent": 30.1},
            ]
        }

        resp = client.post(
            "/api/generate",
            data={"prompt": "What is this?", "model_id": "Qwen2-VL-2B"},
            files={"image": ("test.png", _make_test_image_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["full_text"] == "A red square"
        assert len(data["tokens"]) == 3
        assert data["tokens"][0]["text"] == "A"
        assert data["tokens"][1]["prob_percent"] == 60.7

    def test_model_load_failure(self, mock_engine):
        engine, client = mock_engine
        engine.list_available_models.return_value = []
        engine.load_model.side_effect = RuntimeError("Cannot find model")

        resp = client.post(
            "/api/generate",
            data={"prompt": "Describe", "model_id": "nonexistent"},
            files={"image": ("test.png", _make_test_image_bytes(), "image/png")},
        )
        assert resp.status_code == 500
        assert "Failed to load model" in resp.json()["detail"]

    def test_inference_failure(self, mock_engine):
        engine, client = mock_engine
        engine.list_available_models.return_value = [
            {"id": "Qwen2-VL-2B", "path": "/weights/Qwen2-VL-2B"}
        ]
        engine.load_model.return_value = None
        engine.generate_with_logprobs.side_effect = RuntimeError("CUDA error")

        resp = client.post(
            "/api/generate",
            data={"prompt": "Describe", "model_id": "Qwen2-VL-2B"},
            files={"image": ("test.png", _make_test_image_bytes(), "image/png")},
        )
        assert resp.status_code == 500
        assert "Inference failed" in resp.json()["detail"]

    def test_invalid_image(self, mock_engine):
        engine, client = mock_engine
        engine.list_available_models.return_value = [
            {"id": "Qwen2-VL-2B", "path": "/weights/Qwen2-VL-2B"}
        ]
        engine.load_model.return_value = None

        resp = client.post(
            "/api/generate",
            data={"prompt": "Describe", "model_id": "Qwen2-VL-2B"},
            files={"image": ("garbage.bin", b"not an image at all", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "Invalid image format" in resp.json()["detail"]
        engine.load_model.assert_not_called()

    def test_missing_model_id_field(self, mock_engine):
        """Omitting model_id should return a 422 validation error."""
        engine, client = mock_engine

        resp = client.post(
            "/api/generate",
            data={"prompt": "Describe"},
            files={"image": ("test.png", _make_test_image_bytes(), "image/png")},
        )
        assert resp.status_code == 422

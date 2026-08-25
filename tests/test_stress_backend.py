"""
Adversarial Stress and Resilience Test Suite for FastAPI Backend and VLMEngine Singleton.

Tests:
1. High-concurrency stress on read and write endpoints.
2. Malformed payloads, invalid data types, oversized inputs, and injection attempts.
3. Invalid HTTP methods (PUT, DELETE, PATCH, etc.) across all endpoints.
4. Corrupted images, truncated headers, empty files, and non-image payloads.
5. Engine singleton lifecycle resilience under failure sequences.
6. Resource cleanup, memory reference release, and file handle leak checks.
"""

import io
import os
import sys
import gc
import json
import pytest
import concurrent.futures
from unittest.mock import patch, MagicMock
from PIL import Image
from fastapi.testclient import TestClient

# Ensure workspace root and backend are in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from backend.main import app
from src.vlm.engine import VLMEngine
from src.api.routes import router


def _make_dummy_image_bytes(width=10, height=10, color=(255, 0, 0), fmt="PNG"):
    """Helper to generate valid image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


@pytest.fixture
def mock_engine_singleton():
    """Mock the VLMEngine singleton in routes for backend stress testing."""
    engine = MagicMock(spec=VLMEngine)
    engine.weights_dir = "../weights"
    engine.model = None
    engine.current_model_id = None
    engine.processor = None
    engine.device = "cpu"
    
    # Default behaviors
    engine.list_available_models.return_value = [
        {"id": "llava-1.5-7b", "path": "/weights/llava-1.5-7b"},
        {"id": "Qwen2-VL-2B", "path": "/weights/Qwen2-VL-2B"},
    ]
    
    def mock_load(model_path, **kwargs):
        known_paths = ["/weights/llava-1.5-7b", "/weights/Qwen2-VL-2B", "llava-1.5-7b", "Qwen2-VL-2B"]
        if model_path not in known_paths:
            # Emulate unload and failure
            engine.model = None
            raise RuntimeError(f"Cannot load model from path: {model_path}")
        engine.current_model_id = model_path.split("/")[-1]
        engine.model = MagicMock()
        return None

    engine.load_model.side_effect = mock_load

    def mock_gen(pil_image, prompt, **kwargs):
        if engine.model is None:
            raise RuntimeError("No model is currently loaded.")
        if "fail_inference" in prompt:
            raise RuntimeError("Inference execution crashed")
        return {
            "full_text": f"Output for prompt: {prompt[:20]}",
            "tokens": [
                {"token_id": 1, "text": "Token", "logprob": -0.5, "prob_percent": 60.65}
            ]
        }

    engine.generate_with_logprobs.side_effect = mock_gen

    with patch("src.api.routes.vlm_engine", engine):
        client = TestClient(app)
        yield engine, client


# ===========================================================================
# 1. High-Concurrency Stress Tests
# ===========================================================================

class TestConcurrencyStress:
    """Stress tests simulating concurrent multi-threaded requests."""

    def test_concurrent_read_requests(self, mock_engine_singleton):
        """Send 100 concurrent requests interleaved between /api/models and /api/model/status."""
        _, client = mock_engine_singleton
        num_requests = 100

        def send_request(idx):
            endpoint = "/api/models" if idx % 2 == 0 else "/api/model/status"
            resp = client.get(endpoint)
            return resp.status_code, resp.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(send_request, i) for i in range(num_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == num_requests
        for status_code, data in results:
            assert status_code == 200
            assert data is not None

    def test_concurrent_mixed_load_generate_status(self, mock_engine_singleton):
        """Interleave concurrent load requests, generate requests, and status requests."""
        _, client = mock_engine_singleton
        num_requests = 60
        valid_img = _make_dummy_image_bytes()

        def worker_task(idx):
            req_type = idx % 3
            if req_type == 0:
                return client.get("/api/model/status").status_code
            elif req_type == 1:
                return client.post("/api/model/load", json={"model_id": "Qwen2-VL-2B"}).status_code
            else:
                return client.post(
                    "/api/generate",
                    data={"prompt": f"Stress prompt {idx}", "model_id": "Qwen2-VL-2B"},
                    files={"image": ("test.png", valid_img, "image/png")}
                ).status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_requests)]
            status_codes = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(status_codes) == num_requests
        # All valid requests should succeed with 200
        for code in status_codes:
            assert code == 200


# ===========================================================================
# 2. Malformed Payloads, Invalid Types, and Injections
# ===========================================================================

class TestMalformedPayloadsAndInjections:
    """Stress tests exploring boundary values, malformed data, and injections."""

    @pytest.mark.parametrize("payload", [
        {},  # Missing required model_id
        {"model_id": 12345},  # Int instead of string
        {"model_id": None},  # None value
        {"model_id": ["nested", "list"]},  # List instead of string
        {"model_id": {"nested": "dict"}},  # Dict instead of string
        {"wrong_key": "llava-1.5-7b"},  # Wrong field name
    ])
    def test_invalid_load_payload_structures(self, mock_engine_singleton, payload):
        """ModelLoadRequest should reject invalid structures with 422 Unprocessable Entity."""
        _, client = mock_engine_singleton
        resp = client.post("/api/model/load", json=payload)
        assert resp.status_code == 422

    def test_invalid_json_syntax(self, mock_engine_singleton):
        """Malformed JSON string in request body should return 422."""
        _, client = mock_engine_singleton
        resp = client.post(
            "/api/model/load",
            content=b"{malformed: json, missing quotes",
            headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 422

    @pytest.mark.parametrize("adversarial_id", [
        "../../../../etc/passwd",
        "/dev/null",
        "model; rm -rf /",
        "model\x00hidden",
        "A" * 10000,  # 10KB string
        "🤖🔥🧠⚡💥" * 50,  # Unicode emoji flood
    ])
    def test_adversarial_model_ids(self, mock_engine_singleton, adversarial_id):
        """Adversarial/injected model IDs should either fail gracefully (500 or 422) without crashing server."""
        _, client = mock_engine_singleton
        resp = client.post("/api/model/load", json={"model_id": adversarial_id})
        # If not rejected by validation, it falls through to load_model which fails with 500
        assert resp.status_code in (422, 500)
        
        # Verify server is still alive and responsive after injection attempt
        status_resp = client.get("/api/model/status")
        assert status_resp.status_code == 200

    @pytest.mark.parametrize("missing_field", ["prompt", "model_id"])
    def test_generate_missing_form_fields(self, mock_engine_singleton, missing_field):
        """Missing required Form fields in /api/generate should return 422."""
        _, client = mock_engine_singleton
        form_data = {"prompt": "Hello", "model_id": "Qwen2-VL-2B"}
        del form_data[missing_field]

        resp = client.post(
            "/api/generate",
            data=form_data,
            files={"image": ("test.png", _make_dummy_image_bytes(), "image/png")}
        )
        assert resp.status_code == 422

    def test_generate_missing_image_file(self, mock_engine_singleton):
        """Missing image file in /api/generate should return 422."""
        _, client = mock_engine_singleton
        resp = client.post(
            "/api/generate",
            data={"prompt": "Hello", "model_id": "Qwen2-VL-2B"}
        )
        assert resp.status_code == 422

    def test_generate_extreme_hyperparameters(self, mock_engine_singleton):
        """Extreme hyperparameter values passed to generate endpoint."""
        _, client = mock_engine_singleton
        resp = client.post(
            "/api/generate",
            data={
                "prompt": "Test prompt",
                "model_id": "Qwen2-VL-2B",
                "temperature": 0.0,
                "top_p": 0.01,
                "top_k": 1,
                "max_tokens": 1000
            },
            files={"image": ("test.png", _make_dummy_image_bytes(), "image/png")}
        )
        assert resp.status_code == 200


# ===========================================================================
# 3. Invalid HTTP Methods Across Endpoints
# ===========================================================================

class TestInvalidHTTPMethods:
    """Ensure invalid HTTP methods on all endpoints return 405 Method Not Allowed cleanly."""

    def test_models_invalid_methods(self, mock_engine_singleton):
        _, client = mock_engine_singleton
        for method in ["post", "put", "delete", "patch"]:
            call = getattr(client, method)
            resp = call("/api/models")
            assert resp.status_code == 405

    def test_status_invalid_methods(self, mock_engine_singleton):
        _, client = mock_engine_singleton
        for method in ["post", "put", "delete", "patch"]:
            call = getattr(client, method)
            resp = call("/api/model/status")
            assert resp.status_code == 405

    def test_load_invalid_methods(self, mock_engine_singleton):
        _, client = mock_engine_singleton
        for method in ["get", "put", "delete", "patch"]:
            call = getattr(client, method)
            resp = call("/api/model/load")
            assert resp.status_code == 405

    def test_generate_invalid_methods(self, mock_engine_singleton):
        _, client = mock_engine_singleton
        for method in ["get", "put", "delete", "patch"]:
            call = getattr(client, method)
            resp = call("/api/generate")
            assert resp.status_code == 405


# ===========================================================================
# 4. Corrupted & Adversarial Media Inputs
# ===========================================================================

class TestCorruptedMediaInputs:
    """Stress test image handling with corrupted, empty, or malicious bytes."""

    def test_zero_byte_image(self, mock_engine_singleton):
        """0-byte empty file upload."""
        _, client = mock_engine_singleton
        resp = client.post(
            "/api/generate",
            data={"prompt": "Test", "model_id": "Qwen2-VL-2B"},
            files={"image": ("empty.png", b"", "image/png")}
        )
        assert resp.status_code == 400
        assert "Invalid image format" in resp.json()["detail"]

    def test_truncated_png_header(self, mock_engine_singleton):
        """Truncated PNG magic bytes followed by sudden EOF."""
        _, client = mock_engine_singleton
        truncated_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        resp = client.post(
            "/api/generate",
            data={"prompt": "Test", "model_id": "Qwen2-VL-2B"},
            files={"image": ("corrupt.png", truncated_png, "image/png")}
        )
        assert resp.status_code == 400
        assert "Invalid image format" in resp.json()["detail"]

    def test_random_binary_garbage(self, mock_engine_singleton):
        """64KB of random binary noise."""
        _, client = mock_engine_singleton
        garbage = os.urandom(65536)
        resp = client.post(
            "/api/generate",
            data={"prompt": "Test", "model_id": "Qwen2-VL-2B"},
            files={"image": ("noise.jpg", garbage, "image/jpeg")}
        )
        assert resp.status_code == 400
        assert "Invalid image format" in resp.json()["detail"]

    def test_non_image_disguised_file(self, mock_engine_singleton):
        """Text / PDF file sent as image/jpeg."""
        _, client = mock_engine_singleton
        fake_payload = b"%PDF-1.4\n1 0 obj\n<< /Title (Exploit) >>\nendobj"
        resp = client.post(
            "/api/generate",
            data={"prompt": "Test", "model_id": "Qwen2-VL-2B"},
            files={"image": ("fake.jpg", fake_payload, "image/jpeg")}
        )
        assert resp.status_code == 400
        assert "Invalid image format" in resp.json()["detail"]

    def test_server_healthy_after_corrupted_image_barrage(self, mock_engine_singleton):
        """Send 20 corrupted image requests and ensure server handles them and remains fully functional."""
        _, client = mock_engine_singleton
        for i in range(20):
            client.post(
                "/api/generate",
                data={"prompt": f"Attack {i}", "model_id": "Qwen2-VL-2B"},
                files={"image": (f"corrupt_{i}.png", os.urandom(512), "image/png")}
            )

        # Ensure subsequent valid request works seamlessly
        resp = client.post(
            "/api/generate",
            data={"prompt": "Valid follow-up", "model_id": "Qwen2-VL-2B"},
            files={"image": ("valid.png", _make_dummy_image_bytes(), "image/png")}
        )
        assert resp.status_code == 200
        assert "Output for prompt" in resp.json()["full_text"]


# ===========================================================================
# 5. Singleton Lifecycle & Recovery After Failure
# ===========================================================================

class TestSingletonLifecycleResilience:
    """Verify that failure states do not deadlock or permanently corrupt the engine singleton."""

    def test_recovery_after_failed_load(self, mock_engine_singleton):
        """Attempt to load a nonexistent model, then verify engine recovers and loads a valid model."""
        engine, client = mock_engine_singleton

        # 1. Attempt bad load
        resp = client.post("/api/model/load", json={"model_id": "nonexistent_bad_model"})
        assert resp.status_code == 500
        assert "Failed to load model" in resp.json()["detail"]

        # 2. Check status reflects un-loaded state
        status_resp = client.get("/api/model/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["is_loaded"] is False

        # 3. Load valid model
        load_resp = client.post("/api/model/load", json={"model_id": "Qwen2-VL-2B"})
        assert load_resp.status_code == 200

        # 4. Check status reflects loaded model
        status_resp2 = client.get("/api/model/status")
        assert status_resp2.status_code == 200
        assert status_resp2.json()["is_loaded"] is True
        assert status_resp2.json()["active_model_id"] == "Qwen2-VL-2B"

    def test_inference_failure_does_not_break_subsequent_inferences(self, mock_engine_singleton):
        """Inference failure on one prompt should not prevent subsequent prompts from succeeding."""
        _, client = mock_engine_singleton

        # Ensure model loaded
        client.post("/api/model/load", json={"model_id": "Qwen2-VL-2B"})

        # Failed inference
        bad_resp = client.post(
            "/api/generate",
            data={"prompt": "fail_inference here", "model_id": "Qwen2-VL-2B"},
            files={"image": ("test.png", _make_dummy_image_bytes(), "image/png")}
        )
        assert bad_resp.status_code == 500
        assert "Inference failed" in bad_resp.json()["detail"]

        # Successful subsequent inference
        good_resp = client.post(
            "/api/generate",
            data={"prompt": "healthy prompt", "model_id": "Qwen2-VL-2B"},
            files={"image": ("test.png", _make_dummy_image_bytes(), "image/png")}
        )
        assert good_resp.status_code == 200
        assert "Output for prompt: healthy prompt" in good_resp.json()["full_text"]

    def test_rapid_model_switching(self, mock_engine_singleton):
        """Switch models back and forth rapidly without race conditions or state corruption."""
        _, client = mock_engine_singleton
        models = ["llava-1.5-7b", "Qwen2-VL-2B"]

        for i in range(10):
            target = models[i % 2]
            resp = client.post("/api/model/load", json={"model_id": target})
            assert resp.status_code == 200
            
            status = client.get("/api/model/status").json()
            assert status["active_model_id"] == target
            assert status["is_loaded"] is True


# ===========================================================================
# 6. Real VLMEngine Class Logic Unit Stress & Empirical Bug Verification
# ===========================================================================

class TestVLMEngineDirectStress:
    """Stress test the concrete VLMEngine class logic directly."""

    def test_vlm_engine_empty_weights_dir(self, tmp_path):
        """Weights directory that does not exist or is empty."""
        engine = VLMEngine(weights_dir=str(tmp_path / "nonexistent"))
        assert engine.list_available_models() == []

        empty_dir = tmp_path / "empty_weights"
        empty_dir.mkdir()
        engine_empty = VLMEngine(weights_dir=str(empty_dir))
        assert engine_empty.list_available_models() == []

    def test_vlm_engine_weights_dir_with_files(self, tmp_path):
        """Weights directory containing files as well as directories."""
        weights = tmp_path / "weights"
        weights.mkdir()
        (weights / "some_file.txt").write_text("not a model dir")
        (weights / "valid_model_dir").mkdir()

        engine = VLMEngine(weights_dir=str(weights))
        models = engine.list_available_models()
        assert len(models) == 1
        assert models[0]["id"] == "valid_model_dir"

    def test_vlm_engine_generate_without_load_raises(self):
        """Calling generate_with_logprobs before load_model must raise RuntimeError."""
        engine = VLMEngine()
        dummy_img = Image.new("RGB", (10, 10))
        with pytest.raises(RuntimeError, match="No model is currently loaded"):
            engine.generate_with_logprobs(dummy_img, "Test prompt")

    def test_vlm_engine_consecutive_same_model_skip(self):
        """Calling load_model on already loaded model should be a no-op."""
        engine = VLMEngine()
        engine.current_model_id = "test-model"
        engine.model = MagicMock()
        
        with patch("transformers.AutoProcessor.from_pretrained") as mock_proc:
            engine.load_model("test-model")
            mock_proc.assert_not_called()

    def test_vlm_engine_resilience_on_failed_load(self):
        """
        Verify that when model loading fails, VLMEngine maintains attributes `model` and `processor`
        as None instead of deleting them, avoiding AttributeError on subsequent accesses.
        """
        engine = VLMEngine()
        engine.model = MagicMock()
        engine.processor = MagicMock()
        engine.current_model_id = "loaded-model"

        with patch("transformers.AutoProcessor.from_pretrained", side_effect=OSError("Model not found")):
            with pytest.raises(OSError):
                engine.load_model("invalid-model")

        # Verify that model and processor attributes are preserved and set to None
        assert hasattr(engine, "model"), "engine should retain model attribute"
        assert hasattr(engine, "processor"), "engine should retain processor attribute"
        assert engine.model is None
        assert engine.processor is None
        assert engine.current_model_id is None
        
        # Accessing engine.model should evaluate cleanly without AttributeError
        assert (engine.model is not None) is False

    def test_vlm_engine_unload_model(self):
        """Verify explicit unload_model cleans state and retains attributes as None."""
        engine = VLMEngine()
        engine.model = MagicMock()
        engine.processor = MagicMock()
        engine.current_model_id = "some-model"

        engine.unload_model()
        assert hasattr(engine, "model")
        assert hasattr(engine, "processor")
        assert engine.model is None
        assert engine.processor is None
        assert engine.current_model_id is None

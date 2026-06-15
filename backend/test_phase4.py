import asyncio
import os
from pathlib import Path

from fastapi.testclient import TestClient
from main import app
from core.database import engine, Base

# Set up test database
Base.metadata.create_all(bind=engine)
client = TestClient(app)

def test_plugin_loading():
    """Test that the plugin system dynamically loaded the sample plugin."""
    # Ensure lifespan runs by using TestClient context manager
    with TestClient(app) as client:
        registry = app.state.orchestrator.registry
        # The brightness module should be loaded
        brightness_plugin = registry.get("brightness")
        assert brightness_plugin is not None, "Sample plugin 'brightness' failed to load"
        assert brightness_plugin.display_name == "Image Brightness"
        print("✅ Plugin system successfully loaded 'brightness' plugin.")

def test_video_endpoint():
    """Test that video upload extracts multiple frames as tasks."""
    # Create a fake video file (we can't really test cv2 on a fake mp4 easily, 
    # but we can test if the endpoint exists and handles a bad file correctly,
    # or we can pass a small real mp4 if available. Here we test bad file.)
    with TestClient(app) as client:
        response = client.post(
            "/api/analyze/video",
            files={"video": ("test.mp4", b"fake video content", "video/mp4")}
        )
        print("RESPONSE:", response.status_code, response.json())
        assert response.status_code == 400
        assert "Failed to open video file" in response.json()["detail"]
        print("✅ Video endpoint successfully mounted and caught invalid video.")

def test_image_comparison():
    """Test the image comparison endpoint."""
    with TestClient(app) as client:
        # Upload an image to get a task ID
        # Since we just want to test comparison logic, we'll mock it if needed.
        # It's better to just call the route with fake task_ids and expect a 404
        res = client.get("/api/compare?task1_id=fake1&task2_id=fake2")
        print("COMPARE RESPONSE:", res.status_code, res.json())
        assert res.status_code == 404
        assert "One or both tasks not found" in res.json()["detail"]
        print("✅ Image comparison endpoint successfully mounted.")

if __name__ == "__main__":
    test_plugin_loading()
    test_video_endpoint()
    test_image_comparison()
    print("All Phase 4 API tests passed!")

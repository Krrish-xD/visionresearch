"""Depth Estimation Module using Depth Anything V2."""

import asyncio
import time
import os
import uuid
from typing import Any
import PIL.Image

from core.base import BaseAnalyzer
from core.schemas import AnalysisResult
from config import settings


class DepthAnalyzer(BaseAnalyzer):
    """Estimates monocular depth using Depth Anything V2."""

    name: str = "depth"
    display_name: str = "Depth Estimation"
    estimated_vram_mb: int = 500
    requires_gpu: bool = True
    stage: int = 4

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        self.model_id = model_id
        self.pipe = None

    async def load_model(self, device: str = "cpu") -> None:
        if self.pipe is None:
            # We import here to avoid slow startup
            from transformers import pipeline
            device = 0 if settings.resolve_device() == "cuda" else -1
            self.pipe = pipeline(
                "depth-estimation", 
                model=self.model_id, 
                device=device
            )

    async def unload_model(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None

    async def analyze(self, image: PIL.Image.Image, **kwargs: Any) -> dict:
        if self.pipe is None:
            raise RuntimeError("Model not loaded")

        start_time = time.time()
        
        # Run inference
        # The pipeline returns a dict: {"predicted_depth": tensor, "depth": PIL.Image}
        result = self.pipe(image.convert("RGB"))
        depth_img = result["depth"]
        
        # Save the depth map to the outputs directory
        output_filename = f"depth_{uuid.uuid4().hex[:8]}.png"
        output_path = settings.output_dir / output_filename
        depth_img.save(output_path)
        
        # Path for the web frontend
        web_path = f"/outputs/{output_filename}"

        return {
            "depth_map_path": web_path
        }

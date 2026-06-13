"""Semantic Segmentation Module using SAM 2 via Ultralytics."""

import asyncio
import time
import uuid
from typing import Any
import PIL.Image

from core.base import BaseAnalyzer
from core.schemas import AnalysisResult
from config import settings


class SegmentationAnalyzer(BaseAnalyzer):
    """Generates semantic segmentation masks using SAM 2 (Segment Anything Model)."""

    name: str = "segmentation"
    display_name: str = "Semantic Segmentation"
    estimated_vram_mb: int = 1500
    requires_gpu: bool = True
    stage: int = 4

    def __init__(self, model_id: str = "sam2_t.pt"):
        # We use sam2_t.pt (Tiny) to conserve VRAM, it's very fast and effective
        self.model_id = model_id
        self.model = None

    async def load_model(self, device: str = "cpu") -> None:
        if self.model is None:
            # We import here to avoid slow startup
            from ultralytics import SAM
            self.model = SAM(self.model_id)

    async def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

    async def analyze(self, image: PIL.Image.Image, **kwargs: Any) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start_time = time.time()
        
        # Run SAM 2 in automatic mask generation mode
        results = self.model(image, verbose=False)
        result = results[0]
        
        # Instead of sending hundreds of vector polygons to the frontend,
        # we generate a visual overlay image and save it to disk.
        output_filename = f"segmentation_{uuid.uuid4().hex[:8]}.png"
        output_path = settings.output_dir / output_filename
        
        # Create a black background to disregard original image colors
        import numpy as np
        black_bg = np.zeros_like(result.orig_img)
        
        # Plot only masks (no boxes or labels since SAM is class-agnostic)
        im_bgr = result.plot(img=black_bg, labels=False, boxes=False)
        im_rgb = im_bgr[..., ::-1]  # BGR to RGB
        
        PIL.Image.fromarray(im_rgb).save(output_path)
        
        web_path = f"/outputs/{output_filename}"

        return {
            "segmentation_map_path": web_path
        }

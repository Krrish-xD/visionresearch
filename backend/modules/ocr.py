"""Optical Character Recognition (OCR) Module using EasyOCR."""

from typing import Any
import time
import numpy as np
import PIL.Image

from core.base import BaseAnalyzer
from core.schemas import AnalysisResult, TextRegion, BoundingBox


class OCRAnalyzer(BaseAnalyzer):
    """Extracts text from images using EasyOCR."""

    name: str = "ocr"
    display_name: str = "Text Extraction (OCR)"
    estimated_vram_mb: int = 0   # CPU-only: no VRAM needed
    requires_gpu: bool = False   # Run on CPU to avoid float16/float32 mismatch
    stage: int = 2

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.model = None

    async def load_model(self, device: str = "cpu") -> None:
        if self.model is None:
            import easyocr
            # Force CPU — EasyOCR's GPU path mixes float16/float32 causing crashes.
            # CPU path uses float32 throughout and is still fast enough for single images.
            self.model = easyocr.Reader([self.lang], gpu=False)
            self._is_loaded = True

    async def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

    async def analyze(self, image: PIL.Image.Image, **kwargs: Any) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start_time = time.time()
        
        # Convert PIL Image (RGB) to numpy array for EasyOCR
        img_np = np.array(image.convert("RGB"))
        
        # Run OCR
        # result is a list of tuples: (bbox, text, prob)
        # bbox is a list of 4 points: [[x, y], [x, y], [x, y], [x, y]]
        results = self.model.readtext(img_np)
        
        text_regions: list[TextRegion] = []
        width, height = image.size
        
        for bbox, text, confidence in results:
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            
            box = BoundingBox(
                x_min=max(0.0, float(min(xs)) / width),
                y_min=max(0.0, float(min(ys)) / height),
                x_max=min(1.0, float(max(xs)) / width),
                y_max=min(1.0, float(max(ys)) / height),
            )
            
            text_regions.append(
                TextRegion(
                    content=text,
                    confidence=float(confidence),
                    bbox=box,
                    language=self.lang
                )
            )

        return {
            "text_regions": text_regions
        }

"""NSFW detection module."""

import asyncio
from typing import Any

from PIL import Image
import torch
import torch.nn.functional as F

from core.base import BaseAnalyzer
from core.schemas import NSFWResult


class NSFWDetector(BaseAnalyzer):
    """Detects unsafe content using ViT.
    
    Runs in Stage 1.
    """

    name = "nsfw"
    display_name = "Safety / NSFW"
    estimated_vram_mb = 200  # ViT base is ~300MB, lower in FP16
    requires_gpu = True
    stage = 1

    def __init__(self, model_id: str = "Falconsai/nsfw_image_detection"):
        super().__init__()
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.device = "cpu"

    async def load_model(self, device: str = "cpu") -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        
        self.device = "cuda" if device.startswith("cuda") else "cpu"
        # Always float32 — avoids Half/Float mismatch without meaningful speed cost on ViT
        self._dtype = torch.float32
        
        def _load():
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_id, torch_dtype=torch.float32
            ).to(self.device)
            self.model.eval()
            
        await asyncio.to_thread(_load)
        self._is_loaded = True

    async def analyze(self, image: Image.Image, **kwargs: Any) -> dict:
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")

        def _infer():
            raw = self.processor(images=image, return_tensors="pt")
            # Explicitly cast pixel_values only — avoids BatchEncoding.to() dtype issues
            inputs = {
                "pixel_values": raw["pixel_values"].to(self.device, self._dtype)
            }
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)[0]
                
            # Falconsai output: 0 = normal, 1 = nsfw
            nsfw_prob = float(probs[1].cpu())
            
            is_nsfw = nsfw_prob > 0.5
            cat = "explicit" if nsfw_prob > 0.8 else "suggestive" if is_nsfw else "safe"
            
            return {
                "nsfw": NSFWResult(
                    is_nsfw=is_nsfw,
                    category=cat,
                    confidence=nsfw_prob if is_nsfw else float(probs[0].cpu())
                )
            }

        return await asyncio.to_thread(_infer)

    async def unload_model(self) -> None:
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        self._is_loaded = False

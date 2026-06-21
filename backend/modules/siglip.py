"""SigLIP Module for Image Embeddings and Zero-Shot Classification."""

import asyncio
import time
import torch
from typing import Any
import PIL.Image

from core.base import BaseAnalyzer
from core.schemas import AnalysisResult
from config import settings


class SigLIPAnalyzer(BaseAnalyzer):
    """Generates image embeddings and zero-shot semantic tags using SigLIP."""

    name: str = "siglip"
    display_name: str = "Semantic Analysis (SigLIP)"
    estimated_vram_mb: int = 600
    requires_gpu: bool = True
    stage: int = 2

    def __init__(self, model_id: str = "google/siglip-base-patch16-224"):
        super().__init__()
        self.model_id = model_id
        self.model = None
        self.processor = None
        
        # General tags for zero-shot classification
        self.candidate_tags = [
            "indoor", "outdoor", "nature", "city", "portrait", "animal", "food", 
            "vehicle", "art", "technology", "sports", "night", "daylight", 
            "document", "screenshot", "meme", "illustration"
        ]

    async def load_model(self, device: str = "cpu") -> None:
        if self.model is None:
            # Import here to avoid slow startup
            from transformers import AutoProcessor, AutoModel
            
            self.device = "cuda" if settings.resolve_device() == "cuda" else "cpu"
            # Always use float32 for CPU; float16 only when CUDA available
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id, torch_dtype=dtype).to(self.device)
            self.model.eval()
            self._dtype = dtype

    async def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None

    async def analyze(self, image: PIL.Image.Image, **kwargs: Any) -> dict:
        if self.model is None:
            await self.load_model()

        start_time = time.time()
        
        # Prepare inputs for both text and image
        dtype = getattr(self, "_dtype", torch.float32)
        inputs = self.processor(
            text=self.candidate_tags, 
            images=image.convert("RGB"), 
            padding="max_length", 
            return_tensors="pt"
        )
        # Cast pixel_values to model dtype; keep input_ids as long
        inputs = {
            k: v.to(self.device, dtype) if v.dtype.is_floating_point else v.to(self.device)
            for k, v in inputs.items()
        }
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # 1. Image Embedding
            image_embeds = outputs.image_embeds
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            embedding_list = image_embeds[0].cpu().numpy().tolist()
            
            # 2. Zero-Shot Tags
            text_embeds = outputs.text_embeds
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            # In SigLIP, the logits are typically computed with a temperature/bias
            # outputs.logits_per_image already applies the learned temperature/bias
            logits = outputs.logits_per_image
            
            # SigLIP uses sigmoid instead of softmax for classification
            probs = torch.sigmoid(logits).squeeze()
            
            # If shape is 0-d (only 1 tag), handle it properly
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
                
            probs_np = probs.cpu().numpy()
            
            tags: list[str] = []
            for tag, prob in zip(self.candidate_tags, probs_np):
                if prob > 0.1:  # SigLIP sigmoid probabilities can be lower, 0.1 is a good threshold
                    tags.append(tag)

        return {
            "embedding": embedding_list,
            "tags": tags
        }

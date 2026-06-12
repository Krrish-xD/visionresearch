"""Scene captioning module using Florence-2."""

import asyncio
from typing import Any

from PIL import Image
import torch

from core.base import BaseAnalyzer


class SceneCaptioner(BaseAnalyzer):
    """Generates detailed scene captions using Microsoft Florence-2.
    
    Runs in Stage 3 (VLM stage).
    """

    name = "caption"
    display_name = "Scene Caption"
    estimated_vram_mb = 2000  # Florence-2-large in FP16
    requires_gpu = True
    stage = 3

    def __init__(self, model_id: str = "microsoft/Florence-2-base"):
        super().__init__()
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    async def load_model(self, device: str = "cpu") -> None:
        from transformers import AutoProcessor, AutoModelForCausalLM
        import transformers.dynamic_module_utils as dynamic_utils
        
        # Bypass strict flash_attn check in Florence-2 remote code
        if hasattr(dynamic_utils, "check_imports"):
            dynamic_utils.check_imports = lambda filename: []
        
        self.device = "cuda" if device.startswith("cuda") else "cpu"
        
        def _load():
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            # Fallback if tokenizer failed to load implicitly
            if not getattr(self.processor, "tokenizer", None):
                from transformers import AutoTokenizer
                self.processor.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=self.dtype, trust_remote_code=True
            ).to(self.device)
            
        await asyncio.to_thread(_load)
        self._is_loaded = True

    async def analyze(self, image: Image.Image, **kwargs: Any) -> dict:
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")

        def _infer():
            # Standard caption
            prompt_brief = "<CAPTION>"
            inputs_brief = self.processor(
                text=prompt_brief, images=image, return_tensors="pt"
            ).to(self.device, self.dtype)
            
            generated_ids_brief = self.model.generate(
                input_ids=inputs_brief["input_ids"],
                pixel_values=inputs_brief["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
            caption = self.processor.tokenizer.batch_decode(
                generated_ids_brief, skip_special_tokens=False
            )[0]
            
            # Detailed description
            prompt_det = "<MORE_DETAILED_CAPTION>"
            inputs_det = self.processor(
                text=prompt_det, images=image, return_tensors="pt"
            ).to(self.device, self.dtype)
            
            generated_ids_det = self.model.generate(
                input_ids=inputs_det["input_ids"],
                pixel_values=inputs_det["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
            detailed = self.processor.tokenizer.batch_decode(
                generated_ids_det, skip_special_tokens=False
            )[0]
            
            # Post-process Florence's special tokens
            caption = self.processor.post_process_generation(
                caption, task=prompt_brief, image_size=(image.width, image.height)
            )[prompt_brief]
            
            detailed = self.processor.post_process_generation(
                detailed, task=prompt_det, image_size=(image.width, image.height)
            )[prompt_det]
            
            return {
                "caption": caption,
                "detailed_description": detailed
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

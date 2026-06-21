"""Base analyzer abstract class.

All analysis modules must inherit from BaseAnalyzer and implement
the load_model, analyze, and unload_model methods.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from core.schemas import ModuleResult


class BaseAnalyzer(ABC):
    """Abstract base class for all analysis modules.

    Each module encapsulates a single analysis capability (e.g., object detection,
    OCR, captioning). Modules report their VRAM requirements so the ModelManager
    can orchestrate loading/unloading efficiently.

    Lifecycle:
        1. __init__() — set metadata (name, display_name, vram, etc.)
        2. load_model() — load model weights into GPU/CPU
        3. analyze() — run inference on an image (can be called multiple times)
        4. unload_model() — release model from memory
    """

    # Module identity
    name: str = ""
    display_name: str = ""

    # Resource requirements
    estimated_vram_mb: int = 0
    requires_gpu: bool = False

    # Execution priority (lower = earlier in pipeline)
    # Stage 0: CPU-only, Stage 1: lightweight GPU, Stage 2: OCR+embed, Stage 3: VLM, Stage 4: heavy
    stage: int = 0

    # Dependencies — list of module names that must run before this one
    dependencies: list[str] = []

    def __init__(self) -> None:
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""
        return self._is_loaded

    @abstractmethod
    async def load_model(self, device: str = "cpu") -> None:
        """Load model weights into memory.

        Args:
            device: Target device ("cuda", "cpu", or "cuda:N").
        """
        ...

    @abstractmethod
    async def analyze(self, image: Image.Image, **kwargs: Any) -> dict:
        """Run analysis on an image.

        Args:
            image: PIL Image in RGB mode.
            **kwargs: Module-specific parameters (e.g., confidence_threshold).

        Returns:
            Dictionary of results. The keys depend on the module type and will
            be merged into the top-level AnalysisResult by the pipeline.
        """
        ...

    @abstractmethod
    async def unload_model(self) -> None:
        """Release model from memory.

        Should set self._is_loaded = False and clear GPU memory.
        """
        ...

    async def run(self, image: Image.Image, **kwargs: Any) -> ModuleResult:
        """Execute analysis with timing and error handling.

        This is the main entry point called by the PipelineOrchestrator.
        Wraps analyze() with timing, error handling, and result packaging.
        """
        start = time.perf_counter()
        try:
            try:
                data = await self.analyze(image, **kwargs)
            except RuntimeError as e:
                err_msg = str(e).lower()
                if "allocate" in err_msg or "memory" in err_msg or "oom" in err_msg:
                    import logging
                    import torch
                    import gc
                    import asyncio
                    
                    logger = logging.getLogger(__name__)
                    logger.warning(f"OOM during {self.name} analyze(). Trying to recover...")
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    await asyncio.sleep(2)
                    
                    logger.info(f"Retrying {self.name} analyze() after clearing VRAM...")
                    data = await self.analyze(image, **kwargs)
                else:
                    raise e
                    
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ModuleResult(
                module_name=self.name,
                display_name=self.display_name,
                success=True,
                timing_ms=round(elapsed_ms, 2),
                data=data,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ModuleResult(
                module_name=self.name,
                display_name=self.display_name,
                success=False,
                error=str(e),
                timing_ms=round(elapsed_ms, 2),
            )

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "unloaded"
        return f"<{self.__class__.__name__}({self.name}) [{status}] vram={self.estimated_vram_mb}MB>"

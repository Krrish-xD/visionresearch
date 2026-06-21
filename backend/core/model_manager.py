"""GPU memory manager for loading/unloading models within a VRAM budget.

The ModelManager tracks which models are loaded, their VRAM consumption,
and handles intelligent eviction when the budget is exceeded.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from core.base import BaseAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class LoadedModelInfo:
    """Tracks a loaded model's metadata."""

    module: BaseAnalyzer
    vram_mb: int
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0

    def touch(self) -> None:
        """Update last-used timestamp and increment use count."""
        self.last_used = time.time()
        self.use_count += 1


class ModelManager:
    """Central GPU memory manager.

    Manages model lifecycle (load/unload) within a configurable VRAM budget.
    Uses LRU eviction when space is needed for new models.

    Usage:
        manager = ModelManager(vram_budget_mb=11_000, device="cuda")
        await manager.ensure_loaded(my_detector)
        # ... use my_detector ...
        await manager.ensure_unloaded(my_detector)
    """

    def __init__(self, vram_budget_mb: int = 11_000, device: str = "cpu") -> None:
        self.vram_budget_mb = vram_budget_mb
        self.device = device
        self._loaded: dict[str, LoadedModelInfo] = {}
        self._vram_used_mb: int = 0

    @property
    def vram_used_mb(self) -> int:
        """Current estimated VRAM usage in MB."""
        return self._vram_used_mb

    @property
    def vram_available_mb(self) -> int:
        """Estimated available VRAM in MB."""
        return self.vram_budget_mb - self._vram_used_mb

    @property
    def loaded_modules(self) -> list[str]:
        """Names of currently loaded modules."""
        return list(self._loaded.keys())

    def is_loaded(self, module_name: str) -> bool:
        """Check if a module's model is currently loaded."""
        return module_name in self._loaded

    async def ensure_loaded(self, module: BaseAnalyzer) -> None:
        """Load a model, evicting others if necessary to stay within budget.

        If the module is already loaded, just updates its last-used timestamp.
        """
        if module.name in self._loaded:
            self._loaded[module.name].touch()
            logger.debug(f"Model '{module.name}' already loaded, touching LRU")
            return

        # Check if we have enough VRAM
        needed = module.estimated_vram_mb
        while self._vram_used_mb + needed > self.vram_budget_mb:
            if not self._loaded:
                logger.warning(
                    f"Cannot fit '{module.name}' ({needed}MB) into "
                    f"VRAM budget ({self.vram_budget_mb}MB) — no models to evict."
                )
                break
            await self._evict_lru()

        # Load the model
        logger.info(
            f"Loading '{module.name}' ({needed}MB) on {self.device} "
            f"[{self._vram_used_mb}MB / {self.vram_budget_mb}MB used]"
        )
        try:
            await module.load_model(device=self.device)
        except RuntimeError as e:
            err_msg = str(e).lower()
            if "allocate" in err_msg or "memory" in err_msg or "oom" in err_msg:
                logger.warning(f"OOM during load of {module.name}. Trying to recover...")
                import torch
                import gc
                import asyncio
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                await asyncio.sleep(2)
                logger.info(f"Retrying load of {module.name} after clearing VRAM...")
                await module.load_model(device=self.device)
            else:
                raise
        self._loaded[module.name] = LoadedModelInfo(
            module=module,
            vram_mb=needed,
        )
        self._vram_used_mb += needed
        logger.info(
            f"Loaded '{module.name}' — "
            f"VRAM now {self._vram_used_mb}MB / {self.vram_budget_mb}MB"
        )

    async def ensure_unloaded(self, module: BaseAnalyzer) -> None:
        """Unload a specific module's model."""
        if module.name not in self._loaded:
            return

        info = self._loaded[module.name]
        logger.info(f"Unloading '{module.name}' ({info.vram_mb}MB)")
        await module.unload_model()
        self._vram_used_mb -= info.vram_mb
        del self._loaded[module.name]
        self._clear_gpu_cache()
        logger.info(
            f"Unloaded '{module.name}' — "
            f"VRAM now {self._vram_used_mb}MB / {self.vram_budget_mb}MB"
        )

    async def unload_all(self) -> None:
        """Unload all currently loaded models."""
        module_names = list(self._loaded.keys())
        for name in module_names:
            info = self._loaded[name]
            await info.module.unload_model()
            self._vram_used_mb -= info.vram_mb
            del self._loaded[name]
        self._clear_gpu_cache()
        logger.info("All models unloaded, VRAM cleared.")

    async def _evict_lru(self) -> None:
        """Evict the least recently used model to free VRAM."""
        if not self._loaded:
            return

        lru_name = min(self._loaded, key=lambda k: self._loaded[k].last_used)
        info = self._loaded[lru_name]
        logger.info(
            f"Evicting LRU model '{lru_name}' ({info.vram_mb}MB, "
            f"last used {time.time() - info.last_used:.1f}s ago)"
        )
        await info.module.unload_model()
        self._vram_used_mb -= info.vram_mb
        del self._loaded[lru_name]
        self._clear_gpu_cache()

    def _clear_gpu_cache(self) -> None:
        """Clear CUDA memory cache and Python garbage collect."""
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    def get_status(self) -> dict:
        """Return current memory status for monitoring."""
        return {
            "vram_used_mb": self._vram_used_mb,
            "vram_budget_mb": self.vram_budget_mb,
            "vram_available_mb": self.vram_available_mb,
            "loaded_models": [
                {
                    "name": name,
                    "vram_mb": info.vram_mb,
                    "use_count": info.use_count,
                }
                for name, info in self._loaded.items()
            ],
        }

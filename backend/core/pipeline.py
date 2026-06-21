"""Pipeline orchestrator — coordinates multi-stage analysis execution.

The orchestrator groups modules by execution stage, manages model
loading/unloading between stages, and streams results via SSE events.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from typing import Any, AsyncIterator

from PIL import Image

from core.base import BaseAnalyzer
from core.events import (
    SSEEvent,
    module_complete_event,
    module_error_event,
    module_start_event,
    pipeline_complete_event,
    pipeline_error_event,
    pipeline_start_event,
)
from core.model_manager import ModelManager
from core.schemas import AnalysisResult, ModuleResult

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """Registry of all available analysis modules."""

    def __init__(self) -> None:
        self._modules: dict[str, BaseAnalyzer] = {}

    def register(self, module: BaseAnalyzer) -> None:
        """Register an analysis module."""
        if not module.name:
            raise ValueError(f"Module {module.__class__.__name__} has no name set")
        self._modules[module.name] = module
        logger.info(f"Registered module: {module.name} ({module.display_name})")

    def get(self, name: str) -> BaseAnalyzer | None:
        """Get a module by name."""
        return self._modules.get(name)

    def get_all(self) -> dict[str, BaseAnalyzer]:
        """Get all registered modules."""
        return dict(self._modules)

    def get_by_names(self, names: list[str]) -> list[BaseAnalyzer]:
        """Get modules by names, preserving order."""
        return [self._modules[n] for n in names if n in self._modules]

    @property
    def available_modules(self) -> list[str]:
        """List of all registered module names."""
        return list(self._modules.keys())


class PipelineOrchestrator:
    """Orchestrates multi-stage analysis pipeline with GPU memory management.

    The pipeline groups modules by their execution stage and runs them
    sequentially (stages) with intra-stage parallelism where VRAM allows.
    Results are yielded as SSE events for streaming to the frontend.

    Execution stages:
        Stage 0: CPU-only modules (EXIF, colors) — run in parallel, no GPU
        Stage 1: Lightweight GPU (YOLO, NSFW, pose) — loaded together
        Stage 2: Medium GPU (OCR, embedding) — loaded together
        Stage 3: VLM (Florence-2 caption) — loaded alone
        Stage 4: Heavy GPU (depth, face, SAM) — loaded one at a time
    """

    def __init__(self, registry: ModuleRegistry, model_manager: ModelManager) -> None:
        self.registry = registry
        self.model_manager = model_manager

    def _group_by_stage(self, modules: list[BaseAnalyzer]) -> dict[int, list[BaseAnalyzer]]:
        """Group modules by their execution stage."""
        groups: dict[int, list[BaseAnalyzer]] = defaultdict(list)
        for module in modules:
            groups[module.stage].append(module)
        return dict(sorted(groups.items()))

    async def analyze(
        self,
        image: Image.Image,
        enabled_modules: list[str] | None = None,
        confidence_threshold: float = 0.25,
    ) -> AsyncIterator[SSEEvent]:
        """Run the full analysis pipeline, yielding SSE events.

        Args:
            image: PIL Image in RGB mode.
            enabled_modules: List of module names to run. If None, uses all registered.
            confidence_threshold: Minimum confidence for detections.

        Yields:
            SSEEvent objects for streaming to the frontend.
        """
        task_id = str(uuid.uuid4())
        pipeline_start = time.perf_counter()

        # Resolve which modules to run
        if enabled_modules:
            modules = self.registry.get_by_names(enabled_modules)
        else:
            modules = list(self.registry.get_all().values())

        if not modules:
            yield pipeline_error_event(task_id, "No analysis modules enabled")
            return

        # Group by stage
        stage_groups = self._group_by_stage(modules)
        total_stages = len(stage_groups)
        module_names = [m.name for m in modules]

        yield pipeline_start_event(task_id, module_names, total_stages)

        all_results: list[ModuleResult] = []
        accumulated_data: dict[str, Any] = {}

        # Execute stage by stage
        for stage_num, stage_modules in stage_groups.items():
            logger.info(
                f"Pipeline stage {stage_num}: "
                f"{[m.name for m in stage_modules]}"
            )

            # Load all models for this stage
            for module in stage_modules:
                if module.requires_gpu or module.name == "ocr":
                    try:
                        await self.model_manager.ensure_loaded(module)
                    except Exception as e:
                        logger.error(f"Failed to pre-load model for {module.name}: {e}")

            # Run modules in this stage (parallel for CPU modules, sequential otherwise)
            if stage_num == 0:
                # CPU modules can run in parallel
                for module in stage_modules:
                    yield module_start_event(module.name, module.display_name, stage_num)
                results = await self._run_parallel(
                    stage_modules, image, confidence_threshold
                )
            else:
                # GPU modules run sequentially to avoid memory contention
                # Yield start event right before _run_sequential starts each module?
                # For sequential, we can just yield them all as "running" for the stage,
                # or better yet, we should probably yield them individually in _run_sequential.
                # Since we can't easily yield from _run_sequential without changing it to an async generator,
                # we'll just mark them all as running when the stage begins.
                for module in stage_modules:
                    yield module_start_event(module.name, module.display_name, stage_num)
                results = await self._run_sequential(
                    stage_modules, image, confidence_threshold
                )

            # Yield results and accumulate data
            for result in results:
                all_results.append(result)
                if result.success:
                    accumulated_data.update(result.data)
                    yield module_complete_event(
                        result.module_name,
                        result.display_name,
                        result.timing_ms,
                        result.data,
                    )
                else:
                    yield module_error_event(
                        result.module_name,
                        result.display_name,
                        result.error or "Unknown error",
                    )

            # Unload GPU models from this stage (free VRAM for next stage)
            for module in stage_modules:
                if module.requires_gpu or module.name == "ocr":
                    await self.model_manager.ensure_unloaded(module)

        # Build final result
        total_time_ms = (time.perf_counter() - pipeline_start) * 1000
        final_result = self._build_final_result(
            task_id=task_id,
            filename=accumulated_data.get("_filename", "unknown"),
            all_results=all_results,
            accumulated_data=accumulated_data,
            total_time_ms=total_time_ms,
        )

        yield pipeline_complete_event(
            task_id,
            total_time_ms,
            final_result.model_dump(),
        )

    async def _run_parallel(
        self,
        modules: list[BaseAnalyzer],
        image: Image.Image,
        confidence_threshold: float,
    ) -> list[ModuleResult]:
        """Run multiple modules in parallel (for CPU-only modules)."""
        tasks = [
            module.run(image, confidence_threshold=confidence_threshold)
            for module in modules
        ]
        return list(await asyncio.gather(*tasks))

    async def _run_sequential(
        self,
        modules: list[BaseAnalyzer],
        image: Image.Image,
        confidence_threshold: float,
    ) -> list[ModuleResult]:
        """Run modules sequentially (for GPU modules)."""
        results = []
        for module in modules:
            result = await module.run(image, confidence_threshold=confidence_threshold)
            results.append(result)
        return results

    def _build_final_result(
        self,
        task_id: str,
        filename: str,
        all_results: list[ModuleResult],
        accumulated_data: dict[str, Any],
        total_time_ms: float,
    ) -> AnalysisResult:
        """Build the final AnalysisResult from all module outputs."""
        from datetime import datetime, timezone

        return AnalysisResult(
            image_id=task_id,
            filename=filename,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_processing_time_ms=round(total_time_ms, 2),
            modules_executed=[r.module_name for r in all_results if r.success],
            # Map accumulated data to result fields
            metadata=accumulated_data.get("metadata"),
            caption=accumulated_data.get("caption"),
            detailed_description=accumulated_data.get("detailed_description"),
            tags=accumulated_data.get("tags", []),
            objects=accumulated_data.get("objects", []),
            text_regions=accumulated_data.get("text_regions", []),
            faces=accumulated_data.get("faces", []),
            poses=accumulated_data.get("poses", []),
            colors=accumulated_data.get("colors", []),
            nsfw=accumulated_data.get("nsfw"),
            depth_map_path=accumulated_data.get("depth_map_path"),
            segmentation_map_path=accumulated_data.get("segmentation_map_path"),
            embedding=accumulated_data.get("embedding"),
            module_timings={r.module_name: r.timing_ms for r in all_results},
            module_results=all_results,
        )

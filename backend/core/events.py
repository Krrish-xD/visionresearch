"""SSE event types for streaming analysis progress to the frontend."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel


class EventType(str, Enum):
    """Types of SSE events sent during analysis."""

    # Pipeline lifecycle
    PIPELINE_START = "pipeline_start"
    PIPELINE_COMPLETE = "pipeline_complete"
    PIPELINE_ERROR = "pipeline_error"

    # Module lifecycle
    MODULE_START = "module_start"
    MODULE_PROGRESS = "module_progress"
    MODULE_COMPLETE = "module_complete"
    MODULE_ERROR = "module_error"

    # GPU memory updates
    MEMORY_UPDATE = "memory_update"


class SSEEvent(BaseModel):
    """A single Server-Sent Event payload."""

    event: EventType
    module: str = ""
    data: dict[str, Any] = {}

    def to_sse(self) -> str:
        """Format as an SSE data line.

        Returns a string like: 'data: {"event": "module_complete", ...}\n\n'
        """
        payload = {
            "event": self.event.value,
            "module": self.module,
            **self.data,
        }
        return f"data: {json.dumps(payload)}\n\n"

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for JSON serialization."""
        dumped = self.model_dump(mode="json")
        return {
            "event": dumped["event"],
            "module": dumped["module"],
            **dumped.get("data", {}),
        }


# =============================================================================
# Event Factory Helpers
# =============================================================================


def pipeline_start_event(
    task_id: str, modules: list[str], total_stages: int
) -> SSEEvent:
    """Create a pipeline start event."""
    return SSEEvent(
        event=EventType.PIPELINE_START,
        data={
            "task_id": task_id,
            "results": {"modules": modules},
            "total_stages": total_stages,
        },
    )


def pipeline_complete_event(
    task_id: str, total_time_ms: float, results: dict
) -> SSEEvent:
    """Create a pipeline completion event."""
    return SSEEvent(
        event=EventType.PIPELINE_COMPLETE,
        data={
            "task_id": task_id,
            "total_time_ms": round(total_time_ms, 2),
            "results": results,
        },
    )


def pipeline_error_event(task_id: str, error: str) -> SSEEvent:
    """Create a pipeline error event."""
    return SSEEvent(
        event=EventType.PIPELINE_ERROR,
        data={"task_id": task_id, "error": error},
    )


def module_start_event(module_name: str, display_name: str, stage: int) -> SSEEvent:
    """Create a module start event."""
    return SSEEvent(
        event=EventType.MODULE_START,
        module=module_name,
        data={"display_name": display_name, "stage": stage, "status": "running"},
    )


def module_complete_event(
    module_name: str, display_name: str, timing_ms: float, results: dict
) -> SSEEvent:
    """Create a module completion event."""
    return SSEEvent(
        event=EventType.MODULE_COMPLETE,
        module=module_name,
        data={
            "display_name": display_name,
            "status": "complete",
            "timing_ms": round(timing_ms, 2),
            "results": results,
        },
    )


def module_error_event(module_name: str, display_name: str, error: str) -> SSEEvent:
    """Create a module error event."""
    return SSEEvent(
        event=EventType.MODULE_ERROR,
        module=module_name,
        data={
            "display_name": display_name,
            "status": "error",
            "error": error,
        },
    )


def memory_update_event(vram_used_mb: int, vram_budget_mb: int) -> SSEEvent:
    """Create a GPU memory update event."""
    return SSEEvent(
        event=EventType.MEMORY_UPDATE,
        data={
            "vram_used_mb": vram_used_mb,
            "vram_budget_mb": vram_budget_mb,
        },
    )

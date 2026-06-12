"""API routes and SSE endpoints."""

import io
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile, BackgroundTasks
from sse_starlette.sse import EventSourceResponse
from PIL import Image

from backend.config import settings
from backend.core.pipeline import PipelineOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory store for active analysis tasks (in production, use Redis)
active_tasks: dict[str, dict] = {}


@router.post("/analyze")
async def start_analysis(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
):
    """Upload an image and start analysis. Returns a task ID."""
    task_id = str(uuid.uuid4())
    
    # Read image data
    contents = await image.read()
    
    # Save original image to disk
    ext = Path(image.filename or "image.jpg").suffix
    filepath = settings.upload_dir / f"{task_id}{ext}"
    with open(filepath, "wb") as f:
        f.write(contents)
        
    # Open with PIL
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Resize if too large to prevent OOM
    if max(pil_img.size) > settings.max_image_dimension:
        pil_img.thumbnail(
            (settings.max_image_dimension, settings.max_image_dimension), 
            Image.Resampling.LANCZOS
        )
    
    # Store task metadata
    active_tasks[task_id] = {
        "filename": image.filename,
        "image": pil_img,
        "status": "pending"
    }
    
    return {"task_id": task_id, "status": "pending"}


@router.get("/analyze/{task_id}/stream")
async def stream_analysis(task_id: str, request: Request):
    """SSE endpoint to stream analysis progress and results."""
    if task_id not in active_tasks:
        return {"error": "Task not found"}, 404
        
    task_data = active_tasks[task_id]
    pil_img = task_data["image"]
    
    orchestrator: PipelineOrchestrator = request.app.state.orchestrator
    
    async def event_generator():
        try:
            # We pass the original filename inside kwargs or accumulate it
            # For simplicity, we just use the orchestrator generator
            async for event in orchestrator.analyze(pil_img, enabled_modules=settings.default_modules):
                # If client disconnects, break
                if await request.is_disconnected():
                    logger.info(f"Client disconnected from task {task_id}")
                    break
                
                # yield the event dict, sse-starlette handles conversion to 'data: ...'
                yield event.to_dict()
                
        except Exception as e:
            logger.exception(f"Error in pipeline execution for task {task_id}")
            yield {
                "event": "pipeline_error",
                "module": "pipeline",
                "task_id": task_id,
                "error": str(e)
            }
        finally:
            # Cleanup
            if task_id in active_tasks:
                del active_tasks[task_id]

    return EventSourceResponse(event_generator())


@router.get("/status")
async def system_status(request: Request):
    """Get system and GPU memory status."""
    manager = request.app.state.model_manager
    return manager.get_status()

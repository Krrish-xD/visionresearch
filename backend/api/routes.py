"""API routes and SSE endpoints."""

import io
import uuid
import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile, BackgroundTasks, Depends
from sse_starlette.sse import EventSourceResponse
from PIL import Image
from sqlalchemy.orm import Session

from config import settings
from core.pipeline import PipelineOrchestrator
from core.database import get_db
from core.models import AnalysisTask

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory store for active analysis tasks to avoid reloading PIL images
active_tasks: dict[str, dict] = {}


@router.post("/analyze")
async def start_analysis(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
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
    
    # Store task metadata in memory for streaming
    active_tasks[task_id] = {
        "filename": image.filename,
        "image": pil_img,
        "status": "pending"
    }
    
    # Store task metadata in database for history
    db_task = AnalysisTask(
        id=task_id,
        filename=image.filename or "image.jpg",
        filepath=str(filepath),
        status="pending"
    )
    db.add(db_task)
    db.commit()
    
    return {"task_id": task_id, "status": "pending"}


@router.get("/analyze/{task_id}/stream")
async def stream_analysis(task_id: str, request: Request, db: Session = Depends(get_db)):
    """SSE endpoint to stream analysis progress and results."""
    if task_id not in active_tasks:
        # Check if it completed previously
        db_task = db.query(AnalysisTask).filter(AnalysisTask.id == task_id).first()
        if db_task and db_task.status == "complete":
            # Return a fast completion event
            async def fast_complete():
                yield {"data": json.dumps({
                    "event": "pipeline_complete",
                    "module": "",
                    "task_id": task_id,
                    "total_time_ms": db_task.total_time_ms,
                    "results": db_task.final_results
                })}
            return EventSourceResponse(fast_complete())
        return {"error": "Task not found or not running"}, 404
        
    task_data = active_tasks[task_id]
    pil_img = task_data["image"]
    
    orchestrator: PipelineOrchestrator = request.app.state.orchestrator
    
    async def event_generator():
        # Re-fetch session locally in generator just in case since generators span multiple ticks
        # Actually since we yielded we can just use the outer db safely with fastapi Depends, 
        # but to be totally safe we should query only when needed.
        try:
            async for event in orchestrator.analyze(pil_img, enabled_modules=settings.default_modules):
                if await request.is_disconnected():
                    logger.info(f"Client disconnected from task {task_id}")
                    break
                
                # Check for pipeline completion to update DB
                if event.event.value == "pipeline_complete":
                    db_task = db.query(AnalysisTask).filter(AnalysisTask.id == task_id).first()
                    if db_task:
                        db_task.status = "complete"
                        db_task.total_time_ms = event.data.get("total_time_ms")
                        db_task.final_results = event.data.get("results")
                        db.commit()
                
                yield {
                    "data": json.dumps(event.to_dict())
                }
                
        except Exception as e:
            logger.exception(f"Error in pipeline execution for task {task_id}")
            db_task = db.query(AnalysisTask).filter(AnalysisTask.id == task_id).first()
            if db_task:
                db_task.status = "error"
                db_task.error = str(e)
                db.commit()
                
            yield {
                "data": json.dumps({
                    "event": "pipeline_error",
                    "module": "pipeline",
                    "task_id": task_id,
                    "error": str(e)
                })
            }
        finally:
            # Cleanup
            if task_id in active_tasks:
                del active_tasks[task_id]

    return EventSourceResponse(event_generator())


@router.get("/history")
async def get_history(db: Session = Depends(get_db)):
    """Get history of all analysis tasks."""
    tasks = db.query(AnalysisTask).order_by(AnalysisTask.created_at.desc()).all()
    return [{
        "id": task.id,
        "filename": task.filename,
        "status": task.status,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "total_time_ms": task.total_time_ms
    } for task in tasks]


@router.get("/history/{task_id}")
async def get_history_detail(task_id: str, db: Session = Depends(get_db)):
    """Get detailed results for a specific historical task."""
    task = db.query(AnalysisTask).filter(AnalysisTask.id == task_id).first()
    if not task:
        return {"error": "Task not found"}, 404
        
    return {
        "id": task.id,
        "filename": task.filename,
        "status": task.status,
        "error": task.error,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "total_time_ms": task.total_time_ms,
        "results": task.final_results
    }


from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str

@router.post("/analyze/{task_id}/ask")
async def ask_question(task_id: str, payload: AskRequest, request: Request, db: Session = Depends(get_db)):
    """Ask a question about the image using the VLM."""
    # 1. Get the image
    pil_img = None
    if task_id in active_tasks:
        pil_img = active_tasks[task_id]["image"]
    else:
        # Load from DB
        db_task = db.query(AnalysisTask).filter(AnalysisTask.id == task_id).first()
        if not db_task:
            return {"error": "Task not found"}, 404
        if not Path(db_task.filepath).exists():
            return {"error": "Original image file no longer exists on disk"}, 404
            
        with open(db_task.filepath, "rb") as f:
            pil_img = Image.open(io.BytesIO(f.read())).convert("RGB")
            
    if not pil_img:
        return {"error": "Could not load image"}, 500
        
    # 2. Get the caption module via orchestrator/registry
    manager = request.app.state.model_manager
    orchestrator = request.app.state.orchestrator
    caption_module = orchestrator.registry.get("caption")
    
    if not caption_module:
        return {"error": "Caption module not found"}, 500
        
    # 3. Ensure it's loaded
    await manager.ensure_loaded(caption_module)
    
    # 4. Ask the question
    try:
        # Note: We must ensure ask_question is available on the module
        if not hasattr(caption_module, "ask_question"):
            return {"error": "Caption module does not support VQA"}, 400
            
        answer = await caption_module.ask_question(pil_img, payload.question)
        return {"answer": answer}
    except Exception as e:
        logger.exception("Error during VQA")
        return {"error": str(e)}, 500


@router.get("/status")
async def system_status(request: Request):
    """Get system and GPU memory status."""
    manager = request.app.state.model_manager
    return manager.get_status()

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


@router.post("/analyze/video")
async def start_video_analysis(
    request: Request,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a video and extract frames for batch analysis. Returns a list of task IDs."""
    import cv2
    import os
    
    parent_task_id = str(uuid.uuid4())
    
    # Save video to disk
    ext = Path(video.filename or "video.mp4").suffix
    video_path = settings.upload_dir / f"{parent_task_id}{ext}"
    contents = await video.read()
    with open(video_path, "wb") as f:
        f.write(contents)
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Failed to open video file")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30 # fallback
        
    frame_interval = int(round(fps)) # 1 frame per second
    
    frame_count = 0
    extracted_tasks = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Save frame and create task
            task_id = str(uuid.uuid4())
            frame_filename = f"{parent_task_id}_frame_{frame_count // frame_interval}.jpg"
            frame_filepath = settings.upload_dir / frame_filename
            
            # cv2 reads in BGR, save as JPG
            cv2.imwrite(str(frame_filepath), frame)
            
            # Read back as PIL for memory cache
            pil_img = Image.open(frame_filepath).convert("RGB")
            
            # Resize if too large
            if max(pil_img.size) > settings.max_image_dimension:
                pil_img.thumbnail(
                    (settings.max_image_dimension, settings.max_image_dimension), 
                    Image.Resampling.LANCZOS
                )
                
            active_tasks[task_id] = {
                "filename": frame_filename,
                "image": pil_img,
                "status": "pending"
            }
            
            db_task = AnalysisTask(
                id=task_id,
                parent_task_id=parent_task_id,
                filename=frame_filename,
                filepath=str(frame_filepath),
                status="pending"
            )
            db.add(db_task)
            extracted_tasks.append(task_id)
            
        frame_count += 1
        
    cap.release()
    db.commit()
    
    return {
        "parent_task_id": parent_task_id,
        "task_ids": extracted_tasks,
        "status": "pending"
    }


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
        "results": task.final_results,
        "chat_history": task.chat_history or []
    }


@router.get("/compare")
async def compare_images(task1_id: str, task2_id: str, db: Session = Depends(get_db)):
    """Compare the results and visual similarity of two analyzed images."""
    task1 = db.query(AnalysisTask).filter(AnalysisTask.id == task1_id).first()
    task2 = db.query(AnalysisTask).filter(AnalysisTask.id == task2_id).first()
    
    if not task1 or not task2:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="One or both tasks not found")
        
    res1 = task1.final_results or {}
    res2 = task2.final_results or {}
    
    comparison = {
        "task1": {"id": task1.id, "filename": task1.filename},
        "task2": {"id": task2.id, "filename": task2.filename},
        "embedding_similarity": None,
        "object_intersection": [],
        "tag_intersection": []
    }
    
    # 1. Embedding Similarity (SigLIP)
    emb1 = res1.get("embedding")
    emb2 = res2.get("embedding")
    if emb1 and emb2:
        import numpy as np
        vec1 = np.array(emb1)
        vec2 = np.array(emb2)
        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        comparison["embedding_similarity"] = float(similarity)
        
    # 2. Object Intersection
    objs1 = {obj.get("label") for obj in res1.get("objects", [])}
    objs2 = {obj.get("label") for obj in res2.get("objects", [])}
    comparison["object_intersection"] = list(objs1.intersection(objs2))
    
    # 3. Tag Intersection
    tags1 = set(res1.get("tags", []))
    tags2 = set(res2.get("tags", []))
    comparison["tag_intersection"] = list(tags1.intersection(tags2))
    
    return comparison


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
        
        # Persist to chat history in DB
        db_task = db.query(AnalysisTask).filter(AnalysisTask.id == task_id).first()
        if db_task:
            history = list(db_task.chat_history or [])
            history.append({"role": "user", "content": payload.question})
            history.append({"role": "assistant", "content": answer})
            db_task.chat_history = history
            db.commit()
        
        return {
            "answer": answer,
            "history": db_task.chat_history if db_task else []
        }
    except Exception as e:
        logger.exception("Error during VQA")
        return {"error": str(e)}, 500


@router.get("/status")
async def system_status(request: Request):
    """Get system and GPU memory status."""
    manager = request.app.state.model_manager
    return manager.get_status()

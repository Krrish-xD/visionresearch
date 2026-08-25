from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from PIL import Image
import io
import typing

from src.vlm.server import vlm_engine
from src.api.schemas import TokenLogprob, GenerationResponse, ModelInfo, ModelLoadRequest, ModelStatusResponse

router = APIRouter(prefix="/api")

@router.get("/models", response_model=typing.List[ModelInfo])
async def list_models():
    """Returns available local models purely from the weights directory."""
    return vlm_engine.list_available_models()

@router.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Returns the currently loaded model ID."""
    return ModelStatusResponse(
        active_model_id=vlm_engine.current_model_id,
        is_loaded=vlm_engine.model is not None
    )

@router.post("/model/load")
async def load_model(req: ModelLoadRequest):
    """Explicitly loads a model into memory."""
    # Resolve ID to path
    models = vlm_engine.list_available_models()
    model_path = next((m["path"] for m in models if m["id"] == req.model_id), req.model_id)
    
    try:
        vlm_engine.load_model(model_path)
        return {"status": "success", "message": f"Loaded {req.model_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@router.post("/generate", response_model=GenerationResponse)
async def generate_logprobs(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model_id: str = Form(...),
    temperature: float = Form(1.0),
    top_p: float = Form(1.0),
    top_k: int = Form(50),
    max_tokens: int = Form(100)
):
    # Resolve ID to path
    models = vlm_engine.list_available_models()
    model_path = next((m["path"] for m in models if m["id"] == model_id), model_id)
    
    try:
        vlm_engine.load_model(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    try:
        result = vlm_engine.generate_with_logprobs(
            pil_image, 
            prompt, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k, 
            max_tokens=max_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return GenerationResponse(
        full_text=result["full_text"],
        tokens=[TokenLogprob(**t) for t in result["tokens"]]
    )

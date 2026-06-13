"""FastAPI app initialization and middleware setup."""

import os
# Force transformers to use PyTorch to prevent TensorFlow/Paddle conflicts
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Fix PaddleOCR crashes on CPU with PIR/oneDNN
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import warnings
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import router
from config import settings
from core.model_manager import ModelManager
from core.pipeline import ModuleRegistry, PipelineOrchestrator
from modules.metadata import MetadataAnalyzer
from modules.colors import ColorPaletteAnalyzer
from modules.objects import ObjectDetector
from modules.caption import SceneCaptioner
from modules.nsfw import NSFWDetector
from modules.ocr import OCRAnalyzer
from modules.faces import FaceAnalyzer
from modules.pose import PoseAnalyzer
from modules.depth import DepthAnalyzer
from modules.siglip import SigLIPAnalyzer
from modules.segmentation import SegmentationAnalyzer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application."""
    # Ensure directories exist
    settings.ensure_dirs()
    
    # Initialize GPU Memory Manager
    device = settings.resolve_device()
    model_manager = ModelManager(
        vram_budget_mb=settings.vram_budget_mb,
        device=device
    )
    
    # Initialize and register modules
    registry = ModuleRegistry()
    
    # Phase 1 Modules
    registry.register(MetadataAnalyzer())
    registry.register(ColorPaletteAnalyzer())
    registry.register(ObjectDetector())
    registry.register(SceneCaptioner())
    registry.register(NSFWDetector())
    
    # Phase 2 Modules
    registry.register(OCRAnalyzer())
    registry.register(FaceAnalyzer())
    registry.register(PoseAnalyzer())
    registry.register(DepthAnalyzer())
    registry.register(SigLIPAnalyzer())
    registry.register(SegmentationAnalyzer())
    
    # Initialize Pipeline Orchestrator
    orchestrator = PipelineOrchestrator(registry, model_manager)
    
    # Attach to app state
    app.state.model_manager = model_manager
    app.state.orchestrator = orchestrator
    
    yield
    
    # Cleanup on shutdown
    await model_manager.unload_all()


# Initialize FastAPI app
app = FastAPI(
    title="VisionResearch API",
    description="Maximum image intelligence extraction",
    version="0.1.0",
    lifespan=lifespan,
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api")

# Mount static files
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")
app.mount("/outputs", StaticFiles(directory=settings.output_dir), name="outputs")

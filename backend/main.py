"""FastAPI app initialization and middleware setup."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.config import settings
from backend.core.model_manager import ModelManager
from backend.core.pipeline import ModuleRegistry, PipelineOrchestrator
from backend.modules.metadata import MetadataAnalyzer
from backend.modules.colors import ColorPaletteAnalyzer
from backend.modules.objects import ObjectDetector
from backend.modules.caption import SceneCaptioner
from backend.modules.nsfw import NSFWDetector


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

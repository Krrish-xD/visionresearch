# VisionResearch — Maximum Image Intelligence Extraction

> **Mission**: Take any image and extract as much structured information as possible from it.

## Table of Contents
- [Project Vision](#project-vision)
- [User Review Required](#user-review-required)
- [Open Questions](#open-questions)
- [System Architecture](#system-architecture)
- [Analysis Modules](#analysis-modules)
- [Model Selection Matrix](#model-selection-matrix)
- [GPU Memory Strategy](#gpu-memory-strategy)
- [Output Schema](#output-schema)
- [Web UI Design](#web-ui-design)
- [Tech Stack](#tech-stack)
- [Phased Roadmap](#phased-roadmap)
- [Verification Plan](#verification-plan)

---

## Project Vision

VisionResearch is a **local-first, modular image analysis system** that orchestrates multiple specialized vision models (and VLMs) to produce a comprehensive, structured report about any input image. It runs on consumer hardware (starting at 12GB VRAM) with a beautiful web UI for interactive exploration of results.

**Core Principles**:
1. **Maximum information** — Extract every meaningful signal from an image
2. **Modular architecture** — Each analysis capability is a self-contained module that can be added, removed, or swapped independently
3. **Local-first** — Everything runs on your machine, no cloud dependencies
4. **Progressive results** — Stream analysis results to the UI as each module completes (don't wait for everything)
5. **Structured output** — All results are typed, validated, and JSON-serializable via Pydantic
6. **Hardware-aware** — Intelligently manage GPU memory, loading/unloading models as needed

---

## User Review Required

> [!IMPORTANT]
> ### Architecture Decision: Hybrid Pipeline vs. Single VLM
> We propose a **Hybrid Architecture** — using specialized models for each task (YOLO for detection, PaddleOCR for text, etc.) orchestrated by a central pipeline, with a VLM (Qwen2.5-VL or Florence-2) for scene understanding and captioning.
>
> **Alternative**: Use a single large VLM (like Qwen3-VL-8B) for *everything* — object detection, OCR, captioning, etc. via structured prompting.
>
> **Our recommendation**: Hybrid. Specialists are faster, more accurate on their tasks, and give us finer control. The VLM handles the "glue" — holistic understanding, captioning, and anything the specialists miss.

> [!WARNING]
> ### 12GB VRAM Constraint
> Running all 12 analysis modules simultaneously is **not feasible** on 12GB VRAM. We propose a **model swapping strategy** — only 1-3 models loaded at a time, with intelligent load/unload based on the analysis pipeline stage. This means sequential processing with some parallelism for lightweight models. Total analysis time for a single image will be **5-15 seconds** on 12GB VRAM. When you upgrade to better hardware, we can switch to parallel execution for sub-2-second total latency.

> [!IMPORTANT]
> ### Project Name & Branding
> Current working name: **VisionResearch**. The web UI, CLI, and package will all use this name. Let us know if you'd like something different.

---

## Open Questions

> [!IMPORTANT]
> ### Q1: Python Package Manager
> Which Python package manager do you prefer?
> - **uv** (fast, modern, recommended) 
> - **pip + venv** (traditional)
> - **conda** (for CUDA management)
> - **poetry**

> [!IMPORTANT]
> ### Q2: VLM Choice for Scene Understanding
> For the core "understand this image" capability, we need to pick a VLM:
> - **Qwen2.5-VL-7B** — Best overall open-source VLM, excellent OCR, 200+ languages, ~14GB VRAM in FP16 but runs well in 4-bit quantization (~5GB)
> - **Florence-2-large** — Microsoft's efficient multi-task model, only 0.7B params (~2GB VRAM), fast but less capable for complex reasoning
> - **Gemma 3/4 (4B or 12B)** — Google's efficient models, good agentic support
> - **Phi-4 Multimodal (5.6B)** — Microsoft's lightweight option
> 
> **Recommendation**: Start with **Florence-2** for speed on 12GB VRAM, add **Qwen2.5-VL-7B (4-bit)** as the "deep analysis" option. This gives us a fast path and a thorough path.

> [!NOTE]
> ### Q3: API Support
> Should we expose a REST API (FastAPI) from the start, or focus purely on the web UI first and add API later?
> The web UI will communicate with the backend via an internal API regardless — the question is whether to also expose it externally for programmatic access.

> [!NOTE]
> ### Q4: Batch Processing
> Do you anticipate needing to process folders/batches of images? If so, we should design the queue/worker system early. This affects architecture decisions.

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Web UI (Vite + React)                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  Upload   │  │  Live Result │  │  Interactive Canvas   │  │
│  │  Zone     │  │  Stream      │  │  (bbox, masks, depth) │  │
│  └──────────┘  └──────────────┘  └───────────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │ WebSocket (SSE fallback)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Backend (FastAPI + Python)                  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                Pipeline Orchestrator                  │   │
│  │  • Manages analysis module execution order            │   │
│  │  • Handles GPU memory (load/unload models)            │   │
│  │  • Streams partial results via WebSocket              │   │
│  │  • Aggregates final structured output                 │   │
│  └──────┬───────────────────────────────────────────────┘   │
│         │                                                    │
│  ┌──────▼──────────────────────────────────────────────┐    │
│  │              Analysis Module Registry                │    │
│  │                                                      │    │
│  │  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐  │    │
│  │  │ Object  │ │  OCR    │ │ Caption  │ │  Depth  │  │    │
│  │  │ Detect  │ │ Extract │ │ Generate │ │ Estimate│  │    │
│  │  └─────────┘ └─────────┘ └──────────┘ └─────────┘  │    │
│  │  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐  │    │
│  │  │  Face   │ │ Segment │ │ Classify │ │  Pose   │  │    │
│  │  │ Analyze │ │  (SAM)  │ │ & Tags   │ │ Estimate│  │    │
│  │  └─────────┘ └─────────┘ └──────────┘ └─────────┘  │    │
│  │  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐  │    │
│  │  │  NSFW   │ │ Embed   │ │  Color   │ │  EXIF   │  │    │
│  │  │ Detect  │ │ (CLIP)  │ │ Palette  │ │ Extract │  │    │
│  │  └─────────┘ └─────────┘ └──────────┘ └─────────┘  │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │               Model Manager (GPU Memory)              │   │
│  │  • Loads/unloads models based on pipeline stage       │   │
│  │  • Tracks VRAM usage                                  │   │
│  │  • Supports quantization (FP16, INT8, 4-bit)          │   │
│  │  • Caches frequently-used models                      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Module System**: Each analysis capability is a Python class implementing a `BaseAnalyzer` interface:
   ```python
   class BaseAnalyzer(ABC):
       name: str                    # e.g., "object_detection"
       display_name: str            # e.g., "Object Detection"
       estimated_vram_mb: int       # VRAM needed when loaded
       
       @abstractmethod
       async def load_model(self) -> None: ...
       
       @abstractmethod
       async def analyze(self, image: PIL.Image) -> AnalysisResult: ...
       
       @abstractmethod
       async def unload_model(self) -> None: ...
   ```

2. **Pipeline Orchestrator**: Determines execution order based on:
   - Module dependencies (e.g., face analysis after face detection)
   - Available VRAM (group small models together, run large models alone)
   - User configuration (which modules are enabled)

3. **Streaming Results**: Each module's results are sent to the UI via WebSocket as soon as they complete. The UI progressively renders results — you see object detection boxes appear before captioning finishes.

4. **Model Manager**: Central registry tracking which models are loaded, their VRAM usage, and handling load/unload with proper CUDA memory cleanup (`torch.cuda.empty_cache()`).

---

## Analysis Modules

### Module Inventory (12 Modules)

| # | Module | Model | VRAM (Loaded) | Latency | Priority |
|---|--------|-------|---------------|---------|----------|
| 1 | **EXIF/Metadata** | None (CPU) | 0 MB | <10ms | P0 — Instant |
| 2 | **Color Palette** | None (CPU) | 0 MB | <50ms | P0 — Instant |
| 3 | **Object Detection** | YOLOv8-m / YOLO26 | ~200 MB | 10-30ms | P0 — Core |
| 4 | **OCR / Text** | PaddleOCR v4 | ~300 MB | 50-150ms | P0 — Core |
| 5 | **Scene Caption** | Florence-2-large | ~2 GB | 200-500ms | P0 — Core |
| 6 | **NSFW Detection** | nsfw-detection-384 | ~200 MB | 20-50ms | P0 — Core |
| 7 | **Image Embedding** | SigLIP 2 / CLIP | ~500 MB | 30-80ms | P1 — Important |
| 8 | **Face Analysis** | DeepFace (RetinaFace + ArcFace) | ~500 MB | 50-200ms | P1 — Important |
| 9 | **Classification & Tags** | SigLIP 2 (zero-shot) | Shared w/ #7 | 30-80ms | P1 — Important |
| 10 | **Depth Estimation** | Depth Anything V2 (small/base) | ~500 MB | 50-150ms | P1 — Important |
| 11 | **Pose Estimation** | YOLO26-Pose / RTMPose | ~300 MB | 20-50ms | P2 — Extended |
| 12 | **Semantic Segmentation** | SAM 2 (small) | ~1.5 GB | 100-300ms | P2 — Extended |

**Total VRAM if all loaded**: ~6 GB (with quantization and model sharing)  
**Peak VRAM during execution**: ~3-4 GB (with model swapping)

### Execution Order (12GB VRAM Strategy)

```
Stage 0: CPU-only (parallel)
├── EXIF/Metadata extraction
└── Color palette analysis

Stage 1: Lightweight GPU models (loaded together, ~1 GB total)
├── Object Detection (YOLO)
├── NSFW Detection
└── Pose Estimation (if humans detected by YOLO)

Stage 2: OCR + Embedding (~800 MB)
├── OCR (PaddleOCR)
└── Image Embedding (SigLIP 2)
    └── Classification & Tags (reuses SigLIP, no extra load)

Stage 3: VLM Analysis (~2 GB)
└── Scene Caption + Deep Understanding (Florence-2)
    └── Can also do: detailed captioning, VQA, region descriptions

Stage 4: Heavy models (one at a time, ~1.5 GB each)
├── Depth Estimation (Depth Anything V2)
├── Face Analysis (DeepFace)  — only if faces detected in Stage 1
└── Semantic Segmentation (SAM 2) — optional, user-triggered
```

---

## Model Selection Matrix

### Primary Models (Phase 1)

| Task | Model | Why This One | VRAM | Quantization |
|------|-------|-------------|------|--------------|
| Object Detection | **Ultralytics YOLOv8-m** | Best speed/accuracy, huge ecosystem, battle-tested | ~200MB | FP16 |
| OCR | **PaddleOCR v4** | 80+ languages, fastest production OCR, handles complex layouts | ~300MB | FP32 (CPU) or FP16 |
| Scene Understanding | **Florence-2-large (0.7B)** | Multi-task (caption, detection, OCR, grounding), tiny VRAM footprint | ~2GB | FP16 |
| NSFW | **Marqo/nsfw-image-detection-384** | Lightweight ViT, ~98.5% accuracy | ~200MB | FP16 |
| Embeddings | **SigLIP 2 (ViT-B)** | SOTA vision-language alignment, multilingual | ~500MB | FP16 |
| Depth | **Depth Anything V2 (Small)** | SOTA monocular depth, handles edge cases well | ~500MB | FP16 |
| Faces | **DeepFace** | Wraps RetinaFace (detection) + ArcFace (recognition), easy API | ~500MB | FP32 |
| Segmentation | **SAM 2 (tiny/small)** | Class-agnostic, works with any prompt, Meta-backed | ~1.5GB | FP16 |
| Pose | **YOLOv8-pose / YOLO26-Pose** | Single-pass detection + pose, shares YOLO ecosystem | ~200MB | FP16 |
| Classification | **SigLIP 2** (reused) | Zero-shot via text prompts, shares embedding model | Shared | — |

### Upgrade Path (Phase 2+ / Better Hardware)

| Task | Upgrade To | Why | VRAM |
|------|-----------|-----|------|
| Scene Understanding | **Qwen2.5-VL-7B (4-bit)** | Far more capable reasoning, better OCR, agentic | ~5GB |
| Object Detection | **YOLO26 / Grounding DINO 2** | Open-vocabulary detection | ~1-3GB |
| OCR | **Surya v2 / GOT-OCR 2.0** | Better complex document parsing | ~2GB |
| Segmentation | **SAM 2 (large)** + semantic labels | Higher quality masks | ~3GB |
| Depth | **Depth Anything V2 (Large)** | More detail | ~1.5GB |

---

## GPU Memory Strategy

### For 12GB VRAM

```
Available: ~11 GB usable (OS/display takes ~1 GB)

Strategy: Sequential Stage Execution with Model Swapping

┌──────────────────────────────────────────────┐
│ Stage 0: CPU Only                     0 MB   │
│ Stage 1: YOLO + NSFW + Pose        ~600 MB   │
│ → unload all                                 │
│ Stage 2: PaddleOCR + SigLIP        ~800 MB   │
│ → unload all                                 │
│ Stage 3: Florence-2               ~2000 MB   │
│ → unload                                     │
│ Stage 4a: Depth Anything           ~500 MB   │
│ → unload                                     │
│ Stage 4b: DeepFace                 ~500 MB   │
│ → unload                                     │
│ Stage 4c: SAM 2 (optional)        ~1500 MB   │
│ → unload                                     │
│                                              │
│ Peak VRAM usage: ~2 GB (Stage 3)             │
│ Estimated total time: 3-8 seconds            │
└──────────────────────────────────────────────┘
```

### For 24GB+ VRAM (Future)

```
Strategy: Parallel Fan-Out — load all models, run concurrently

All models loaded simultaneously: ~6 GB
Execution: asyncio.gather() for independent modules
Estimated total time: 0.5-2 seconds
```

### Memory Management Implementation

```python
class ModelManager:
    """Central GPU memory manager."""
    
    def __init__(self, vram_budget_mb: int = 11_000):
        self.vram_budget = vram_budget_mb
        self.loaded_models: dict[str, LoadedModel] = {}
        self.vram_used: int = 0
    
    async def ensure_loaded(self, module: BaseAnalyzer) -> None:
        """Load a model, evicting others if necessary."""
        if module.name in self.loaded_models:
            return
        
        # Evict models if needed to fit
        while self.vram_used + module.estimated_vram_mb > self.vram_budget:
            await self._evict_least_recently_used()
        
        await module.load_model()
        self.loaded_models[module.name] = LoadedModel(...)
        self.vram_used += module.estimated_vram_mb
    
    async def _evict_least_recently_used(self) -> None:
        """Unload the least recently used model."""
        lru = min(self.loaded_models.values(), key=lambda m: m.last_used)
        await lru.module.unload_model()
        torch.cuda.empty_cache()
        self.vram_used -= lru.module.estimated_vram_mb
        del self.loaded_models[lru.name]
```

---

## Output Schema

### Pydantic Models (Core)

```python
from pydantic import BaseModel
from enum import Enum

class BoundingBox(BaseModel):
    x_min: float    # normalized 0-1
    y_min: float
    x_max: float
    y_max: float

class DetectedObject(BaseModel):
    label: str
    confidence: float
    bbox: BoundingBox
    area_fraction: float  # % of image area

class TextRegion(BaseModel):
    content: str
    confidence: float
    bbox: BoundingBox
    language: str | None = None

class FaceAnalysis(BaseModel):
    bbox: BoundingBox
    age: int | None = None
    gender: str | None = None
    emotion: str | None = None
    emotion_confidence: float | None = None

class Keypoint(BaseModel):
    name: str
    x: float
    y: float
    confidence: float

class PoseEstimation(BaseModel):
    person_bbox: BoundingBox
    keypoints: list[Keypoint]
    confidence: float

class ColorInfo(BaseModel):
    hex: str
    rgb: tuple[int, int, int]
    percentage: float
    name: str  # nearest named color

class NSFWResult(BaseModel):
    is_nsfw: bool
    category: str  # "safe", "suggestive", "explicit"
    confidence: float

class ImageMetadata(BaseModel):
    width: int
    height: int
    format: str
    file_size_bytes: int
    exif: dict | None = None
    camera: str | None = None
    date_taken: str | None = None
    gps: dict | None = None

# === Top-Level Result ===

class AnalysisResult(BaseModel):
    """Complete analysis output for a single image."""
    
    # Meta
    image_id: str
    filename: str
    timestamp: str
    total_processing_time_ms: float
    modules_executed: list[str]
    schema_version: str = "1.0.0"
    
    # Results
    metadata: ImageMetadata
    caption: str | None = None
    detailed_description: str | None = None
    tags: list[str] = []
    objects: list[DetectedObject] = []
    text_regions: list[TextRegion] = []
    faces: list[FaceAnalysis] = []
    poses: list[PoseEstimation] = []
    colors: list[ColorInfo] = []
    nsfw: NSFWResult | None = None
    depth_map_path: str | None = None       # path to saved depth map image
    segmentation_map_path: str | None = None # path to saved segmentation image
    embedding: list[float] | None = None     # image embedding vector
    
    # Per-module timing
    module_timings: dict[str, float] = {}  # module_name -> ms
```

---

## Web UI Design

### Technology Choice: **Vite + React + TypeScript**

**Why React**:
- Richest ecosystem for interactive image visualization
- Excellent canvas libraries (react-konva, fabricjs bindings)
- Component model fits our modular results display perfectly
- TypeScript gives us type safety matching our Pydantic backend schemas

### UI Layout

```
┌────────────────────────────────────────────────────────────────┐
│  VisionResearch                              [Settings] [Dark] │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────┐  ┌────────────────────────────────┐  │
│  │                      │  │  Analysis Results               │  │
│  │                      │  │                                  │  │
│  │    Image Canvas      │  │  ┌─ Caption ──────────────────┐ │  │
│  │    (with overlays)   │  │  │ "A golden retriever..."    │ │  │
│  │                      │  │  └────────────────────────────┘ │  │
│  │   [bbox] [mask]      │  │                                  │  │
│  │   [depth] [pose]     │  │  ┌─ Objects (5 found) ────────┐ │  │
│  │                      │  │  │ 🐕 dog       98.2%         │ │  │
│  │  ──── overlays ──── │  │  │ 🏠 house     94.1%         │ │  │
│  │  □ Bounding Boxes    │  │  │ 🌳 tree      91.3%         │ │  │
│  │  □ Segmentation      │  │  └────────────────────────────┘ │  │
│  │  □ Depth Map         │  │                                  │  │
│  │  □ Pose Skeleton     │  │  ┌─ Text/OCR ─────────────────┐ │  │
│  │  □ Face Boxes        │  │  │ "STOP" (confidence: 99.1%) │ │  │
│  │                      │  │  └────────────────────────────┘ │  │
│  └──────────────────────┘  │                                  │  │
│                            │  ┌─ Colors ───────────────────┐  │  │
│  ┌──────────────────────┐  │  │ ██ #4A7C59 (32%)           │  │  │
│  │ Upload / Drag & Drop │  │  │ ██ #8B4513 (24%)           │  │  │
│  │ 📁 or paste from     │  │  │ ██ #87CEEB (18%)           │  │  │
│  │    clipboard          │  │  └────────────────────────────┘ │  │
│  └──────────────────────┘  │                                  │  │
│                            │  [📋 Copy JSON] [💾 Export]      │  │
│  ── Processing ──────────  └────────────────────────────────┘  │
│  ████████░░░░  Stage 2/4                                       │
│  ✅ EXIF  ✅ YOLO  🔄 OCR  ⏳ Caption  ⏳ Depth               │
└────────────────────────────────────────────────────────────────┘
```

### Key UI Features

1. **Interactive Canvas**: HTML5 Canvas with overlay toggle — switch between bounding boxes, segmentation masks, depth heatmap, pose skeletons
2. **Progressive Loading**: Results stream in module-by-module with animations. A progress bar shows which modules have completed.
3. **Hover Interaction**: Hover over a detected object in the results panel → highlights its bounding box on the canvas (and vice versa)
4. **Dark Mode**: Default dark theme with optional light mode
5. **Drag & Drop + Clipboard**: Upload via drag-and-drop, file picker, or Ctrl+V paste
6. **Export**: Copy full JSON to clipboard, download as JSON, or save annotated image
7. **Settings Panel**: Toggle individual modules on/off, adjust confidence thresholds, select model variants

### Design Aesthetic
- **Glassmorphism** panels with subtle blur backgrounds
- **Smooth animations** for results appearing (slide-in, fade-in)
- **Color-coded** module results (each module gets a signature color)
- **Modern typography** (Inter or Geist font)
- **Responsive** — works on tablet-sized screens too

---

## Tech Stack

### Backend
| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11+ |
| Web Framework | FastAPI | Latest |
| Real-time | WebSockets (via FastAPI) | — |
| Validation | Pydantic v2 | Latest |
| ML Framework | PyTorch | 2.x |
| Image Processing | Pillow, OpenCV | — |
| EXIF | Pillow / exifread | — |
| Task Queue | (Phase 2) Celery + Redis | — |

### Frontend
| Component | Technology | Version |
|-----------|-----------|---------|
| Build Tool | Vite | 6.x |
| Framework | React | 19.x |
| Language | TypeScript | 5.x |
| Styling | Vanilla CSS (CSS Modules) | — |
| Canvas | HTML5 Canvas API / react-konva | — |
| Icons | Lucide React | — |
| Fonts | Inter (Google Fonts) | — |

### DevOps
| Component | Technology |
|-----------|-----------|
| Package Manager | uv (Python) / npm (JS) |
| Containerization | Docker + docker-compose (Phase 2) |
| Code Quality | ruff, mypy, eslint, prettier |
| Testing | pytest (backend), vitest (frontend) |

---

## Phased Roadmap

### Phase 1: Foundation (Weeks 1-2)
> Get the core pipeline working end-to-end with 4-5 modules and a basic web UI.

**Backend**:
- [ ] Project scaffolding (uv, FastAPI, folder structure)
- [ ] `BaseAnalyzer` abstract class and module registry
- [ ] `ModelManager` with VRAM tracking and model swapping
- [ ] `PipelineOrchestrator` with sequential stage execution
- [ ] WebSocket streaming of partial results
- [ ] Implement core modules:
  - [ ] EXIF/Metadata (CPU-only)
  - [ ] Color Palette (CPU-only)
  - [ ] Object Detection (YOLOv8)
  - [ ] Scene Captioning (Florence-2)
  - [ ] NSFW Detection

**Frontend**:
- [ ] Vite + React + TypeScript scaffolding
- [ ] Drag-and-drop image upload
- [ ] WebSocket connection to backend
- [ ] Progressive results display
- [ ] Basic image canvas with bounding box overlay
- [ ] Dark theme, responsive layout

### Phase 2: Full Module Suite (Weeks 3-4)
> Add all remaining analysis modules and polish the UI.

- [ ] OCR module (PaddleOCR)
- [ ] Image embedding (SigLIP 2)
- [ ] Zero-shot classification & tags
- [ ] Depth estimation (Depth Anything V2)
- [ ] Face analysis (DeepFace)
- [ ] Pose estimation (YOLO-Pose)
- [ ] Semantic segmentation (SAM 2)
- [ ] UI: Overlay toggles (depth map, segmentation, pose)
- [ ] UI: Settings panel with module enable/disable
- [ ] UI: JSON export and copy

### Phase 3: Intelligence & Polish (Weeks 5-6)
> Add VLM deep analysis, optimize performance, and make it production-quality.

- [ ] Upgrade VLM to Qwen2.5-VL (optional, hardware-dependent)
- [ ] VLM-powered "Ask about this image" feature (free-form VQA)
- [ ] Cross-module intelligence (e.g., "person holding a phone" from combining object detection + pose)
- [ ] Batch processing (folder of images)
- [ ] Analysis history (SQLite)
- [ ] Performance optimization (model caching, pre-loading)
- [ ] Docker containerization

### Phase 4: Advanced Features (Ongoing)
- [ ] Image comparison (diff two images)
- [ ] Video frame analysis
- [ ] REST API with OpenAPI docs
- [ ] Plugin system for community modules
- [ ] Model fine-tuning interface
- [ ] Cloud deployment option

---

## Project Structure

```
visionresearch/
├── GEMINI.md                    # Project context for AI assistants
├── README.md
├── pyproject.toml               # Python project config (uv)
├── docker-compose.yml           # (Phase 2)
│
├── backend/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Settings, env vars, model paths
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseAnalyzer ABC
│   │   ├── pipeline.py          # PipelineOrchestrator
│   │   ├── model_manager.py     # GPU memory manager
│   │   ├── schemas.py           # Pydantic output models
│   │   └── events.py            # WebSocket event types
│   │
│   ├── modules/                 # Analysis modules (one file each)
│   │   ├── __init__.py
│   │   ├── metadata.py          # EXIF/metadata extraction
│   │   ├── colors.py            # Color palette analysis
│   │   ├── objects.py           # Object detection (YOLO)
│   │   ├── ocr.py               # Text extraction (PaddleOCR)
│   │   ├── caption.py           # Scene captioning (Florence-2 / VLM)
│   │   ├── nsfw.py              # NSFW detection
│   │   ├── embedding.py         # Image embedding (SigLIP)
│   │   ├── classification.py    # Zero-shot classification
│   │   ├── depth.py             # Depth estimation
│   │   ├── faces.py             # Face analysis
│   │   ├── pose.py              # Pose estimation
│   │   └── segmentation.py      # Semantic segmentation (SAM 2)
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py            # REST endpoints
│   │   └── websocket.py         # WebSocket handlers
│   │
│   └── utils/
│       ├── __init__.py
│       ├── image.py             # Image preprocessing utilities
│       └── gpu.py               # GPU monitoring utilities
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   │
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── index.css            # Design system, CSS variables
│   │   │
│   │   ├── components/
│   │   │   ├── ImageUpload.tsx   # Drag & drop upload zone
│   │   │   ├── ImageCanvas.tsx   # Canvas with overlays
│   │   │   ├── ResultsPanel.tsx  # Scrollable results sidebar
│   │   │   ├── ProgressBar.tsx   # Module execution progress
│   │   │   ├── SettingsPanel.tsx # Module toggles, thresholds
│   │   │   ├── ModuleCard.tsx    # Individual module result card
│   │   │   └── OverlayControls.tsx # Toggle overlays on canvas
│   │   │
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts  # WebSocket connection hook
│   │   │   └── useAnalysis.ts   # Analysis state management
│   │   │
│   │   ├── types/
│   │   │   └── analysis.ts      # TypeScript types matching Pydantic
│   │   │
│   │   └── utils/
│   │       ├── canvas.ts        # Canvas drawing utilities
│   │       └── colors.ts        # Color utilities
│   │
│   └── public/
│       └── favicon.svg
│
└── tests/
    ├── backend/
    │   ├── test_pipeline.py
    │   ├── test_model_manager.py
    │   └── test_modules/
    └── frontend/
        └── ...
```

---

## Verification Plan

### Automated Tests
```bash
# Backend unit tests
pytest tests/backend/ -v

# Frontend tests  
cd frontend && npm run test

# Type checking
mypy backend/
cd frontend && npx tsc --noEmit

# Lint
ruff check backend/
cd frontend && npm run lint
```

### Manual Verification
1. **End-to-end test**: Upload a complex image (with people, text, objects) and verify all 12 modules produce reasonable output
2. **VRAM monitoring**: Run `nvidia-smi` during analysis to verify VRAM stays under budget
3. **Streaming test**: Verify results stream progressively (not all at once)
4. **Edge cases**: Test with:
   - Very large images (4K+)
   - Very small images (<100px)
   - Images with no objects/text/faces
   - Corrupted/truncated files
   - Non-image files
5. **Performance benchmark**: Time full pipeline on 10 diverse images, record per-module latency

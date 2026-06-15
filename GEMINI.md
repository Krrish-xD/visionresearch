# VisionResearch

## Overview
VisionResearch is a **local-first, modular image analysis system** that orchestrates multiple specialized vision models to extract maximum structured information from any input image. It runs on consumer hardware (12GB+ VRAM) with a web UI for interactive exploration of results.

## Architecture
- **Hybrid Pipeline**: Specialized models (YOLO, PaddleOCR, etc.) for specific tasks, orchestrated by a central `PipelineOrchestrator`, with Florence-2 VLM for scene understanding
- **Model Swapping**: `ModelManager` handles GPU memory — loads/unloads models across 4 execution stages to stay within VRAM budget
- **SSE Streaming**: Results stream progressively to the frontend via Server-Sent Events as each module completes
- **Modular Design**: Each analysis capability is a self-contained module implementing `BaseAnalyzer`

## Tech Stack
### Backend
- **Python 3.11+** managed with `uv`
- **FastAPI** for REST API + SSE streaming
- **Pydantic v2** for all data schemas
- **PyTorch** for ML model inference
- **sse-starlette** for Server-Sent Events

### Frontend
- **Vite + React 19 + TypeScript**
- **Konva.js (react-konva)** for layered canvas rendering
- **Vanilla CSS** with CSS Modules (no Tailwind)
- **Inter** font from Google Fonts

## Project Structure
```
visionresearch/
├── backend/                  # Python backend (FastAPI)
│   ├── core/                 # Framework: BaseAnalyzer, ModelManager, Pipeline, Schemas
│   ├── modules/              # Analysis modules (one per file)
│   ├── api/                  # Routes and SSE handlers
│   └── utils/                # Image processing, GPU monitoring
├── frontend/                 # React frontend (Vite)
│   └── src/
│       ├── components/       # React components
│       ├── hooks/            # Custom hooks (useSSE, useAnalysis)
│       ├── types/            # TypeScript types
│       └── utils/            # Canvas drawing, color utilities
└── tests/                    # pytest (backend) + vitest (frontend)
```

## Key Conventions
- All analysis modules extend `BaseAnalyzer` (see `backend/core/base.py`)
- Output schemas are Pydantic models in `backend/core/schemas.py`
- Bounding boxes use normalized coordinates (0-1): `[x_min, y_min, x_max, y_max]`
- Each module reports `estimated_vram_mb` for the ModelManager
- SSE events follow the format in `backend/core/events.py`
- Frontend TypeScript types mirror the backend Pydantic schemas

## Analysis Modules (Phase 1)
1. **EXIF/Metadata** — CPU-only, extracts camera info, GPS, timestamps
2. **Color Palette** — CPU-only, dominant colors via k-means clustering
3. **Object Detection** — YOLOv8-m, bounding boxes + labels
4. **Scene Captioning** — Florence-2-large, natural language descriptions
5. **NSFW Detection** — ViT-based classifier

## GPU Memory Strategy
- 4 execution stages, models loaded/unloaded between stages
- Peak VRAM: ~2GB (Florence-2 stage)
- `torch.cuda.empty_cache()` between stages
- LRU eviction when VRAM budget exceeded

## Running the Application
To start both the frontend and backend in quiet mode (suppressing debug logs and terminal clutter) concurrently:
```bash
./start.sh
```
Press `Ctrl+C` in the terminal to cleanly shut down both processes.

# Project: VisionResearch VLM Integration & Verification

## Architecture
- **Backend API**: FastAPI application (`backend/main.py`) serving model metadata and lifecycle status endpoints (`/api/models`, `/api/model/status`).
- **Core Engine**: `src/vlm/engine.py` encapsulating the `VLMEngine` singleton and runtime model state.
- **Offline Evaluation**: `src/vlm/evaluate.py` providing CLI offline batch inference & dataset evaluation with `--mock` simulation mode and output generation under `results/raw_predictions/`.
- **Architectural Boundary**: Decoupled engine core, standalone server layer, and modular evaluation runner.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Exploration & Pre-flight Inspection | Examine codebase structure, imports, engine singleton, server routes, evaluation CLI, and python venv environment | none | DONE |
| 2 | FastAPI Backend Server Integration | Start uvicorn server in background, query `/api/models` and `/api/model/status`, verify HTTP 200 and state consistency | M1 | DONE |
| 3 | Offline Evaluation Pipeline Testing | Run `venv/bin/python -m src.vlm.evaluate --mock --limit 2`, verify exit code 0 and generated JSONL output in `results/raw_predictions/` | M1 | DONE |
| 4 | Multi-Agent Review, Challenge & Audit | Multi-agent review of logs, challenger stress testing, and forensic audit of code integrity | M2, M3 | DONE |
| 5 | Final Synthesis & Victory Reporting | Aggregate all artifacts, update GEMINI.md, and report victory to Sentinel | M4 | DONE |

## Interface Contracts
### `backend/main.py` ↔ `src/vlm/engine.py`
- Endpoints query `vlm_engine` state.
- `GET /api/models`: Returns list of available models and configuration metadata.
- `GET /api/model/status`: Returns current loaded model status, device, and memory/readiness status.

### `src/vlm/evaluate.py` ↔ `results/raw_predictions/`
- CLI arguments: `--mock` (bool), `--limit` (int, e.g. 2).
- Output: Writes JSONL predictions to `results/raw_predictions/` containing prediction records with prompt, generated text/tokens, and evaluation metadata.

## Code Layout
- `backend/`: FastAPI application, endpoints, router modules, startup scripts.
- `src/vlm/`: VLM model engine (`engine.py`), evaluation script (`evaluate.py`), pipeline utilities.
- `results/`: Output directories for predictions (`raw_predictions/`) and evaluation metrics.
- `venv/`: Project virtual environment with python 3.x and dependencies installed.
- `.agents/`: Coordination and metadata directories for orchestrator and subagents.

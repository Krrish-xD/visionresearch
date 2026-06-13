FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_TORCH=1 \
    USE_TF=0 \
    USE_FLAX=0 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    FLAGS_enable_pir_api=0 \
    FLAGS_use_mkldnn=0

# Create application directory
WORKDIR /app

# Copy backend dependencies and install
COPY backend/pyproject.toml ./backend/
WORKDIR /app/backend
RUN uv pip install --system -r pyproject.toml

# Copy backend source code
COPY backend/ ./

# Copy frontend static build files (assuming they are built externally)
# Or we could build frontend inside docker, but typically it's served via nginx.
# Here we'll just serve via FastAPI static files as we are already doing.
# To do this cleanly, we'll copy the frontend dist into /app/frontend/dist
WORKDIR /app
COPY frontend/dist/ ./frontend/dist/

# Set working directory back to backend for running FastAPI
WORKDIR /app/backend

# We will modify FastAPI to serve the frontend from /app/frontend/dist
# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

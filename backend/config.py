"""Application configuration using pydantic-settings."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Paths
    upload_dir: Path = Path("./uploads")
    output_dir: Path = Path("./outputs")
    model_cache_dir: Path = Path("./models")

    # GPU / Memory
    vram_budget_mb: int = 0  # 0 = auto-detect at startup
    device: str = "auto"  # "auto", "cuda", "cpu"

    # Pipeline
    default_modules: list[str] = [
        "metadata",
        "colors",
        "object_detection",
        "pose",
        "nsfw",
        "ocr",
        "siglip",
        "caption",
        "faces",
        "depth",
        "segmentation",
    ]

    # Analysis defaults
    confidence_threshold: float = 0.25
    max_image_dimension: int = 1920  # Resize images larger than this (limits activation memory)

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = {"env_prefix": "VR_", "env_file": ".env"}

    def resolve_device(self) -> str:
        """Determine the best available device."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def resolve_vram_budget(self) -> int:
        """Auto-detect usable VRAM, leaving 10% headroom for CUDA overhead."""
        if self.vram_budget_mb > 0:
            return self.vram_budget_mb
        try:
            import torch
            if torch.cuda.is_available():
                total_bytes = torch.cuda.get_device_properties(0).total_memory
                total_mb = total_bytes // (1024 * 1024)
                # Leave 10% headroom for CUDA context and driver
                usable = int(total_mb * 0.88)
                return usable
        except Exception:
            pass
        return 4000  # Conservative fallback for CPU

    def ensure_dirs(self) -> None:
        """Create required directories."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()

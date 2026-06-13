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
    vram_budget_mb: int = 11_000  # 11GB usable on a 12GB card
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
    max_image_dimension: int = 4096  # Resize images larger than this

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

    def ensure_dirs(self) -> None:
        """Create required directories."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()

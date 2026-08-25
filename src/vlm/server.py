"""FastAPI backend singleton state for VLM Engine."""

from src.vlm.engine import VLMEngine

# Global singleton instance for the backend to use
vlm_engine = VLMEngine()

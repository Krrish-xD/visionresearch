import logging
from PIL import Image
from core.base import BaseAnalyzer
from core.schemas import ModuleResult

logger = logging.getLogger(__name__)

class BrightnessAnalyzer(BaseAnalyzer):
    """A sample plugin that calculates the average brightness of an image."""

    @property
    def name(self) -> str:
        return "brightness"

    @property
    def display_name(self) -> str:
        return "Image Brightness"

    @property
    def stage(self) -> int:
        return 0  # CPU stage

    @property
    def requires_gpu(self) -> bool:
        return False

    async def load_model(self, device: str = "cpu") -> None:
        self._is_loaded = True

    async def unload_model(self) -> None:
        self._is_loaded = False

    async def analyze(self, image: Image.Image, **kwargs) -> dict:
        # Convert image to grayscale
        grayscale_image = image.convert("L")
        
        # Calculate average brightness (0-255)
        pixels = list(grayscale_image.getdata())
        avg_brightness = sum(pixels) / len(pixels)
        
        # Normalize to 0-1 range
        normalized_brightness = avg_brightness / 255.0
        
        # Determine human-readable label
        if normalized_brightness < 0.3:
            label = "Dark"
        elif normalized_brightness > 0.7:
            label = "Bright"
        else:
            label = "Normal"

        return {
            "metadata": {
                "average_brightness": round(normalized_brightness, 3),
                "lighting_condition": label
            }
        }

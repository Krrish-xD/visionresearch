"""Face Analysis Module using DeepFace."""

import time
import logging
from typing import Any
import numpy as np
import PIL.Image

from core.base import BaseAnalyzer
from core.schemas import AnalysisResult, FaceAnalysis, BoundingBox


logger = logging.getLogger(__name__)


class FaceAnalyzer(BaseAnalyzer):
    """Analyzes faces for age, gender, and emotion."""

    name: str = "faces"
    display_name: str = "Face Analysis"
    estimated_vram_mb: int = 500
    requires_gpu: bool = True
    stage: int = 4

    def __init__(self, backend: str = "retinaface"):
        # retinaface is very accurate, opencv is fast but less accurate
        self.backend = backend
        self.models_loaded = False

    async def load_model(self, device: str = "cpu") -> None:
        if not self.models_loaded:
            # DeepFace loads models automatically on the first call.
            # We can force a load by passing a dummy image.
            try:
                from deepface import DeepFace
                dummy = np.zeros((224, 224, 3), dtype=np.uint8)
                DeepFace.analyze(
                    dummy, 
                    actions=["age", "gender", "emotion"], 
                    enforce_detection=False,
                    detector_backend="opencv", # use fast backend just for warming up weights
                    silent=True
                )
            except Exception as e:
                logger.warning(f"Failed to pre-load DeepFace models: {e}")
            self.models_loaded = True

    async def unload_model(self) -> None:
        # DeepFace caches models globally in memory. It's difficult to unload them 
        # specifically. In a real highly-constrained environment, we'd manage the 
        # tf.keras sessions directly, but for now we just let them stay.
        pass

    async def analyze(self, image: PIL.Image.Image, **kwargs: Any) -> dict:
        start_time = time.time()
        
        from deepface import DeepFace
        
        # Convert PIL Image (RGB) to OpenCV format (BGR)
        img_np = np.array(image.convert("RGB"))
        img_bgr = img_np[:, :, ::-1]
        
        faces: list[FaceAnalysis] = []
        width, height = image.size
        
        try:
            # enforce_detection=True will raise ValueError if no face found
            results = DeepFace.analyze(
                img_bgr,
                actions=["age", "gender", "emotion"],
                detector_backend=self.backend,
                enforce_detection=True,
                silent=True
            )
            
            # If a single face is found, it might return a dict instead of a list
            if isinstance(results, dict):
                results = [results]
                
            for face in results:
                region = face.get("region", {})
                x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
                
                # If width or height is 0, skip
                if w == 0 or h == 0:
                    continue
                
                bbox = BoundingBox(
                    x_min=max(0.0, float(x) / width),
                    y_min=max(0.0, float(y) / height),
                    x_max=min(1.0, float(x + w) / width),
                    y_max=min(1.0, float(y + h) / height),
                )
                
                emotion_dict = face.get("emotion", {})
                dominant_emotion = face.get("dominant_emotion")
                emotion_confidence = float(emotion_dict.get(dominant_emotion, 0)) / 100.0 if dominant_emotion else 0.0
                
                faces.append(
                    FaceAnalysis(
                        bbox=bbox,
                        age=int(face.get("age", 0)),
                        gender=face.get("dominant_gender", "unknown"),
                        emotion=dominant_emotion,
                        emotion_confidence=emotion_confidence
                    )
                )
                
        except ValueError as e:
            # ValueError: Face could not be detected.
            pass
        except Exception as e:
            logger.error(f"DeepFace analysis failed: {e}")

        return {
            "faces": faces
        }

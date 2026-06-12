"""Object detection module using YOLOv8."""

import asyncio
from typing import Any

from PIL import Image
import torch

from backend.core.base import BaseAnalyzer
from backend.core.schemas import BoundingBox, DetectedObject


class ObjectDetector(BaseAnalyzer):
    """Detects objects using Ultralytics YOLOv8.
    
    Runs in Stage 1.
    """

    name = "object_detection"
    display_name = "Object Detection"
    estimated_vram_mb = 250  # YOLOv8-m in FP16 is tiny
    requires_gpu = True
    stage = 1

    def __init__(self, model_id: str = "yolov8m.pt"):
        super().__init__()
        self.model_id = model_id
        self.model = None

    async def load_model(self, device: str = "cpu") -> None:
        from ultralytics import YOLO
        
        # Load model (downloads if not present)
        # Using to_thread because model loading does sync I/O
        self.model = await asyncio.to_thread(YOLO, self.model_id)
        if device.startswith("cuda"):
            self.model.to("cuda")
        
        self._is_loaded = True

    async def analyze(self, image: Image.Image, **kwargs: Any) -> dict:
        if not self.model:
            raise RuntimeError("Model not loaded")
            
        conf_thresh = kwargs.get("confidence_threshold", 0.25)
        
        # Run inference
        results = await asyncio.to_thread(
            self.model, image, conf=conf_thresh, verbose=False
        )
        
        result = results[0]
        detected = []
        
        # Parse Ultralytics Results object
        if result.boxes:
            # normalized xyxy format
            boxes = result.boxes.xyxyn.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            names = result.names
            
            for box, conf, cls_id in zip(boxes, confs, class_ids):
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                
                # Calculate fraction of image area
                area = (x2 - x1) * (y2 - y1)
                
                detected.append(
                    DetectedObject(
                        label=names[cls_id],
                        confidence=float(conf),
                        bbox=BoundingBox(
                            x_min=x1, y_min=y1, x_max=x2, y_max=y2
                        ),
                        area_fraction=float(area)
                    )
                )
                
        return {"objects": detected}

    async def unload_model(self) -> None:
        if self.model:
            del self.model
            self.model = None
        self._is_loaded = False

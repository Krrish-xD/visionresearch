"""YOLOv8 Pose Estimation Module."""

import time
from typing import Any
import PIL.Image

from core.base import BaseAnalyzer
from core.schemas import AnalysisResult, PoseEstimation, Keypoint, BoundingBox


class PoseAnalyzer(BaseAnalyzer):
    """Estimates human poses using YOLOv8-pose."""

    name: str = "pose"
    display_name: str = "Pose Estimation"
    estimated_vram_mb: int = 200
    requires_gpu: bool = True
    stage: int = 1

    def __init__(self, model_id: str = "yolov8m-pose.pt", conf_threshold: float = 0.5):
        self.model_id = model_id
        self.conf_threshold = conf_threshold
        self.model = None

    async def load_model(self, device: str = "cpu") -> None:
        if self.model is None:
            # We import here to avoid slow startup
            from ultralytics import YOLO
            self.model = YOLO(self.model_id)

    async def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

    async def analyze(self, image: PIL.Image.Image, **kwargs: Any) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start_time = time.time()
        
        # YOLOv8 predicts bounding boxes AND keypoints
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        result = results[0]
        
        poses: list[PoseEstimation] = []
        
        width, height = image.size
        
        if result.keypoints is not None and result.keypoints.has_visible:
            # Result contains keypoints.data of shape (N, 17, 3) where 3 is (x, y, conf)
            # Result contains boxes.data of shape (N, 6) where 6 is (x1, y1, x2, y2, conf, cls)
            
            kpts_data = result.keypoints.data.cpu().numpy()
            boxes_data = result.boxes.data.cpu().numpy()
            
            # COCO keypoint names
            keypoint_names = [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ]
            
            for i, (kpts, box) in enumerate(zip(kpts_data, boxes_data)):
                # box: [x1, y1, x2, y2, conf, cls]
                person_bbox = BoundingBox(
                    x_min=float(box[0]) / width,
                    y_min=float(box[1]) / height,
                    x_max=float(box[2]) / width,
                    y_max=float(box[3]) / height,
                )
                person_conf = float(box[4])
                
                keypoints_list: list[Keypoint] = []
                for j, kp in enumerate(kpts):
                    # kp: [x, y, conf]
                    if float(kp[2]) > 0.3:  # Only include visible keypoints
                        keypoints_list.append(
                            Keypoint(
                                name=keypoint_names[j] if j < len(keypoint_names) else f"kp_{j}",
                                x=float(kp[0]) / width,
                                y=float(kp[1]) / height,
                                confidence=float(kp[2])
                            )
                        )
                
                poses.append(
                    PoseEstimation(
                        person_bbox=person_bbox,
                        keypoints=keypoints_list,
                        confidence=person_conf
                    )
                )

        return {
            "poses": poses
        }

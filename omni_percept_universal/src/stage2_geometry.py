import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline
from .memory_utils import clear_memory

def run_geometry_extraction(image_path, grounding_data, output_dir):
    """
    Stage 2: Metric Depth & 3D Feature Extraction
    Calculates median depth, 3D centroids, and object orientation.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("[Stage 2] Loading Depth Model...")
    try:
        # Depth Anything V2 as primary/fallback, available widely in transformers
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
    except Exception as e:
        print(f"[Stage 2] Depth Anything V2 failed ({e}). Attempting older standard DPT model.")
        pipe = pipeline(task="depth-estimation", model="Intel/dpt-large", device=device)

    image = Image.open(image_path).convert("RGB")
    depth_result = pipe(image)
    
    # depth_result["depth"] is a PIL Image
    depth_map = np.array(depth_result["depth"], dtype=np.float32)
    
    # Convert disparity-like map to distance proxy
    # Distance = 1 / (Disparity + eps). Depth maps usually have closer objects as brighter (higher values).
    epsilon = 1e-6
    distance_map = 1.0 / (depth_map + epsilon)
    # Normalize distance (0 to 100 meters purely for Z3 solver stability and relative comparisons)
    distance_map = (distance_map - distance_map.min()) / (distance_map.max() - distance_map.min() + epsilon) * 100.0

    del pipe
    clear_memory()
    
    print("[Stage 2] Extracting geometric features for objects...")
    
    results = []
    
    for obj in grounding_data:
        mask_path = obj.get("mask_path")
        if not mask_path or not os.path.exists(mask_path):
            continue
            
        mask = np.load(mask_path) # Boolean mask [H, W]
        
        # 1. Z-axis: Median depth of pixels inside mask
        mask_distances = distance_map[mask]
        z = float(np.median(mask_distances)) if len(mask_distances) > 0 else 0.0
        
        # 2. X, Y: Centroid via Image Moments
        mask_uint8 = (mask * 255).astype(np.uint8)
        M = cv2.moments(mask_uint8)
        
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
        centroid_3d = (float(cx), float(cy), float(z))
        # 3. Orientation
        orientation_source = "body"
        orientation_vector = (0.0, 0.0)
        angle_deg = 0.0
        
        parts = obj.get("parts", [])
        nose_part = next((p for p in parts if p.get("label") == "nose"), None)
        eyes_part = next((p for p in parts if p.get("label") == "eyes"), None)
        head_part = next((p for p in parts if p.get("label") == "head"), None)
        
        if nose_part and eyes_part:
            # Snout-Vector Logic
            nx1, ny1, nx2, ny2 = nose_part["bbox_xyxy"]
            ex1, ey1, ex2, ey2 = eyes_part["bbox_xyxy"]
            
            c_nose_x = (nx1 + nx2) / 2.0
            c_nose_y = (ny1 + ny2) / 2.0
            
            c_eyes_x = (ex1 + ex2) / 2.0
            c_eyes_y = (ey1 + ey2) / 2.0
            
            # Vector from eyes to nose
            v_x = c_nose_x - c_eyes_x
            v_y = c_nose_y - c_eyes_y
            
            orientation_vector = (float(v_x), float(v_y))
            orientation_source = "snout_vector"
        elif any(p.get("label", "").lower() in ["headlight", "grille", "front grille"] for p in parts):
            front_part = next((p for p in parts if p.get("label", "").lower() in ["headlight", "grille", "front grille"]), None)
            bx1, by1, bx2, by2 = obj["bbox_xyxy"] # Body bbox
            c_body_x = (bx1 + bx2) / 2.0
            c_body_y = (by1 + by2) / 2.0
            
            fx1, fy1, fx2, fy2 = front_part["bbox_xyxy"]
            c_front_x = (fx1 + fx2) / 2.0
            c_front_y = (fy1 + fy2) / 2.0
            
            v_x = c_front_x - c_body_x
            v_y = c_front_y - c_body_y
            orientation_vector = (float(v_x), float(v_y))
            orientation_source = "vehicle_front_vector"
            
        elif any(p.get("label", "").lower() in ["taillight", "license plate"] for p in parts):
            rear_part = next((p for p in parts if p.get("label", "").lower() in ["taillight", "license plate"]), None)
            bx1, by1, bx2, by2 = obj["bbox_xyxy"]
            c_body_x = (bx1 + bx2) / 2.0
            c_body_y = (by1 + by2) / 2.0
            
            rx1, ry1, rx2, ry2 = rear_part["bbox_xyxy"]
            c_rear_x = (rx1 + rx2) / 2.0
            c_rear_y = (ry1 + ry2) / 2.0
            
            v_x = c_rear_x - c_body_x
            v_y = c_rear_y - c_body_y
            orientation_vector = (float(v_x), float(v_y))
            orientation_source = "vehicle_rear_vector"
        else:
            # Fallback to mask-based orientation
            orientation_mask_uint8 = mask_uint8
            if head_part:
                head_mask_path = head_part.get("mask_path")
                if head_mask_path and os.path.exists(head_mask_path):
                    head_mask = np.load(head_mask_path)
                    orientation_mask_uint8 = (head_mask * 255).astype(np.uint8)
                    orientation_source = "head_bbox_fallback"
                    
            contours, _ = cv2.findContours(orientation_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if len(largest_contour) >= 3:
                    rect = cv2.minAreaRect(largest_contour)
                    angle_deg = rect[2]
                    rad = np.deg2rad(angle_deg)
                    orientation_vector = (float(np.cos(rad)), float(np.sin(rad)))
                    
        obj_result = {
            "id": obj["id"],
            "label": obj["label"],
            "centroid_3d": centroid_3d,
            "orientation_vector": orientation_vector,
            "orientation_angle": float(angle_deg),
            "orientation_source": orientation_source
        }
        results.append(obj_result)
        
    os.makedirs(output_dir, exist_ok=True)
    geom_data_path = os.path.join(output_dir, "geometry_data.json")
    with open(geom_data_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("[Stage 2] Completed.")
    return geom_data_path, results

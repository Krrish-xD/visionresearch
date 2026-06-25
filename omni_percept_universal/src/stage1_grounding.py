import os
import json
import numpy as np
import torch
import supervision as sv
from PIL import Image
from ultralytics import YOLO, FastSAM
import cv2
from .memory_utils import clear_memory

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def run_grounding_and_masking(image_path, output_dir, counting_target=None, location_targets=None):
    """
    Stage 1: Open-Vocabulary Grounding & Masking
    Finds objects autonomously using Florence-2, performs foveal part-segmentation, generates masks, and annotates the image.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 1. Detection (Florence-2 with YOLO-World fallback)
    detections = None
    if counting_target:
        classes = [counting_target]
    elif location_targets:
        classes = [location_targets["subject"], location_targets["reference"]]
    else:
        DEFAULT_CLASSES = "person, dog, cat, car, truck, tree, plant, flower, grass, ball, animal, bird, bicycle, building"
        classes = [cls.strip() for cls in DEFAULT_CLASSES.split(",")]
    
    # parts_map[obj_idx] = [ {"label": label, "bbox_xyxy": [x1, y1, x2, y2]} ]
    parts_map = {}
        
    try:
        print("[Stage 1] Attempting to load Florence-2-large...")
        from transformers import AutoProcessor, AutoModelForCausalLM
        model_id = "microsoft/Florence-2-large"
        
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
        
        orig_image = Image.open(image_path).convert("RGB")
        
        if counting_target:
            # --- 1. Isolated Counting Route (Top-Level) ---
            print(f"[Stage 1] Counting route isolated. Finding '{counting_target}' directly.")
            directional_map = {
                "eye": "left eye, right eye",
                "wheel": "front wheel, rear wheel",
                "door": "front door, rear door",
                "leg": "front leg, rear leg"
            }
            prompt_text = directional_map.get(counting_target.lower(), counting_target)
            task_prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {prompt_text}"
            
            inputs = processor(text=task_prompt, images=orig_image, return_tensors="pt").to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(generated_text, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=(orig_image.width, orig_image.height))
            
            results = parsed_answer.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
            
            all_bboxes = []
            all_labels = []
            
            if "bboxes" in results and len(results["bboxes"]) > 0:
                all_bboxes = results["bboxes"]
                all_labels = results.get("labels", [])
                
            if len(all_bboxes) == 0:
                print(f"[Stage 1] Global detection found 0 '{counting_target}'. Triggering Patch-Based Fallback...")
                w, h = orig_image.size
                mid_x, mid_y = w // 2, h // 2
                patches = [
                    (0, 0, mid_x, mid_y),            # Top-Left
                    (mid_x, 0, w, mid_y),            # Top-Right
                    (0, mid_y, mid_x, h),            # Bottom-Left
                    (mid_x, mid_y, w, h)             # Bottom-Right
                ]
                for px1, py1, px2, py2 in patches:
                    patch = orig_image.crop((px1, py1, px2, py2))
                    inputs = processor(text=task_prompt, images=patch, return_tensors="pt").to(device)
                    try:
                        with torch.no_grad():
                            patch_gen_ids = model.generate(
                                input_ids=inputs["input_ids"],
                                pixel_values=inputs["pixel_values"],
                                max_new_tokens=1024,
                                num_beams=3
                            )
                        patch_gen_text = processor.batch_decode(patch_gen_ids, skip_special_tokens=False)[0]
                        patch_parsed = processor.post_process_generation(patch_gen_text, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=(patch.width, patch.height))
                        
                        patch_results = patch_parsed.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
                        if "bboxes" in patch_results and len(patch_results["bboxes"]) > 0:
                            for box, label in zip(patch_results["bboxes"], patch_results.get("labels", [])):
                                global_box = [box[0] + px1, box[1] + py1, box[2] + px1, box[3] + py1]
                                all_bboxes.append(global_box)
                                all_labels.append(label)
                    except Exception as e:
                        print(f"[Stage 1] Failed patch fallback: {e}")
                        
            # --- 2. Containment NMS ---
            if len(all_bboxes) > 0:
                boxes_with_areas = []
                for i, box in enumerate(all_bboxes):
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    boxes_with_areas.append({
                        "index": i, "bbox": box, "label": all_labels[i], "area": area, "keep": True
                    })
                    
                for i in range(len(boxes_with_areas)):
                    if not boxes_with_areas[i]["keep"]: continue
                    for j in range(i + 1, len(boxes_with_areas)):
                        if not boxes_with_areas[j]["keep"]: continue
                        box1 = boxes_with_areas[i]["bbox"]
                        box2 = boxes_with_areas[j]["bbox"]
                        x1 = max(box1[0], box2[0])
                        y1 = max(box1[1], box2[1])
                        x2 = min(box1[2], box2[2])
                        y2 = min(box1[3], box2[3])
                        
                        inter = max(0, x2 - x1) * max(0, y2 - y1)
                        if inter > 0:
                            area1 = boxes_with_areas[i]["area"]
                            area2 = boxes_with_areas[j]["area"]
                            smaller_area = min(area1, area2)
                            containment = inter / smaller_area if smaller_area > 0 else 0
                            
                            if containment > 0.60:
                                if area1 > area2:
                                    boxes_with_areas[i]["keep"] = False
                                    break
                                else:
                                    boxes_with_areas[j]["keep"] = False
                                    
                valid_boxes = [b for b in boxes_with_areas if b["keep"]]
                xyxy = np.array([b["bbox"] for b in valid_boxes])
                result_labels = [b["label"] for b in valid_boxes]
                confidence = np.ones(len(xyxy))
                
                class_ids = []
                for label in result_labels:
                    matched = False
                    for i, c in enumerate(classes):
                        if c.lower() in label.lower() or label.lower() in c.lower():
                            class_ids.append(i)
                            matched = True
                            break
                    if not matched:
                        classes.append(label)
                        class_ids.append(len(classes) - 1)
                        
                detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=confidence,
                    class_id=np.array(class_ids)
                )
                print(f"[Stage 1] Found {len(detections)} '{counting_target}' objects after NMS.")
        else:
            # --- Pass 1: Global Detection (Multi-Scale Fallback) ---
            if location_targets:
                task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
                prompt_input = f"<CAPTION_TO_PHRASE_GROUNDING>{location_targets['subject']}, {location_targets['reference']}"
            else:
                task_prompt = "<OD>"
                prompt_input = task_prompt
                
            scales = [1.0, 0.75, 0.5, 0.35]
            found_bboxes = False
            xyxy = None
            class_ids = None
            confidence = None
            
            for scale in scales:
                if scale == 1.0:
                    scaled_image = orig_image
                else:
                    new_w = int(orig_image.width * scale)
                    new_h = int(orig_image.height * scale)
                    scaled_image = orig_image.resize((new_w, new_h))
                    
                inputs = processor(text=prompt_input, images=scaled_image, return_tensors="pt").to(device)
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        num_beams=3
                    )
                    
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(scaled_image.width, scaled_image.height))
                
                results = parsed_answer.get(task_prompt, {})
                
                if "bboxes" in results and len(results["bboxes"]) > 0:
                    xyxy = np.array(results["bboxes"]) / scale
                    result_labels = results.get("bboxes_labels", results.get("labels", []))
                    confidence = np.ones(len(xyxy))
                    
                    class_ids = []
                    for label in result_labels:
                        matched = False
                        for i, c in enumerate(classes):
                            if c.lower() in label.lower() or label.lower() in c.lower():
                                class_ids.append(i)
                                matched = True
                                break
                        if not matched:
                            classes.append(label)
                            class_ids.append(len(classes) - 1)
                            
                    print(f"[Stage 1] Detection succeeded at scale: {scale}")
                    found_bboxes = True
                    break
                    
            if found_bboxes:
                detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=confidence,
                    class_id=np.array(class_ids)
                )
                
                # --- Pass 2: Foveal Part-Segmentation ---
                print(f"[Stage 1] Found {len(detections)} objects. Running foveal crop part-detection...")
                
                parts_map_vocab = {
                    "person": "nose, eyes", "man": "nose, eyes", "woman": "nose, eyes", 
                    "boy": "nose, eyes", "girl": "nose, eyes",
                    "dog": "nose, eyes", "cat": "nose, eyes", "eagle": "nose, eyes", "bird": "nose, eyes",
                    "truck": "headlight, taillight, license plate, front grille", 
                    "car": "headlight, taillight, license plate, front grille", 
                    "bus": "headlight, taillight, license plate, front grille"
                }
                
                for idx, box in enumerate(xyxy):
                    parts_map[idx] = []
                    x1, y1, x2, y2 = [int(v) for v in box]
                    # Ensure valid crop
                    if x2 > x1 and y2 > y1:
                        crop = orig_image.crop((x1, y1, x2, y2))
                        
                        main_label = "head"
                        if len(class_ids) > idx:
                            main_label = classes[class_ids[idx]].lower()
                            
                        part_prompt = None
                        for key, prompt in parts_map_vocab.items():
                            if key in main_label:
                                part_prompt = prompt
                                break
                        
                        if not part_prompt:
                            part_prompt = None
                            
                        if part_prompt and "head" not in part_prompt and not any(v in main_label for v in ["bus", "car", "truck", "vehicle", "van", "train", "plane"]):
                            part_prompt += ", head"
                            
                        if not part_prompt:
                            print(f"[Stage 1] Skipping part-detection for '{main_label}' (No specific parts mapped for this object).")
                            continue
                            
                        part_prompt_full = f"<CAPTION_TO_PHRASE_GROUNDING> {part_prompt}"
                        print(f"[Stage 1] Looking for '{part_prompt}' on '{main_label}'")
                            
                        crop_inputs = processor(text=part_prompt_full, images=crop, return_tensors="pt").to(device)
                        try:
                            with torch.no_grad():
                                crop_gen_ids = model.generate(
                                    input_ids=crop_inputs["input_ids"],
                                    pixel_values=crop_inputs["pixel_values"],
                                    max_new_tokens=1024,
                                    num_beams=3
                                )
                            crop_gen_text = processor.batch_decode(crop_gen_ids, skip_special_tokens=False)[0]
                            crop_parsed = processor.post_process_generation(crop_gen_text, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=(crop.width, crop.height))
                            
                            crop_results = crop_parsed.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
                            
                            if "bboxes" in crop_results and len(crop_results["bboxes"]) > 0:
                                cb_boxes = crop_results["bboxes"]
                                cb_labels = crop_results.get("labels", [])
                                
                                for c_idx, c_label in enumerate(cb_labels):
                                    c_label_lower = c_label.lower()
                                    final_label = None
                                    
                                    vocab_words = [w.strip() for w in part_prompt.split(",")]
                                    for vocab_word in vocab_words:
                                        if vocab_word in c_label_lower:
                                            final_label = vocab_word
                                            break
                                    if not final_label and "eye" in c_label_lower:
                                        final_label = "eyes"
                                        
                                    if final_label:
                                        hx1, hy1, hx2, hy2 = cb_boxes[c_idx]
                                        global_hx1 = hx1 + x1
                                        global_hy1 = hy1 + y1
                                        global_hx2 = hx2 + x1
                                        global_hy2 = hy2 + y1
                                        
                                        # Don't add duplicate labels
                                        if not any(p["label"] == final_label for p in parts_map[idx]):
                                            parts_map[idx].append({
                                                "label": final_label, 
                                                "bbox_xyxy": [global_hx1, global_hy1, global_hx2, global_hy2]
                                            })
                        except Exception as e:
                            print(f"[Stage 1] Failed to detect sub-features for object {idx}: {e}")
            else:
                raise ValueError("Florence-2 returned no bounding boxes.")
        del model
        del processor
        clear_memory()
        print("[Stage 1] Successfully used Florence-2-large.")
        
    except Exception as e:
        print(f"[Stage 1] Florence-2 failed ({e}). Falling back to YOLO-World.")
        clear_memory()
        
        # Fallback to YOLO-World
        model = YOLO('yolov8s-worldv2.pt')
        model.set_classes(classes)
        
        # Inference
        results = model.predict(image_path, device=device, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        del model
        clear_memory()
    
    if detections is None or len(detections) == 0:
        print("[Stage 1] No objects detected.")
        return None, None, []
        
    # Combine bounding boxes for FastSAM (Main object, 'head', and counting_target parts)
    boxes_xyxy = detections.xyxy.tolist()
    head_indices = {} # Map from obj_idx -> combined_index
    current_idx = len(boxes_xyxy)
    
    for obj_idx, parts_list in parts_map.items():
        for part in parts_list:
            if part["label"] == "head" or (counting_target and counting_target.lower() in part["label"].lower()):
                boxes_xyxy.append(part["bbox_xyxy"])
                if part["label"] == "head":
                    head_indices[obj_idx] = current_idx
                part["mask_idx"] = current_idx
                current_idx += 1
        
    combined_boxes_np = np.array(boxes_xyxy)
    
    print(f"[Stage 1] Running FastSAM globally for main objects and heads...")
    # 2. Segmentation (FastSAM)
    fastsam = FastSAM('FastSAM-s.pt')
    
    sam_results = fastsam(image_path, device=device, bboxes=combined_boxes_np, verbose=False)
    
    combined_masks = []
    if sam_results and sam_results[0].masks is not None:
        orig_shape = sam_results[0].orig_shape
        masks_data = sam_results[0].masks.data.cpu().numpy()
        
        for m in masks_data:
            m_resized = cv2.resize(m, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
            combined_masks.append(m_resized > 0)
            
        # Pad if missing
        empty_mask = np.zeros((orig_shape[0], orig_shape[1]), dtype=bool)
        while len(combined_masks) < len(combined_boxes_np):
            combined_masks.append(empty_mask)
    else:
        print("[Stage 1] FastSAM failed to generate masks.")
        img_h, img_w = cv2.imread(image_path).shape[:2]
        empty_mask = np.zeros((img_h, img_w), dtype=bool)
        combined_masks = [empty_mask] * len(combined_boxes_np)
        
    # Assign masks back
    main_masks = combined_masks[:len(detections)]
    detections.mask = np.array(main_masks)
    
    del fastsam
    clear_memory()
    
    # Annotation block moved to the end of the script to filter out discarded items
    
    # 3. Absolute Area Visibility Filter
    # Prepare JSON output
    output_data = []
    for i in range(len(detections)):
        class_id = detections.class_id[i] if detections.class_id is not None else 0
        cls_name = classes[class_id] if class_id < len(classes) else f"Obj_{i}"
        
        # Check if main object is filtered out
        if counting_target and counting_target.lower() in cls_name.lower():
            area = detections.mask[i].sum() if detections.mask is not None else 0
            if area == 0:
                x1, y1, x2, y2 = detections.xyxy[i]
                area = (x2 - x1) * (y2 - y1)
            if area < 300:
                print(f"[Stage 1] Visibility Filter: Discarded main object '{cls_name}' with area {area} (< 300)")
                continue
        
        mask_path = os.path.join(output_dir, f"mask_{i}.npy")
        if detections.mask is not None:
            np.save(mask_path, detections.mask[i])
            
        obj_data = {
            "id": i,
            "label": cls_name,
            "bbox_xyxy": detections.xyxy[i].tolist(),
            "confidence": float(detections.confidence[i]) if detections.confidence is not None else 1.0,
            "mask_path": mask_path if detections.mask is not None else None,
            "parts": []
        }
        
        if i in parts_map:
            for part in parts_map[i]:
                part_data = {
                    "label": part["label"],
                    "bbox_xyxy": part["bbox_xyxy"]
                }
                
                # Verify part filter
                if counting_target and counting_target.lower() in part["label"].lower() and "mask_idx" in part:
                    mask_idx = part["mask_idx"]
                    area = combined_masks[mask_idx].sum() if mask_idx < len(combined_masks) else 0
                    if area == 0:
                        x1, y1, x2, y2 = part["bbox_xyxy"]
                        area = (x2 - x1) * (y2 - y1)
                    if area < 300:
                        print(f"[Stage 1] Visibility Filter: Discarded part '{part['label']}' with area {area} (< 300)")
                        continue
                
                if "mask_idx" in part:
                    part_mask_path = os.path.join(output_dir, f"mask_{i}_{part['label'].replace(' ', '_')}_{part['mask_idx']}.npy")
                    mask_idx = part["mask_idx"]
                    np.save(part_mask_path, combined_masks[mask_idx])
                    part_data["mask_path"] = part_mask_path
                elif part["label"] == "head" and i in head_indices:
                    head_mask_path = os.path.join(output_dir, f"mask_{i}_head.npy")
                    mask_idx = head_indices[i]
                    np.save(head_mask_path, combined_masks[mask_idx])
                    part_data["mask_path"] = head_mask_path
                    
                obj_data["parts"].append(part_data)
            
        output_data.append(obj_data)
        
    json_path = os.path.join(output_dir, "grounding_data.json")
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    # 3. Annotation (Filtered Valid Detections Only)
    print("[Stage 1] Annotating valid objects...")
    image = cv2.imread(image_path)
    annotated_image = image.copy()
    
    valid_main_boxes = []
    valid_main_labels = []
    valid_main_masks = []
    
    valid_part_boxes = []
    valid_part_labels = []
    valid_part_masks = []
    
    for obj in output_data:
        is_target = counting_target and counting_target.lower() in obj["label"].lower()
        
        if is_target:
            # Draw as a RED part with a label
            valid_part_boxes.append(obj["bbox_xyxy"])
            valid_part_labels.append(obj["label"])
            if obj["mask_path"] and os.path.exists(obj["mask_path"]):
                valid_part_masks.append(np.load(obj["mask_path"]))
        else:
            # Draw as a MAIN object
            valid_main_boxes.append(obj["bbox_xyxy"])
            valid_main_labels.append(f'{obj["label"]} {obj["confidence"]:.2f}')
            if obj["mask_path"] and os.path.exists(obj["mask_path"]):
                valid_main_masks.append(np.load(obj["mask_path"]))
                
        # Also process sub-parts (like if a car has a wheel part)
        for part in obj["parts"]:
            is_part_target = counting_target and counting_target.lower() in part["label"].lower()
            # Draw if it's a counting target, OR if it's a facial feature used for orientation
            if is_part_target or part["label"].lower() in ["head", "nose", "eyes"]:
                valid_part_boxes.append(part["bbox_xyxy"])
                valid_part_labels.append(part["label"])
                if "mask_path" in part and os.path.exists(part["mask_path"]):
                    valid_part_masks.append(np.load(part["mask_path"]))

    # Main object annotation (Default color)
    if len(valid_main_boxes) > 0:
        main_mask_array = np.array(valid_main_masks) if len(valid_main_masks) == len(valid_main_boxes) else None
        main_detections = sv.Detections(
            xyxy=np.array(valid_main_boxes),
            mask=main_mask_array,
            class_id=np.zeros(len(valid_main_boxes), dtype=int)
        )
        box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.DEFAULT)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=main_detections)
        
        if main_detections.mask is not None:
            mask_annotator = sv.MaskAnnotator(color=sv.ColorPalette.DEFAULT)
            annotated_image = mask_annotator.annotate(scene=annotated_image, detections=main_detections)
            
        try:
            label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.DEFAULT)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=main_detections, labels=valid_main_labels)
        except AttributeError:
            pass

    # Part annotation (Red color)
    if len(valid_part_boxes) > 0:
        part_mask_array = np.array(valid_part_masks) if len(valid_part_masks) == len(valid_part_boxes) else None
        part_detections = sv.Detections(
            xyxy=np.array(valid_part_boxes),
            mask=part_mask_array,
            class_id=np.zeros(len(valid_part_boxes), dtype=int)
        )
        red_color = sv.ColorPalette([sv.Color(r=255, g=0, b=0)])
        part_box_annotator = sv.BoxAnnotator(color=red_color)
        annotated_image = part_box_annotator.annotate(scene=annotated_image, detections=part_detections)
        
        if part_detections.mask is not None:
            part_mask_annotator = sv.MaskAnnotator(color=red_color)
            annotated_image = part_mask_annotator.annotate(scene=annotated_image, detections=part_detections)
            
        try:
            part_label_annotator = sv.LabelAnnotator(color=red_color)
            annotated_image = part_label_annotator.annotate(scene=annotated_image, detections=part_detections, labels=valid_part_labels)
        except AttributeError:
            pass

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    annotated_img_path = os.path.join(output_dir, "annotated_image.jpg")
    cv2.imwrite(annotated_img_path, annotated_image)
        
    print("[Stage 1] Completed.")
    return annotated_img_path, json_path, output_data

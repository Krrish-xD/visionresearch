import os
import json
import numpy as np
import torch
import supervision as sv
from PIL import Image
from ultralytics import YOLO, FastSAM
import cv2
from .memory_utils import clear_memory

def run_grounding_and_masking(image_path, output_dir, counting_target=None):
    """
    Stage 1: Open-Vocabulary Grounding & Masking
    Finds objects autonomously using Florence-2, performs foveal part-segmentation, generates masks, and annotates the image.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 1. Detection (Florence-2 with YOLO-World fallback)
    detections = None
    if counting_target:
        classes = [counting_target]
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
        
        # --- Pass 1: Global Detection (Multi-Scale Fallback) ---
        task_prompt = "<OD>"
            
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
                
            inputs = processor(text=task_prompt, images=scaled_image, return_tensors="pt").to(device)
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
                
        if not found_bboxes and counting_target:
            print("[Stage 1] Multi-Scale failed. Triggering Patch-Based Dense Grounding...")
            w, h = orig_image.size
            mid_x, mid_y = w // 2, h // 2
            patches = [
                (0, 0, mid_x, mid_y),            # Top-Left
                (mid_x, 0, w, mid_y),            # Top-Right
                (0, mid_y, mid_x, h),            # Bottom-Left
                (mid_x, mid_y, w, h)             # Bottom-Right
            ]
            
            all_bboxes = []
            all_labels = []
            
            patch_task_prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {counting_target}"
            for px1, py1, px2, py2 in patches:
                patch = orig_image.crop((px1, py1, px2, py2))
                inputs = processor(text=patch_task_prompt, images=patch, return_tensors="pt").to(device)
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        num_beams=3
                    )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = processor.post_process_generation(generated_text, task=patch_task_prompt, image_size=(patch.width, patch.height))
                results = parsed_answer.get(patch_task_prompt, {})
                
                if "bboxes" in results and len(results["bboxes"]) > 0:
                    for box, label in zip(results["bboxes"], results.get("bboxes_labels", results.get("labels", []))):
                        global_box = [box[0] + px1, box[1] + py1, box[2] + px1, box[3] + py1]
                        all_bboxes.append(global_box)
                        all_labels.append(label)
                        
            if all_bboxes:
                xyxy = np.array(all_bboxes)
                result_labels = all_labels
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
                        
                found_bboxes = True
                print(f"[Stage 1] Patch-Based Dense Grounding succeeded. Found {len(xyxy)} parts.")
                
        if found_bboxes:
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=np.array(class_ids)
            )
            
            # --- Pass 2: Foveal Part-Segmentation ---
            print(f"[Stage 1] Found {len(detections)} objects. Running foveal crop part-detection...")
            
            parts_map_vocab = {
                "dog": "nose, eyes", "cat": "nose, eyes", "eagle": "nose, eyes", "bird": "nose, eyes",
                "truck": "door, wheel, window", "car": "door, wheel, window", "bus": "door, wheel, window"
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
                        
                    if counting_target:
                        part_prompt = counting_target
                    else:
                        part_prompt = None
                        for key, prompt in parts_map_vocab.items():
                            if key in main_label:
                                part_prompt = prompt
                                break
                        
                        if not part_prompt:
                            non_living_keywords = ["mirror", "wheel", "window", "door", "light", "reflector", "wiper", "bumper", "tire", "plate", "windshield", "engine", "seat", "handle", "sign", "tree", "plant", "flower", "grass", "road", "sky", "building", "house", "wall", "fence", "pole", "wire"]
                            if any(kw in main_label for kw in non_living_keywords):
                                part_prompt = None
                            else:
                                part_prompt = "nose, eyes, head"
                                
                        if part_prompt and "head" not in part_prompt and not any(v in main_label for v in ["bus", "car", "truck", "vehicle", "van", "train", "plane"]):
                            part_prompt += ", head"
                            
                    if not part_prompt:
                        print(f"[Stage 1] Skipping part-detection for '{main_label}' (Non-living or no parts specified).")
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
    
    # 3. Annotation
    print("[Stage 1] Annotating image...")
    image = cv2.imread(image_path)
    
    # Main object annotation
    box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.DEFAULT)
    if detections.mask is not None:
        mask_annotator = sv.MaskAnnotator(color=sv.ColorPalette.DEFAULT)
        annotated_image = mask_annotator.annotate(scene=image, detections=detections)
    else:
        annotated_image = image.copy()
        
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    
    labels = []
    for i in range(len(detections)):
        class_id = detections.class_id[i] if detections.class_id is not None else 0
        cls_name = classes[class_id] if class_id < len(classes) else f"Obj_{i}"
        conf = detections.confidence[i] if detections.confidence is not None else 1.0
        labels.append(f"{cls_name} {conf:.2f}")
    
    try:
        label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.DEFAULT)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    except AttributeError:
        pass
        
    # Part annotation in Red (heads, noses, eyes)
    part_xyxy = []
    part_labels = []
    head_masks_arr = []
    head_xyxy = []
    
    for obj_idx, parts_list in parts_map.items():
        for part in parts_list:
            part_xyxy.append(part["bbox_xyxy"])
            part_labels.append(part["label"])
            if part["label"] == "head":
                head_xyxy.append(part["bbox_xyxy"])
                mask_idx = head_indices[obj_idx]
                head_masks_arr.append(combined_masks[mask_idx])
                
    red_color = sv.ColorPalette([sv.Color(r=255, g=0, b=0)])
    
    # Annotate head masks if any exist
    if len(head_xyxy) > 0:
        head_detections = sv.Detections(
            xyxy=np.array(head_xyxy),
            mask=np.array(head_masks_arr),
            class_id=np.zeros(len(head_xyxy), dtype=int)
        )
        part_mask_annotator = sv.MaskAnnotator(color=red_color)
        annotated_image = part_mask_annotator.annotate(scene=annotated_image, detections=head_detections)
        
    # Annotate all part bboxes
    if len(part_xyxy) > 0:
        all_part_detections = sv.Detections(
            xyxy=np.array(part_xyxy),
            class_id=np.zeros(len(part_xyxy), dtype=int)
        )
        part_box_annotator = sv.BoxAnnotator(color=red_color)
        annotated_image = part_box_annotator.annotate(scene=annotated_image, detections=all_part_detections)
        
        try:
            part_label_annotator = sv.LabelAnnotator(color=red_color)
            annotated_image = part_label_annotator.annotate(scene=annotated_image, detections=all_part_detections, labels=part_labels)
        except AttributeError:
            pass

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    annotated_img_path = os.path.join(output_dir, "annotated_image.jpg")
    cv2.imwrite(annotated_img_path, annotated_image)
    
    # Calculate mask area threshold for counting_target
    max_target_area = 0
    if counting_target:
        target_lower = counting_target.lower()
        if detections.mask is not None:
            for i in range(len(detections)):
                class_id = detections.class_id[i] if detections.class_id is not None else 0
                cls_name = classes[class_id] if class_id < len(classes) else f"Obj_{i}"
                if target_lower in cls_name.lower():
                    area = detections.mask[i].sum()
                    if area > max_target_area:
                        max_target_area = area
                        
        for obj_idx, parts_list in parts_map.items():
            for part in parts_list:
                if target_lower in part["label"].lower() and "mask_idx" in part:
                    mask_idx = part["mask_idx"]
                    if mask_idx < len(combined_masks):
                        area = combined_masks[mask_idx].sum()
                        if area > max_target_area:
                            max_target_area = area
                            
    area_threshold = 0.15 * max_target_area

    # Prepare JSON output
    output_data = []
    for i in range(len(detections)):
        class_id = detections.class_id[i] if detections.class_id is not None else 0
        cls_name = classes[class_id] if class_id < len(classes) else f"Obj_{i}"
        
        # Check if main object is filtered out
        if counting_target and counting_target.lower() in cls_name.lower() and detections.mask is not None:
            area = detections.mask[i].sum()
            if area < area_threshold or area < 500:
                print(f"[Stage 1] Visibility Filter: Discarded main object '{cls_name}' with area {area} (< {area_threshold} or < 500)")
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
                    area = combined_masks[mask_idx].sum()
                    if area < area_threshold or area < 500:
                        print(f"[Stage 1] Visibility Filter: Discarded part '{part['label']}' with area {area} (< {area_threshold} or < 500)")
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
        
    print("[Stage 1] Completed.")
    return annotated_img_path, json_path, output_data

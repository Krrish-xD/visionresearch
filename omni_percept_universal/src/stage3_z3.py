import os
import math
import numpy as np
import z3
import cv2
from .memory_utils import clear_memory

def run_z3_prover(grounding_data, geometry_data, output_dir, question_category="GENERAL", relevant_objects=None, location_targets=None):
    """
    Stage 3: Universal Z3 Theorem Proving
    Proves spatial relationships (distance, occlusion, facing) based on extracted features.
    """
    print("[Stage 3] Initializing Z3 Solver...")
    solver = z3.Solver()
    
    proofs = []
    
    if relevant_objects is None:
        relevant_objects = []
        
    # Absolute Orientation Logic Branch
    if question_category == "ABSOLUTE_ORIENTATION":
        for target_obj in relevant_objects:
            target_id = target_obj['id']
            geom = next((g for g in geometry_data if g['id'] == target_id), None)
            if geom:
                label = f"{geom['label']}_{geom['id']}"
                vec_x = geom["orientation_vector"][0]
                source = geom.get("orientation_source", "body")
                
                facing_left = z3.Bool(f'{label}_facing_left')
                facing_right = z3.Bool(f'{label}_facing_right')
                facing_camera = z3.Bool(f'{label}_facing_camera')
                facing_away = z3.Bool(f'{label}_facing_away')
                
                if source == "snout_vector":
                    solver.add(facing_left == (vec_x < -5))
                    solver.add(facing_right == (vec_x > 5))
                    solver.add(facing_camera == (vec_x >= -5 and vec_x <= 5))
                    
                    if solver.check() == z3.sat:
                        model = solver.model()
                        if z3.is_true(model[facing_left]):
                            proofs.append(f"[Z3] PROVED: {label} is facing LEFT (Nose is left of eyes: Vx = {vec_x:.2f})")
                        elif z3.is_true(model[facing_right]):
                            proofs.append(f"[Z3] PROVED: {label} is facing RIGHT (Nose is right of eyes: Vx = {vec_x:.2f})")
                        elif z3.is_true(model[facing_camera]):
                            proofs.append(f"[Z3] PROVED: {label} is facing CAMERA (Nose aligns with eyes: Vx = {vec_x:.2f})")
                elif source == "vehicle_front_vector":
                    solver.add(facing_left == (vec_x < -5))
                    solver.add(facing_right == (vec_x > 5))
                    solver.add(facing_camera == (vec_x >= -5 and vec_x <= 5))
                    
                    if solver.check() == z3.sat:
                        model = solver.model()
                        if z3.is_true(model[facing_left]):
                            proofs.append(f"[Z3] PROVED: {label} is facing LEFT (Front headlight is left of body: Vx = {vec_x:.2f})")
                        elif z3.is_true(model[facing_right]):
                            proofs.append(f"[Z3] PROVED: {label} is facing RIGHT (Front headlight is right of body: Vx = {vec_x:.2f})")
                        elif z3.is_true(model[facing_camera]):
                            proofs.append(f"[Z3] PROVED: {label} is facing CAMERA (Front headlight is centered on body: Vx = {vec_x:.2f})")
                elif source == "vehicle_rear_vector":
                    solver.add(facing_left == (vec_x > 5))
                    solver.add(facing_right == (vec_x < -5))
                    solver.add(facing_away == (vec_x >= -5 and vec_x <= 5))
                    
                    if solver.check() == z3.sat:
                        model = solver.model()
                        if z3.is_true(model[facing_left]):
                            proofs.append(f"[Z3] PROVED: {label} is facing LEFT (Taillight is right of body: Vx = {vec_x:.2f})")
                        elif z3.is_true(model[facing_right]):
                            proofs.append(f"[Z3] PROVED: {label} is facing RIGHT (Taillight is left of body: Vx = {vec_x:.2f})")
                        elif z3.is_true(model[facing_away]):
                            proofs.append(f"[Z3] PROVED: {label} is facing AWAY (Taillight/license plate is centered on body: Vx = {vec_x:.2f})")
                else:
                    if vec_x < 0:
                        solver.add(facing_left == True)
                        solver.add(facing_right == False)
                    else:
                        solver.add(facing_right == True)
                        solver.add(facing_left == False)
                        
                    if solver.check() == z3.sat:
                        model = solver.model()
                        if z3.is_true(model[facing_left]):
                            proofs.append(f"[Z3] PROVED: {label} is facing LEFT (Vector derived from {source} mask: x = {vec_x:.2f}).")
                        elif z3.is_true(model[facing_right]):
                            proofs.append(f"[Z3] PROVED: {label} is facing RIGHT (Vector derived from {source} mask: x = {vec_x:.2f}).")
                solver.reset()
                
    elif question_category == "RELATIVE_LOCATION" and location_targets:
        subject = location_targets["subject"]
        reference = location_targets["reference"]
        
        obj_sub = next((g for g in geometry_data if subject.lower() in g['label'].lower()), None)
        obj_ref = next((g for g in geometry_data if reference.lower() in g['label'].lower()), None)
        
        if obj_sub:
            x_sub = obj_sub["centroid_3d"][0]
            label_sub = f"{obj_sub['label']}_{obj_sub['id']}"
            
            subject_on_left = z3.Bool(f'{label_sub}_on_left')
            subject_on_right = z3.Bool(f'{label_sub}_on_right')
            
            if obj_ref:
                x_ref = obj_ref["centroid_3d"][0]
                label_ref = f"{obj_ref['label']}_{obj_ref['id']}"
                
                solver.add(subject_on_left == (x_sub < x_ref))
                solver.add(subject_on_right == (x_sub > x_ref))
                
                if solver.check() == z3.sat:
                    model = solver.model()
                    if z3.is_true(model[subject_on_left]):
                        proofs.append(f"[Z3] PROVED: {subject} is on the LEFT of the {reference} (Subject X: {x_sub:.2f} < Reference X: {x_ref:.2f})")
                    elif z3.is_true(model[subject_on_right]):
                        proofs.append(f"[Z3] PROVED: {subject} is on the RIGHT of the {reference} (Subject X: {x_sub:.2f} > Reference X: {x_ref:.2f})")
            else:
                obj_sub_ground = next((g for g in grounding_data if g['id'] == obj_sub['id']), None)
                x_ref = 512
                if obj_sub_ground and obj_sub_ground.get("mask_path") and os.path.exists(obj_sub_ground["mask_path"]):
                    mask = np.load(obj_sub_ground["mask_path"])
                    x_ref = mask.shape[1] / 2.0
                
                solver.add(subject_on_left == (x_sub < x_ref))
                solver.add(subject_on_right == (x_sub > x_ref))
                
                if solver.check() == z3.sat:
                    model = solver.model()
                    if z3.is_true(model[subject_on_left]):
                        proofs.append(f"[Z3] PROVED: {subject} is on the LEFT of the image center (Subject X: {x_sub:.2f} < Center X: {x_ref:.2f})")
                    elif z3.is_true(model[subject_on_right]):
                        proofs.append(f"[Z3] PROVED: {subject} is on the RIGHT of the image center (Subject X: {x_sub:.2f} > Center X: {x_ref:.2f})")
            solver.reset()
    for i in range(len(geometry_data)):
        for j in range(len(geometry_data)):
            if i == j:
                continue
                
            obj_a_geom = geometry_data[i]
            obj_b_geom = geometry_data[j]
            
            obj_a_ground = grounding_data[i]
            obj_b_ground = grounding_data[j]
            
            label_a = f"{obj_a_geom['label']}_{obj_a_geom['id']}"
            label_b = f"{obj_b_geom['label']}_{obj_b_geom['id']}"
            
            # Logic A: Relative Position (Distance)
            cx_a, cy_a, cz_a = obj_a_geom["centroid_3d"]
            cx_b, cy_b, cz_b = obj_b_geom["centroid_3d"]
            
            # Use strict typing as requested
            dist_var = z3.Real(f'Dist_{label_a}_{label_b}')
            
            actual_dist = math.sqrt((cx_b - cx_a)**2 + (cy_b - cy_a)**2 + (cz_b - cz_a)**2)
            solver.add(dist_var == actual_dist)
            
            # Logic B: Occlusion
            occludes_var = z3.Bool(f'{label_a}_occludes_{label_b}')
            
            mask_a_path = obj_a_ground.get("mask_path")
            mask_b_path = obj_b_ground.get("mask_path")
            
            is_occluding = False
            if mask_a_path and mask_b_path and os.path.exists(mask_a_path) and os.path.exists(mask_b_path):
                mask_a = np.load(mask_a_path)
                mask_b = np.load(mask_b_path)
                
                # Check 2D intersection
                intersection = np.logical_and(mask_a, mask_b)
                if np.any(intersection):
                    # Intersects. A occludes B if A's distance is smaller (closer to camera)
                    if cz_a < cz_b:
                        is_occluding = True
            
            solver.add(occludes_var == is_occluding)
            
            # Logic C: Orientation/Facing
            facing_var = z3.Bool(f'{label_a}_facing_{label_b}')
            
            vec_ab_x = cx_b - cx_a
            vec_ab_y = cy_b - cy_a
            mag_ab = math.sqrt(vec_ab_x**2 + vec_ab_y**2)
            
            is_facing = False
            if mag_ab > 0:
                vec_ab_norm = (vec_ab_x / mag_ab, vec_ab_y / mag_ab)
                orient_a = obj_a_geom["orientation_vector"]
                
                # Dot product
                dot_prod = orient_a[0] * vec_ab_norm[0] + orient_a[1] * vec_ab_norm[1]
                
                # High dot product (>0.707 for 45 deg) means axes align
                if abs(dot_prod) > 0.707:
                    is_facing = True
                    
            solver.add(facing_var == is_facing)
            
            # Check satisfiability and extract proven facts
            if solver.check() == z3.sat:
                model = solver.model()
                
                dist_val = model[dist_var]
                if isinstance(dist_val, z3.RatNumRef):
                    dist_float = float(dist_val.numerator_as_long()) / float(dist_val.denominator_as_long())
                else:
                    dist_float = actual_dist
                
                # Avoid symmetrical distance spam
                if i < j:
                    proofs.append(f"[Z3] PROVED: Distance between {label_a} and {label_b} is {dist_float:.2f} units.")
                
                # These are directional, so they apply for both i->j and j->i
                if z3.is_true(model[occludes_var]):
                    proofs.append(f"[Z3] PROVED: {label_a} physically occludes {label_b}.")
                    
                if z3.is_true(model[facing_var]):
                    source_a = obj_a_geom.get("orientation_source", "body")
                    # Only generate this generic proof if we don't have a specific vector already
                    if source_a not in ["snout_vector", "vehicle_front_vector", "vehicle_rear_vector"]:
                        proofs.append(f"[Z3] PROVED: {label_a} is oriented towards/aligns with {label_b} (Vector derived from {source_a} mask).")
                    
            # Reset for clean slate next iteration
            solver.reset()
            
    proof_string = "\n".join(proofs) if proofs else "[Z3] No spatial relationships proven."
    
    print("[Stage 3] Z3 Proofs generated.")
    
    # Z3 is pure CPU math. Run clear_memory() after formatting output as requested.
    clear_memory()
    
    return proof_string

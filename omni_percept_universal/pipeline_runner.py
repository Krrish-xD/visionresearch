import os
import gradio as gr
from src.stage1_grounding import run_grounding_and_masking
from src.stage2_geometry import run_geometry_extraction
from src.stage3_z3 import run_z3_prover
from src.stage4_rag import run_rag_retrieval
from src.stage5_vlm import run_vlm_generation
from src.memory_utils import clear_memory
from src.question_parser import parse_question_type

def execute_pipeline(image_path, critical_question):
    """
    Executes the Omni-Percept Universal Pipeline sequentially.
    Yields exactly 4 outputs to match Gradio UI components:
    (annotated_image, z3_terminal, rag_context, final_report)
    Pro-Tip 2 Applied: Use gr.update() for unchanged components to prevent tuple length mismatch.
    """
    if not image_path:
        yield (gr.update(), gr.update(), gr.update(), "⚠️ Please upload an image.")
        return
        
    if not critical_question:
        yield (gr.update(), gr.update(), gr.update(), "⚠️ Please provide a critical question.")
        return

    output_dir = "data/temp"
    os.makedirs(output_dir, exist_ok=True)
    
    # Smart Parse (Early Routing)
    category, relevant_objects, target_noun = parse_question_type(critical_question)
    
    if category in ["ATTRIBUTE", "GENERAL"]:
        yield (gr.update(), "Skipped (Non-Spatial Query)", "Skipped (Non-Spatial Query)", "Routing: Direct VLM (Non-Spatial Query)...")
        final_report = run_vlm_generation(None, None, critical_question, None, image_path, direct_vlm=True)
        yield (gr.update(), "Skipped (Non-Spatial Query)", "Skipped (Non-Spatial Query)", final_report)
        return
        
    if category == "COUNTING":
        yield (gr.update(), "Skipped (Counting Query)", "Skipped (Counting Query)", "Routing: Mathematical Counting (Florence-2 + FastSAM)...")
        annotated_img_path, json_path, grounding_data = run_grounding_and_masking(image_path, output_dir, counting_target=target_noun)
        clear_memory()
        
        mask_count = 0
        if grounding_data and target_noun:
            for obj in grounding_data:
                label = obj.get("label", "")
                if label and target_noun.lower() in label.lower():
                    mask_count += 1
                for part in obj.get("parts", []):
                    part_label = part.get("label", "")
                    if part_label and target_noun.lower() in part_label.lower():
                        mask_count += 1
                        
        final_report = run_vlm_generation(None, None, critical_question, None, image_path, mask_count=mask_count, counting_target=target_noun)
        yield (annotated_img_path if annotated_img_path else gr.update(), "Skipped (Counting Query)", "Skipped (Counting Query)", final_report)
        return

    if category == "PRESENCE":
        yield (gr.update(), "Skipped (Presence Query)", "Skipped (Presence Query)", "Routing: Provable Presence (Florence-2)...")
        annotated_img_path, json_path, grounding_data = run_grounding_and_masking(image_path, output_dir, counting_target=target_noun)
        clear_memory()
        
        mask_count = 0
        if grounding_data and target_noun:
            for obj in grounding_data:
                label = obj.get("label", "")
                if label and target_noun.lower() in label.lower():
                    mask_count += 1
                for part in obj.get("parts", []):
                    part_label = part.get("label", "")
                    if part_label and target_noun.lower() in part_label.lower():
                        mask_count += 1
                        
        final_report = run_vlm_generation(None, None, critical_question, None, image_path, mask_count=mask_count, counting_target=target_noun, presence_mode=True)
        yield (annotated_img_path if annotated_img_path else gr.update(), "Skipped (Presence Query)", "Skipped (Presence Query)", final_report)
        return

    # STAGE 1: Grounding & Masking
    yield (gr.update(), "Stage 1: Grounding & Masking running...", gr.update(), "Stage 1 in progress...")
    if category == "RELATIVE_LOCATION":
        annotated_img_path, json_path, grounding_data = run_grounding_and_masking(image_path, output_dir, location_targets=target_noun)
    else:
        annotated_img_path, json_path, grounding_data = run_grounding_and_masking(image_path, output_dir)
    clear_memory()
    
    if not grounding_data:
        yield (
            annotated_img_path if annotated_img_path else gr.update(), 
            "No objects found.", 
            "No objects found.", 
            "⚠️ Pipeline stopped: No target objects were detected."
        )
        return
        
    # Re-extract objects now that grounding_data is available
    _, relevant_objects, target_noun_re = parse_question_type(critical_question, grounding_data)
        
    # STAGE 2: Geometry Extraction
    yield (
        annotated_img_path, 
        "Stage 2: Geometry Extraction running...", 
        gr.update(), 
        "Stage 2 in progress..."
    )
    geom_data_path, geometry_data = run_geometry_extraction(image_path, grounding_data, output_dir)
    clear_memory()
    
    # STAGE 3: Z3 Prover
    yield (
        gr.update(), 
        "Stage 3: Z3 Prover running...", 
        gr.update(), 
        "Stage 3 in progress..."
    )
    location_targets = target_noun_re if category == "RELATIVE_LOCATION" else None
    z3_facts = run_z3_prover(grounding_data, geometry_data, output_dir, category, relevant_objects, location_targets=location_targets)
    clear_memory()
    
    # STAGE 4: Local RAG
    yield (
        gr.update(), 
        z3_facts, 
        "Stage 4: Local RAG running...", 
        "Stage 4 in progress..."
    )
    rag_context = run_rag_retrieval(z3_facts, output_dir)
    clear_memory()
    
    # STAGE 5: Constrained Generation
    yield (
        gr.update(), 
        gr.update(), 
        rag_context, 
        "Stage 5: Constrained Generation running..."
    )
    detected_list = [obj['label'] for obj in grounding_data]
    final_report = run_vlm_generation(
        z3_facts, 
        rag_context, 
        critical_question, 
        detected_list, 
        image_path, 
        location_mode=(category == "RELATIVE_LOCATION"), 
        counting_target=target_noun_re
    )
    clear_memory()
    
    # Final Output
    yield (
        gr.update(), 
        gr.update(), 
        gr.update(), 
        final_report
    )

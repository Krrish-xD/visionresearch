import requests
import base64
import os
from requests.exceptions import ConnectionError
from .memory_utils import clear_memory

def run_vlm_generation(z3_facts, rag_context, critical_question, detected_list=None, image_path=None, direct_vlm=False, mask_count=None, counting_target=None, presence_mode=False):
    """
    Stage 5: Constrained Generation (VLM)
    Uses local Ollama API to answer the user's question based on proven facts.
    """
    print("[Stage 5] Contacting Ollama API...")
    
    if detected_list is None:
        detected_list = []
        
    z3_empty = not z3_facts or "No spatial relationships proven" in z3_facts
    rag_empty = not rag_context or "No relevant spatial context found" in rag_context
    
    if presence_mode:
        if mask_count > 0:
            prompt = f"Mathematical detection PROVED the existence of {counting_target}. Do not guess. Confirm you see the {counting_target}.\nQuestion: {critical_question}"
        else:
            prompt = f"Mathematical detection found NO instances of {counting_target}. State that it is not visible.\nQuestion: {critical_question}"
    elif mask_count is not None:
        if mask_count == 0:
            prompt = f"Mathematical detection failed to find {counting_target}. Attempt best-effort visual analysis and mark as UNVERIFIED.\nQuestion: {critical_question}"
        else:
            prompt = f"""You are a precision visual analyst. 
Mathematical detection found exactly {mask_count} instances of "{counting_target}" in the image.
Do not guess. Do not look for more. The mathematically verified count is {mask_count}.
Answer the user's question directly based on this fact.
Question: {critical_question}"""
    elif direct_vlm:
        prompt = f"Answer the user's question based on the image. Question: {critical_question}"
    elif z3_empty and rag_empty:
        prompt = f"WARNING: Mathematical verification unavailable. Detected objects: {detected_list}. Question: {critical_question}. Provide best-effort visual analysis and explicitly mark this answer as UNVERIFIED."
    else:
        prompt = f"""You are a precision visual analyst. 
Here are mathematically proven facts: 
{z3_facts}

Here is spatial context: 
{rag_context}

Note: Spatial orientation was mathematically verified by calculating the directional vector from the subject's eyes to its nose (Snout-Vector), ensuring accurate gaze direction regardless of body pose.
Answer the user's critical question using ONLY these facts. Do not hallucinate.

Critical Question: {critical_question}
"""

    url = "http://localhost:11434/api/generate"
    payload = {
        # Assuming the user pulls one of these. We use a standard name.
        "model": "qwen2.5vl:3b",
        "prompt": prompt,
        "stream": False
    }
    
    if image_path and os.path.exists(image_path):
        from PIL import Image
        import io
        try:
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.thumbnail((1024, 1024))
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                payload["images"] = [img_b64]
        except Exception as e:
            print(f"[Stage 5] Warning: Failed to process image for VLM: {e}")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        report = data.get("response", "No response generated.")
        print("[Stage 5] Report generated successfully.")
    except ConnectionError:
        print("[Stage 5] ERROR: Ollama server not running.")
        report = "⚠️ **Error:** Ollama server not running. Please run `ollama serve` in your terminal."
    except Exception as e:
        print(f"[Stage 5] ERROR: {e}")
        report = f"⚠️ **Error during generation:** {str(e)}"
        
    clear_memory()
    return report

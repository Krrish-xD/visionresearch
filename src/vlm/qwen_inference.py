"""
Qwen2.5-VL-specific inference handler.

Qwen2.5-VL uses the Qwen2VLProcessor and Qwen2VLForConditionalGeneration.
Unlike InternVL3, it IS compatible with standard HuggingFace generation
(return_dict_in_generate=True, output_scores=True), but requires:
  1. A specific chat template via apply_chat_template with vision content dict
  2. Image resizing to a multiple of 28 pixels (Qwen2-VL spec)
  3. min_pixels / max_pixels params to the processor for token budget control
"""

import math
import torch
from typing import Optional, Dict, Any
from PIL import Image


# Qwen2.5-VL image preprocessing: resize to multiple of 28, cap total pixels
QWEN_MIN_PIXELS = 256 * 28 * 28       # ~200k pixels
QWEN_MAX_PIXELS = 1280 * 28 * 28      # ~1M pixels


def _resize_to_multiple_of_28(pil_image: Image.Image) -> Image.Image:
    """Resize so both dimensions are multiples of 28 (Qwen2-VL requirement)."""
    w, h = pil_image.size
    new_w = max(28, round(w / 28) * 28)
    new_h = max(28, round(h / 28) * 28)

    # Ensure total pixels within budget
    total = new_w * new_h
    if total > QWEN_MAX_PIXELS:
        scale = math.sqrt(QWEN_MAX_PIXELS / total)
        new_w = max(28, round(w * scale / 28) * 28)
        new_h = max(28, round(h * scale / 28) * 28)

    if (new_w, new_h) == (w, h):
        return pil_image
    return pil_image.resize((new_w, new_h), Image.BICUBIC)


def qwen_generate_with_logprobs(
    model,
    processor,
    pil_image: Optional[Image.Image],
    prompt_text: str,
    max_new_tokens: int = 16,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run Qwen2.5-VL inference and return logprobs.

    Returns dict with keys matching engine.generate_with_logprobs():
        full_text, tokens, outputs, prompt_len, generated_ids
    """
    # Build messages in Qwen2-VL chat format
    if pil_image is not None:
        pil_image = _resize_to_multiple_of_28(pil_image)
        content = [
            {"type": "image", "image": pil_image},
            {"type": "text",  "text": prompt_text},
        ]
    else:
        content = [{"type": "text", "text": prompt_text}]

    messages = [{"role": "user", "content": content}]

    # Apply chat template to get formatted text
    try:
        formatted_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for older processor versions
        formatted_text = (
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
            if pil_image is not None
            else f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        )

    # Tokenize with image
    try:
        if pil_image is not None:
            inputs = processor(
                text=[formatted_text],
                images=[pil_image],
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(
                text=[formatted_text],
                padding=True,
                return_tensors="pt",
            )
    except Exception as e:
        # Some processor versions need different kwargs
        try:
            if pil_image is not None:
                inputs = processor(
                    text=formatted_text,
                    images=pil_image,
                    return_tensors="pt",
                )
            else:
                inputs = processor(
                    text=formatted_text,
                    return_tensors="pt",
                )
        except Exception:
            raise RuntimeError(f"Qwen processor failed: {e}")

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    # Generate with scores
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode new tokens only
    generated_ids     = outputs.sequences                       # [1, prompt+new]
    new_token_ids     = outputs.sequences[0, prompt_len:]       # [new_tokens]
    full_text = processor.decode(new_token_ids, skip_special_tokens=True).strip()

    # Strip common chat separators
    for sep in ["<|im_end|>", "</s>", "<|endoftext|>"]:
        if sep in full_text:
            full_text = full_text.split(sep)[0].strip()

    # Per-token logprobs
    tokens = []
    if outputs.scores:
        for i, step_logits_batch in enumerate(outputs.scores):
            if i >= len(new_token_ids):
                break
            step_logits = step_logits_batch[0]
            step_lprobs = torch.log_softmax(step_logits, dim=-1)
            tok_id      = new_token_ids[i].item()
            tok_logprob = step_lprobs[tok_id].item()
            try:
                tok_text = processor.decode([tok_id])
            except Exception:
                tok_text = str(tok_id)
            tokens.append({
                "token_id":    tok_id,
                "text":        tok_text,
                "logprob":     tok_logprob,
                "prob_percent": math.exp(tok_logprob) * 100.0,
            })

    return {
        "full_text":     full_text,
        "tokens":        tokens,
        "outputs":       outputs,
        "prompt_len":    prompt_len,
        "generated_ids": generated_ids,
    }

"""
InternVL3-specific inference handler.

InternVL3 is NOT a standard HuggingFace generate() model.
It uses a custom generate() wrapper that:
  1. Takes preprocessed pixel_values tensors (448x448, InternVL normalisation)
  2. Builds <IMG_CONTEXT> token strings via its own tokenizer
  3. Its wrapper does NOT support return_dict_in_generate / output_scores

To extract token logprobs we:
  - Build the full prompt string with <img><IMG_CONTEXT>*N</img> manually
  - Embed vision features ourselves via model.extract_feature()
  - Call model.language_model.generate() directly with inputs_embeds,
    return_dict_in_generate=True, output_scores=True
"""

import math
import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

# InternVL3 image preprocessing constants
INTERNVL_IMAGENET_MEAN = (0.485, 0.456, 0.406)
INTERNVL_IMAGENET_STD  = (0.229, 0.224, 0.225)
INTERNVL_IMAGE_SIZE    = 448   # force_image_size from config
INTERNVL_MAX_TILES     = 6     # max dynamic tiles (keeps VRAM manageable)


def _build_transform(image_size: int = INTERNVL_IMAGE_SIZE):
    """Build InternVL3's exact image preprocessing transform."""
    try:
        from torchvision import transforms
    except ImportError:
        raise ImportError("torchvision is required for InternVL3 inference. pip install torchvision")
    return transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=INTERNVL_IMAGENET_MEAN, std=INTERNVL_IMAGENET_STD),
    ])


def _dynamic_preprocess(pil_image, image_size: int = INTERNVL_IMAGE_SIZE, max_num: int = INTERNVL_MAX_TILES):
    """
    Tile a PIL image into up to max_num tiles of size image_size x image_size.
    Always returns at least 1 tile (the whole image resized).
    Returns (pixel_values_tensor [N, 3, H, W], num_patches int).
    """
    transform = _build_transform(image_size)

    orig_w, orig_h = pil_image.size
    aspect = orig_w / orig_h

    # Decide tile grid
    best_num = 1
    best_ratio = float("inf")
    for n in range(1, max_num + 1):
        # Try n tiles in a roughly square arrangement
        rows = max(1, round(math.sqrt(n / aspect)))
        cols = math.ceil(n / rows)
        ratio = abs(cols / rows - aspect)
        if ratio < best_ratio:
            best_ratio = ratio
            best_num = n
            best_rows = rows
            best_cols = cols

    # Resize image to fill the grid
    target_w = best_cols * image_size
    target_h = best_rows * image_size
    try:
        from PIL import Image
        resized = pil_image.resize((target_w, target_h), Image.BICUBIC)
    except Exception:
        resized = pil_image

    tiles = []
    for row in range(best_rows):
        for col in range(best_cols):
            left   = col * image_size
            upper  = row * image_size
            right  = left + image_size
            lower  = upper + image_size
            try:
                tile = resized.crop((left, upper, right, lower))
            except Exception:
                tile = resized
            tiles.append(transform(tile))

    # Also include the global thumbnail
    try:
        from PIL import Image
        thumb = pil_image.resize((image_size, image_size), Image.BICUBIC)
    except Exception:
        thumb = pil_image
    tiles.append(transform(thumb))

    pixel_values = torch.stack(tiles, dim=0)   # [N, 3, H, W]
    return pixel_values, len(tiles)


def internvl_generate_with_logprobs(
    model,
    tokenizer,
    pil_image,
    prompt_text: str,
    max_new_tokens: int = 16,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run InternVL3 inference and return logprobs.

    Returns dict with keys matching engine.generate_with_logprobs():
        full_text, tokens, outputs, prompt_len, generated_ids
    """
    IMG_START   = "<img>"
    IMG_END     = "</img>"
    IMG_CONTEXT = "<IMG_CONTEXT>"

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT)
    model.img_context_token_id = img_context_token_id

    # -- Image preprocessing --
    if pil_image is not None:
        pixel_values, num_patches = _dynamic_preprocess(pil_image)
        pixel_values = pixel_values.to(device=model.device,
                                        dtype=model.language_model.dtype
                                        if hasattr(model, "language_model") else torch.bfloat16)
    else:
        pixel_values = None
        num_patches  = 0

    # -- Build prompt string with image tokens --
    if pixel_values is not None:
        image_tokens = (IMG_START
                        + IMG_CONTEXT * model.num_image_token * num_patches
                        + IMG_END)
        question = f"<image>\n{prompt_text}"
        question_with_img = image_tokens + "\n" + prompt_text
    else:
        question_with_img = prompt_text

    # -- Build conversation template --
    try:
        from .internvl_inference import _build_internvl_query
        query = _build_internvl_query(model, tokenizer, question_with_img,
                                      pixel_values is not None, num_patches,
                                      IMG_START, IMG_END, IMG_CONTEXT)
    except Exception:
        # Fallback: use the conversation template from the model files if available
        try:
            import importlib.util, sys, os
            model_dir = getattr(model.config, "_name_or_path", "")
            if not os.path.isdir(model_dir):
                # Try to find the conversation module from the loaded model
                conv_mod = sys.modules.get("conversation", None)
                if conv_mod is None:
                    raise ImportError("conversation module not found")
                get_conv = conv_mod.get_conv_template
            else:
                spec = importlib.util.spec_from_file_location(
                    "conversation", os.path.join(model_dir, "conversation.py"))
                conv_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(conv_mod)
                get_conv = conv_mod.get_conv_template

            template_name = getattr(model.config, "template", "internvl2_5")
            template = get_conv(template_name)
            template.system_message = model.system_message
            eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

            template.append_message(template.roles[0], question_with_img)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
        except Exception:
            # Last resort plain format
            query = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                     f"<|im_start|>user\n{question_with_img}<|im_end|>\n"
                     f"<|im_start|>assistant\n")
            eos_token_id = tokenizer.eos_token_id

    # -- Tokenize --
    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids    = model_inputs["input_ids"].to(model.device)
    attn_mask    = model_inputs["attention_mask"].to(model.device)

    # -- Build inputs_embeds by fusing vision features --
    if pixel_values is not None:
        with torch.no_grad():
            vit_embeds = model.extract_feature(pixel_values)   # [num_patches, tokens, C]
        input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
        B, N, C = input_embeds.shape
        flat_embeds = input_embeds.reshape(B * N, C)
        flat_ids    = input_ids.reshape(B * N)
        selected    = (flat_ids == img_context_token_id)
        if selected.sum() > 0:
            vit_flat = vit_embeds.reshape(-1, C).to(flat_embeds.dtype)
            n_img_tokens = min(selected.sum(), vit_flat.shape[0])
            flat_embeds[selected][:n_img_tokens] = vit_flat[:n_img_tokens]
        input_embeds = flat_embeds.reshape(B, N, C)
    else:
        with torch.no_grad():
            input_embeds = model.language_model.get_input_embeddings()(input_ids)

    prompt_len = input_ids.shape[1]

    # -- Generate via language_model.generate() to get scores --
    gen_kwargs = {
        "inputs_embeds": input_embeds,
        "attention_mask": attn_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
        "use_cache": True,
    }
    # Set eos if we have it
    try:
        gen_kwargs["eos_token_id"] = eos_token_id
    except NameError:
        pass

    with torch.no_grad():
        outputs = model.language_model.generate(**gen_kwargs)

    # Decode — outputs.sequences has shape [1, new_tokens] since we used inputs_embeds
    generated_ids = outputs.sequences   # [1, new_tokens]
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    # Strip any residual template separators
    for sep in ["<|im_end|>", "</s>", "<|endoftext|>"]:
        if sep in full_text:
            full_text = full_text.split(sep)[0].strip()

    # -- Extract per-token logprobs --
    tokens = []
    if outputs.scores:
        for i, step_logits_batch in enumerate(outputs.scores):
            step_logits  = step_logits_batch[0]
            step_lprobs  = torch.log_softmax(step_logits, dim=-1)
            tok_id       = generated_ids[0, i].item()
            tok_logprob  = step_lprobs[tok_id].item()
            tok_text     = tokenizer.decode([tok_id])
            tokens.append({
                "token_id":    tok_id,
                "text":        tok_text,
                "logprob":     tok_logprob,
                "prob_percent": math.exp(tok_logprob) * 100.0,
            })

    # Build a compatible fake sequences tensor with prompt prepended
    # (for compatibility with extract_token_confidence which uses prompt_len offset)
    dummy_prompt = torch.zeros(1, prompt_len, dtype=torch.long, device=model.device)
    full_sequences = torch.cat([dummy_prompt, generated_ids], dim=1)

    return {
        "full_text":      full_text,
        "tokens":         tokens,
        "outputs":        outputs,
        "prompt_len":     prompt_len,
        "generated_ids":  full_sequences,
    }


def _build_internvl_query(model, tokenizer, question_with_img, has_image,
                           num_patches, IMG_START, IMG_END, IMG_CONTEXT):
    """Build the full query string using the model's conversation template."""
    # Try to import conversation from wherever it was loaded
    import sys
    # The conversation module is usually registered under the model's module name
    conv_cls = None
    for mod_name, mod in sys.modules.items():
        if "conversation" in mod_name and hasattr(mod, "get_conv_template"):
            conv_cls = mod
            break

    if conv_cls is None:
        raise ImportError("Could not find conversation module")

    template_name = getattr(model.config, "template", "internvl2_5")
    template = conv_cls.get_conv_template(template_name)
    template.system_message = getattr(model, "system_message", "")

    template.append_message(template.roles[0], question_with_img)
    template.append_message(template.roles[1], None)
    return template.get_prompt()

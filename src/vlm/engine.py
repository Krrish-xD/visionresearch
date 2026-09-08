"""Core VLM Inference Engine for model loading and generation.

Supports three model families with dedicated handlers:
  - llava:    Standard HuggingFace LlavaForConditionalGeneration
  - internvl: InternVL3 custom pipeline (src/vlm/internvl_inference.py)
  - qwen:     Qwen2.5-VL pipeline (src/vlm/qwen_inference.py)
"""

import os
import math
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModel

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None

try:
    from transformers import LlavaForConditionalGeneration
except ImportError:
    LlavaForConditionalGeneration = None

try:
    from transformers import LlavaNextForConditionalGeneration
except ImportError:
    LlavaNextForConditionalGeneration = None

try:
    from transformers import LlavaOnevisionForConditionalGeneration
except ImportError:
    LlavaOnevisionForConditionalGeneration = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None
    Qwen2VLForConditionalGeneration = None

try:
    from transformers import Qwen2_5_VLProcessor, Qwen2VLProcessor
except ImportError:
    Qwen2_5_VLProcessor = None
    Qwen2VLProcessor = None


def _detect_family(model_id_or_path: str) -> str:
    """Return 'llava', 'internvl', 'qwen', or 'generic'."""
    lower = model_id_or_path.lower()
    if "internvl" in lower or "intern_vl" in lower:
        return "internvl"
    if "qwen" in lower or "qwen2" in lower:
        return "qwen"
    if "llava" in lower:
        return "llava"
    return "generic"


class VLMEngine:
    """Core PyTorch/HuggingFace model loading and inference logic.

    Routes generation to family-specific handlers so that InternVL3 and
    Qwen2.5-VL both produce valid token logprobs.
    """

    def __init__(self, weights_dir: str = "../weights"):
        self.weights_dir = weights_dir
        self.processor = None   # AutoProcessor / AutoTokenizer / Qwen2VLProcessor
        self.model = None
        self.current_model_id = None
        self.model_family = "generic"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_available_models(self):
        """Scans the weights directory purely for available local models."""
        models = []
        if os.path.exists(self.weights_dir):
            for item in os.listdir(self.weights_dir):
                full_path = os.path.join(self.weights_dir, item)
                if os.path.isdir(full_path):
                    models.append({"id": item, "path": full_path})
        return models

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def unload_model(self):
        """Safely unloads current model and processor, resetting state."""
        if getattr(self, "model", None) is not None:
            try:
                del self.model
            except AttributeError:
                pass
        if getattr(self, "processor", None) is not None:
            try:
                del self.processor
            except AttributeError:
                pass
        self.model = None
        self.processor = None
        self.current_model_id = None
        self.model_family = "generic"
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, model_id_or_path: str, load_in_4bit: bool = True,
                   trust_remote_code: bool = True):
        """Loads a model dynamically, unloading the previous one if necessary."""
        if self.current_model_id == model_id_or_path and self.model is not None:
            return  # Already loaded
        print(f"[Engine] Loading model: {model_id_or_path}")
        self.unload_model()

        family = _detect_family(model_id_or_path)
        self.model_family = family
        print(f"[Engine] Detected family: {family}")

        try:
            if family == "internvl":
                self._load_internvl(model_id_or_path, load_in_4bit, trust_remote_code)
            elif family == "qwen":
                self._load_qwen(model_id_or_path, load_in_4bit, trust_remote_code)
            else:
                self._load_generic(model_id_or_path, load_in_4bit, trust_remote_code, family)

            if self.model is not None:
                self.model.eval()
            self.current_model_id = model_id_or_path
            print(f"[Engine] Model {model_id_or_path} loaded successfully.")

        except Exception:
            self.unload_model()
            raise

    # ------------------------------------------------------------------
    # Family-specific loaders
    # ------------------------------------------------------------------

    def _build_bnb_config(self, family: str = "generic"):
        try:
            from transformers import BitsAndBytesConfig
            kwargs = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            }
            if family == "qwen":
                kwargs["llm_int8_skip_modules"] = ["visual"]
            return BitsAndBytesConfig(**kwargs)
        except ImportError:
            return None

    def _load_internvl(self, path: str, load_in_4bit: bool, trust_remote_code: bool):
        """Load InternVL3 model + tokenizer (not AutoProcessor)."""
        # InternVL3 uses its own tokenizer, not AutoProcessor
        self.processor = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )

        kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto" if self.device == "cuda" else None,
            "trust_remote_code": trust_remote_code,
        }
        if load_in_4bit and self.device == "cuda":
            bnb = self._build_bnb_config(family="internvl")
            if bnb:
                kwargs["quantization_config"] = bnb

        # InternVL3 registers itself as AutoModelForCausalLM via auto_map
        self.model = AutoModelForCausalLM.from_pretrained(path, **kwargs)

        # Ensure img_context_token_id is set
        img_ctx_id = self.processor.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = img_ctx_id

    def _load_qwen(self, path: str, load_in_4bit: bool, trust_remote_code: bool):
        """Load Qwen2.5-VL model + processor."""
        kwargs_proc = {"trust_remote_code": trust_remote_code}

        # Try Qwen2_5_VLProcessor / Qwen2VLProcessor first, fall back to AutoProcessor
        if Qwen2_5_VLProcessor is not None:
            try:
                self.processor = Qwen2_5_VLProcessor.from_pretrained(path, **kwargs_proc)
            except Exception:
                self.processor = AutoProcessor.from_pretrained(path, **kwargs_proc)
        elif Qwen2VLProcessor is not None:
            try:
                self.processor = Qwen2VLProcessor.from_pretrained(path, **kwargs_proc)
            except Exception:
                self.processor = AutoProcessor.from_pretrained(path, **kwargs_proc)
        else:
            self.processor = AutoProcessor.from_pretrained(path, **kwargs_proc)

        kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto" if self.device == "cuda" else None,
            "trust_remote_code": trust_remote_code,
        }
        if load_in_4bit and self.device == "cuda":
            bnb = self._build_bnb_config(family="qwen")
            if bnb:
                kwargs["quantization_config"] = bnb

        if Qwen2_5_VLForConditionalGeneration is not None:
            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, **kwargs)
                return
            except Exception:
                pass

        if Qwen2VLForConditionalGeneration is not None:
            try:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(path, **kwargs)
                return
            except Exception:
                pass

        self.model = AutoModelForCausalLM.from_pretrained(path, **kwargs)

    def _load_generic(self, path: str, load_in_4bit: bool, trust_remote_code: bool,
                      family: str):
        """Load LLaVA-family or generic vision-language model."""
        try:
            self.processor = AutoProcessor.from_pretrained(
                path, trust_remote_code=trust_remote_code
            )
        except Exception:
            self.processor = AutoTokenizer.from_pretrained(
                path, trust_remote_code=trust_remote_code
            )

        kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "device_map": "auto" if self.device == "cuda" else None,
            "trust_remote_code": trust_remote_code,
        }
        if load_in_4bit and self.device == "cuda":
            bnb = self._build_bnb_config(family=family)
            if bnb:
                kwargs["quantization_config"] = bnb

        # Try specific LLaVA classes first
        model_lower = path.lower()
        if "onevision" in model_lower and LlavaOnevisionForConditionalGeneration is not None:
            try:
                self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(path, **kwargs)
                return
            except Exception:
                pass

        if ("next" in model_lower or "v1.6" in model_lower) and LlavaNextForConditionalGeneration is not None:
            try:
                self.model = LlavaNextForConditionalGeneration.from_pretrained(path, **kwargs)
                return
            except Exception:
                pass

        # Try LLaVA class
        if family == "llava" and LlavaForConditionalGeneration is not None:
            try:
                self.model = LlavaForConditionalGeneration.from_pretrained(path, **kwargs)
                return
            except Exception:
                pass

        # Try AutoModelForVision2Seq
        if AutoModelForVision2Seq is not None:
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(path, **kwargs)
                return
            except Exception:
                pass

        # Try AutoModelForCausalLM
        try:
            self.model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
            return
        except Exception:
            pass

        # Fallback to AutoModel
        self.model = AutoModel.from_pretrained(path, **kwargs)
        if hasattr(self.model, "img_context_token_id") and getattr(self.model, "img_context_token_id", None) is None:
            self.model.img_context_token_id = 151667

    # ------------------------------------------------------------------
    # Generation — routes to family handler
    # ------------------------------------------------------------------

    def generate_with_logprobs(self, pil_image, prompt_text: str,
                               temperature: float = 0.0, top_p: float = 1.0,
                               top_k: int = 50, max_tokens: int = 16):
        """Runs inference and extracts exact token logprobs.

        Routes to the correct family handler; all return the same dict schema:
          full_text, tokens, outputs, prompt_len, generated_ids
        """
        if self.model is None:
            raise RuntimeError("No model is currently loaded.")

        if self.model_family == "internvl":
            return self._generate_internvl(pil_image, prompt_text, max_tokens)
        elif self.model_family == "qwen":
            return self._generate_qwen(pil_image, prompt_text, max_tokens)
        else:
            return self._generate_generic(pil_image, prompt_text, temperature,
                                          top_p, top_k, max_tokens)

    def _generate_internvl(self, pil_image, prompt_text: str, max_tokens: int):
        from src.vlm.internvl_inference import internvl_generate_with_logprobs
        return internvl_generate_with_logprobs(
            model=self.model,
            tokenizer=self.processor,
            pil_image=pil_image,
            prompt_text=prompt_text,
            max_new_tokens=max_tokens,
            device=self.device,
        )

    def _generate_qwen(self, pil_image, prompt_text: str, max_tokens: int):
        from src.vlm.qwen_inference import qwen_generate_with_logprobs
        return qwen_generate_with_logprobs(
            model=self.model,
            processor=self.processor,
            pil_image=pil_image,
            prompt_text=prompt_text,
            max_new_tokens=max_tokens,
            device=self.device,
        )

    def _generate_generic(self, pil_image, prompt_text: str, temperature: float,
                          top_p: float, top_k: int, max_tokens: int):
        """Standard HuggingFace generate for LLaVA and generic models."""
        if hasattr(self.processor, "apply_chat_template"):
            messages = [{
                "role": "user",
                "content": [
                    *([{"type": "image"}] if pil_image else []),
                    {"type": "text", "text": prompt_text},
                ],
            }]
            try:
                formatted = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                if pil_image:
                    inputs = self.processor(text=[formatted], images=[pil_image],
                                            padding=True, return_tensors="pt")
                else:
                    inputs = self.processor(text=[formatted], padding=True,
                                            return_tensors="pt")
            except Exception:
                formatted = (f"USER: <image>\n{prompt_text}\nASSISTANT:"
                             if pil_image else f"USER: {prompt_text}\nASSISTANT:")
                if pil_image:
                    inputs = self.processor(text=formatted, images=pil_image,
                                            return_tensors="pt")
                else:
                    inputs = self.processor(text=formatted, return_tensors="pt")
        else:
            formatted = (f"USER: <image>\n{prompt_text}\nASSISTANT:"
                         if pil_image else f"USER: {prompt_text}\nASSISTANT:")
            if pil_image:
                inputs = self.processor(text=formatted, images=pil_image,
                                        return_tensors="pt")
            else:
                inputs = self.processor(text=formatted, return_tensors="pt")

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if temperature > 0.0:
            gen_kwargs.update({"do_sample": True, "temperature": temperature,
                               "top_p": top_p, "top_k": top_k})
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated_sequence = outputs.sequences[0][input_len:]
        full_text = self.processor.decode(generated_sequence,
                                          skip_special_tokens=True).strip()

        tokens = []
        for i, token_id_tensor in enumerate(generated_sequence):
            token_id    = token_id_tensor.item()
            step_logits = outputs.scores[i][0]
            step_lprobs = torch.nn.functional.log_softmax(step_logits, dim=-1)
            tok_logprob = step_lprobs[token_id].item()

            if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
                tok_text = self.processor.tokenizer.decode([token_id])
            else:
                tok_text = str(token_id)

            tokens.append({
                "token_id":    token_id,
                "text":        tok_text,
                "logprob":     tok_logprob,
                "prob_percent": math.exp(tok_logprob) * 100.0,
            })

        return {
            "full_text":     full_text,
            "tokens":        tokens,
            "outputs":       outputs,
            "prompt_len":    input_len,
            "generated_ids": outputs.sequences,
        }

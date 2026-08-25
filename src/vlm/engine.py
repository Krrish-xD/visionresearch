"""Core VLM Inference Engine for model loading and generation."""

import os
import math
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None

try:
    from transformers import LlavaForConditionalGeneration
except ImportError:
    LlavaForConditionalGeneration = None

try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None

class VLMEngine:
    """Core PyTorch/HuggingFace model loading and inference logic."""
    def __init__(self, weights_dir: str = "../weights"):
        self.weights_dir = weights_dir
        self.processor = None
        self.model = None
        self.current_model_id = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def list_available_models(self):
        """Scans the weights directory purely for available local models."""
        models = []
        if os.path.exists(self.weights_dir):
            for item in os.listdir(self.weights_dir):
                full_path = os.path.join(self.weights_dir, item)
                if os.path.isdir(full_path):
                    models.append({"id": item, "path": full_path})
        return models

    def load_model(self, model_id_or_path: str, load_in_4bit: bool = True, trust_remote_code: bool = True):
        """Loads a model dynamically, unloading the previous one if necessary."""
        if self.current_model_id == model_id_or_path and self.model is not None:
            return  # Already loaded
            
        print(f"Loading model {model_id_or_path}...")
        
        # Unload previous model to free VRAM
        if self.model is not None:
            del self.model
            del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        self.processor = AutoProcessor.from_pretrained(
            model_id_or_path, 
            trust_remote_code=trust_remote_code
        )
        
        # Routing heuristic
        model_lower = model_id_or_path.lower()
        if "llava" in model_lower and LlavaForConditionalGeneration:
            model_class = LlavaForConditionalGeneration
        elif "qwen" in model_lower and Qwen2VLForConditionalGeneration:
            model_class = Qwen2VLForConditionalGeneration
        else:
            model_class = AutoModelForCausalLM
            
        kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "device_map": "auto" if self.device == "cuda" else None,
            "trust_remote_code": trust_remote_code
        }
        
        if load_in_4bit and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            except ImportError:
                pass

        try:
            self.model = model_class.from_pretrained(model_id_or_path, **kwargs)
        except Exception:
            try:
                if AutoModelForVision2Seq is not None:
                    self.model = AutoModelForVision2Seq.from_pretrained(model_id_or_path, **kwargs)
                else:
                    raise RuntimeError("AutoModelForVision2Seq not available")
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)
            
        self.model.eval()
        self.current_model_id = model_id_or_path
        print(f"Model {model_id_or_path} loaded successfully.")

    def generate_with_logprobs(self, pil_image, prompt_text: str, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 50, max_tokens: int = 100):
        """Runs inference and extracts exact token logprobs."""
        if self.model is None:
            raise RuntimeError("No model is currently loaded.")

        if hasattr(self.processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"} if pil_image else None,
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            messages[0]["content"] = [c for c in messages[0]["content"] if c is not None]
            
            try:
                formatted_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if pil_image:
                    inputs = self.processor(text=[formatted_prompt], images=[pil_image], padding=True, return_tensors="pt").to(self.model.device)
                else:
                    inputs = self.processor(text=[formatted_prompt], padding=True, return_tensors="pt").to(self.model.device)
            except Exception:
                formatted_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:" if pil_image else f"USER: {prompt_text}\nASSISTANT:"
                if pil_image:
                    inputs = self.processor(text=formatted_prompt, images=pil_image, return_tensors="pt").to(self.model.device)
                else:
                    inputs = self.processor(text=formatted_prompt, return_tensors="pt").to(self.model.device)
        else:
            formatted_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:" if pil_image else f"USER: {prompt_text}\nASSISTANT:"
            if pil_image:
                inputs = self.processor(text=formatted_prompt, images=pil_image, return_tensors="pt").to(self.model.device)
            else:
                inputs = self.processor(text=formatted_prompt, return_tensors="pt").to(self.model.device)
        
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k
        else:
            gen_kwargs["do_sample"] = False
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs.input_ids.shape[1] if inputs.input_ids is not None else 0
        generated_sequence = outputs.sequences[0][input_len:] 
        full_text = self.processor.decode(generated_sequence, skip_special_tokens=True).strip()

        tokens = []
        logits_tuple = outputs.scores

        for i, token_id_tensor in enumerate(generated_sequence):
            token_id = token_id_tensor.item()
            step_logits = logits_tuple[i][0]
            step_logprobs = torch.nn.functional.log_softmax(step_logits, dim=-1)
            token_logprob = step_logprobs[token_id].item()
            token_prob_percent = math.exp(token_logprob) * 100.0

            if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
                token_text = self.processor.tokenizer.decode([token_id])
            else:
                token_text = str(token_id)
            
            tokens.append({
                "token_id": token_id,
                "text": token_text,
                "logprob": token_logprob,
                "prob_percent": token_prob_percent
            })

        return {
            "full_text": full_text,
            "tokens": tokens,
            "outputs": outputs,
            "prompt_len": input_len,
            "generated_ids": outputs.sequences
        }

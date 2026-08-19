import os
import torch
import math
from transformers import AutoProcessor, AutoModelForCausalLM
try:
    from transformers import LlavaForConditionalGeneration
except ImportError:
    LlavaForConditionalGeneration = None

try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None

class VLMEngine:
    def __init__(self, weights_dir: str = "../weights"):
        self.weights_dir = weights_dir
        self.processor = None
        self.model = None
        self.current_model_id = None

    def list_available_models(self):
        """Scans the weights directory purely for available local models."""
        models = []
        if os.path.exists(self.weights_dir):
            for item in os.listdir(self.weights_dir):
                full_path = os.path.join(self.weights_dir, item)
                if os.path.isdir(full_path):
                    models.append({"id": item, "path": full_path})
            
        return models

    def load_model(self, model_id_or_path: str):
        """Loads a model dynamically, unloading the previous one if necessary."""
        if self.current_model_id == model_id_or_path and self.model is not None:
            return # Already loaded
            
        print(f"Loading model {model_id_or_path}...")
        
        # Unload previous model to free VRAM
        if self.model is not None:
            del self.model
            del self.processor
            torch.cuda.empty_cache()
            
        self.processor = AutoProcessor.from_pretrained(
            model_id_or_path, 
            trust_remote_code=True
        )
        
        # Simple heuristic to determine model class (LLaVA vs standard causal LM)
        model_lower = model_id_or_path.lower()
        if "llava" in model_lower and LlavaForConditionalGeneration:
            model_class = LlavaForConditionalGeneration
        elif "qwen" in model_lower and Qwen2VLForConditionalGeneration:
            model_class = Qwen2VLForConditionalGeneration
        else:
            model_class = AutoModelForCausalLM
            
        kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        try:
            import bitsandbytes
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        except ImportError:
            pass # load in standard fp16 if bitsandbytes is not installed

        self.model = model_class.from_pretrained(
            model_id_or_path, 
            **kwargs
        )
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
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            try:
                formatted_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[formatted_prompt], images=[pil_image], padding=True, return_tensors="pt").to(self.model.device)
            except Exception:
                # Fallback if chat template fails
                formatted_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
                inputs = self.processor(text=formatted_prompt, images=pil_image, return_tensors="pt").to(self.model.device)
        else:
            # Naive prompt formatting
            formatted_prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
            inputs = self.processor(text=formatted_prompt, images=pil_image, return_tensors="pt").to(self.model.device)
        
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
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )

        generated_sequence = outputs.sequences[0][inputs.input_ids.shape[1]:] 
        full_text = self.processor.decode(generated_sequence, skip_special_tokens=True)

        tokens = []
        logits_tuple = outputs.scores

        for i, token_id_tensor in enumerate(generated_sequence):
            token_id = token_id_tensor.item()
            step_logits = logits_tuple[i][0]
            step_logprobs = torch.nn.functional.log_softmax(step_logits, dim=-1)
            token_logprob = step_logprobs[token_id].item()
            token_prob_percent = math.exp(token_logprob) * 100.0

            # Decode the token ID properly to resolve BPE symbols like 'Ġ' into normal spaces
            token_text = self.processor.tokenizer.decode([token_id])
            
            tokens.append({
                "token_id": token_id,
                "text": token_text,
                "logprob": token_logprob,
                "prob_percent": token_prob_percent
            })

        return {
            "full_text": full_text,
            "tokens": tokens
        }

# Global singleton instance for the backend to use
vlm_engine = VLMEngine()

# VisionResearch Model Downloads & VRAM Guide

This document lists the latest generation Vision-Language Models (VLMs) optimized for a **16 GB VRAM GPU budget** (e.g. NVIDIA RTX 2000 Ada Generation 16 GB / RTX 4080 16 GB).

---

## 💾 16 GB VRAM Sizing & Compatibility Chart

| Model Key | Hugging Face ID | Params | VRAM (4-bit NF4) | VRAM (FP16/BF16) | Best Use Case |
|---|---|---|---|---|---|
| **`internvl3-8b`** | `OpenGVLab/InternVL3-8B` | 8B | ~5.5 GB | ~16.0 GB | **Latest Flagship InternVL**: SOTA visual grounding & reasoning |
| **`llava-onevision-7b`** | `lmms-lab/llava-onevision-qwen2-7b-ov` | 7B | ~5.0 GB | ~14.5 GB | **Latest LLaVA Generation**: SigLIP + Qwen2 unified image/video SOTA |
| **`llava-next-7b`** | `llava-hf/llava-v1.6-mistral-7b-hf` | 7B | ~5.0 GB | ~14.5 GB | **LLaVA-NeXT (v1.6)**: AnyRes dynamic resolution for fine details |
| **`qwen2.5-vl-7b`** | `Qwen/Qwen2.5-VL-7B-Instruct` | 7B | ~5.5 GB | ~15.0 GB | **Latest Qwen2.5-VL**: Native dynamic aspect-ratio support |

---

## 🚀 Quick Download Commands

You can use the unified resource manager or download directly with `huggingface-cli`:

### Download via Unified Resource Manager
```bash
# Download specific model
python -m scripts.setup_resources --model internvl3-8b
python -m scripts.setup_resources --model llava-onevision-7b
python -m scripts.setup_resources --model llava-next-7b
python -m scripts.setup_resources --model qwen2.5-vl-7b

# Or download all configured models at once
python -m scripts.setup_resources --models
```

### Or Direct CLI Download
```bash
# Latest InternVL3 8B (Primary Model)
huggingface-cli download OpenGVLab/InternVL3-8B --local-dir weights/internvl3-8b

# Latest LLaVA-OneVision 7B (SigLIP + Qwen2)
huggingface-cli download lmms-lab/llava-onevision-qwen2-7b-ov --local-dir weights/llava-onevision-7b

# LLaVA-NeXT 7B (v1.6 Mistral)
huggingface-cli download llava-hf/llava-v1.6-mistral-7b-hf --local-dir weights/llava-next-7b

# Qwen2.5-VL 7B
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir weights/qwen2.5-vl-7b
```

---

## ⚙️ Model Loading Notes

- All 7B/8B models load in **4-bit NF4 quantization** by default when `load_in_4bit: true` is set in `configs/models.yaml`, consuming only **4.5 GB to 5.5 GB of VRAM**.
- `InternVL3` requires `trust_remote_code: true` due to custom vision-language transformer layers.
- Once downloaded into `weights/<model_id>`, the Web UI (`python start.py`) will automatically discover and display them in the **Model Library** tab for one-click VRAM loading.

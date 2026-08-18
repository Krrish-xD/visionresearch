# Model Downloads

Install the Python packages first:

```bash
pip install -r requirements.txt
```

Then log in to Hugging Face if needed:

```bash
huggingface-cli login
```

## Primary Models

Start with one model for the pilot:

```bash
huggingface-cli download llava-hf/llava-1.5-7b-hf --local-dir models/llava-1.5-7b-hf
```

Add the second model for the main experiment:

```bash
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir models/qwen2-vl-7b-instruct
```

## Optional Smaller Backup Models

Use these if a teammate has weaker hardware:

```bash
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir models/qwen2-vl-2b-instruct
huggingface-cli download llava-hf/llava-1.5-3b-hf --local-dir models/llava-1.5-3b-hf
```

## Notes

- Use Python 3.10 or 3.11.
- Install PyTorch separately for your CUDA version before `pip install -r requirements.txt`.
- `bitsandbytes` is included for 4-bit or 8-bit loading on supported systems.
- Keep model paths consistent across the team so configs can be shared easily.

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

Start with the primary modern model:

```bash
huggingface-cli download OpenGVLab/InternVL3-8B --local-dir models/internvl3-8b
```

Add one stable baseline:

```bash
huggingface-cli download llava-hf/llava-1.5-7b-hf --local-dir models/llava-1.5-7b-hf
```

Optional stronger baseline if your machine can support it and the code path is stable:

```bash
huggingface-cli download llava-hf/llava-v1.6-mistral-7b-hf --local-dir models/llava-v1.6-mistral-7b-hf
```

## Optional Smaller Backup Models

Use these if a teammate has weaker hardware:

```bash
huggingface-cli download OpenGVLab/InternVL3-1B --local-dir models/internvl3-1b
```

## Notes

- Use Python 3.10 or 3.11.
- Install PyTorch separately for your CUDA version before `pip install -r requirements.txt`.
- `bitsandbytes` is included for 4-bit or 8-bit loading on supported systems.
- `InternVL3` requires `trust_remote_code=True` when loading due to custom model architecture.
- Keep model paths consistent across the team so configs can be shared easily.

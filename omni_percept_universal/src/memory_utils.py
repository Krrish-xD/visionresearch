import gc
import torch

def clear_memory():
    """
    Strictly enforces memory wiping for the Model Carousel pattern.
    Clears Apple Silicon (MPS) cache if available, otherwise runs garbage collection.
    """
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

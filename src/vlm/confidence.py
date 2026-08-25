"""Confidence extraction from Hugging Face model generation logits."""

import torch
import numpy as np
from typing import List, Tuple

def extract_token_confidence(
    scores: Tuple[torch.Tensor, ...],
    generated_ids: torch.Tensor,
    prompt_len: int = 0
) -> Tuple[float, List[float]]:
    """
    Extract token probabilities and length-normalized geometric mean confidence.
    
    Args:
        scores: Tuple of logits tensors for each generated step from model.generate(..., output_scores=True)
        generated_ids: Tensor of full sequence IDs [batch_size, seq_len]
        prompt_len: Length of input prompt tokens
        
    Returns:
        (raw_confidence, token_logprobs)
    """
    if not scores:
        return 1.0, [0.0]

    token_logprobs = []
    
    # Sequence of generated tokens
    gen_tokens = generated_ids[0, prompt_len:] if generated_ids.dim() > 1 else generated_ids[prompt_len:]
    num_steps = min(len(scores), len(gen_tokens))

    if num_steps == 0:
        return 1.0, [0.0]

    for step_idx in range(num_steps):
        step_logits = scores[step_idx][0]  # [vocab_size]
        step_logprobs = torch.log_softmax(step_logits, dim=-1)
        tok_id = gen_tokens[step_idx].item()
        tok_logprob = step_logprobs[tok_id].item()
        token_logprobs.append(tok_logprob)

    # For single-token answers: p = exp(logprob_0)
    # For multi-token answers: p = exp(mean(logprob))
    mean_logprob = float(np.mean(token_logprobs))
    raw_confidence = float(np.exp(mean_logprob))
    # Clip to valid probability bounds [0.0001, 1.0]
    raw_confidence = max(0.0001, min(1.0, raw_confidence))

    return raw_confidence, token_logprobs

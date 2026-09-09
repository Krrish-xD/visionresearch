"""Confidence extraction from Hugging Face model generation logits."""

import torch
import numpy as np
from typing import List, Tuple

def extract_token_confidence(
    scores: Tuple[torch.Tensor, ...],
    generated_ids: torch.Tensor,
    prompt_len: int = 0,
    top_k: int = 10,
    batch_idx: int = 0
) -> Tuple[float, List[float], List[float], List[int]]:
    """
    Extract token probabilities, length-normalized geometric mean confidence,
    and the pre-softmax logit vector for the primary answer token position.
    
    Args:
        scores: Tuple of logits tensors for each generated step from model.generate(..., output_scores=True)
        generated_ids: Tensor of full sequence IDs [batch_size, seq_len]
        prompt_len: Length of input prompt tokens
        top_k: Number of top candidate logits to extract for multinomial temperature scaling
        batch_idx: Batch item index to extract scores for (default: 0)
        
    Returns:
        (raw_confidence, token_logprobs, answer_logits, candidate_token_ids)
    """
    if not scores:
        return 1.0, [0.0], [0.0, -2.0], [0, 1]

    token_logprobs = []
    
    # Sequence of generated tokens for this batch item
    if generated_ids.dim() > 1:
        gen_tokens = generated_ids[batch_idx, prompt_len:] if batch_idx < generated_ids.shape[0] else generated_ids[0, prompt_len:]
    else:
        gen_tokens = generated_ids[prompt_len:]
    num_steps = min(len(scores), len(gen_tokens))

    if num_steps == 0:
        return 1.0, [0.0], [0.0, -2.0], [0, 1]

    for step_idx in range(num_steps):
        step_scores = scores[step_idx]
        step_logits = step_scores[batch_idx] if step_scores.dim() > 1 and batch_idx < step_scores.shape[0] else step_scores[0]
        step_logprobs = torch.log_softmax(step_logits, dim=-1)
        tok_id = gen_tokens[step_idx].item()
        tok_logprob = step_logprobs[tok_id].item()
        token_logprobs.append(tok_logprob)

    # Extract top-k pre-softmax logit vector at the primary answer token (step 0)
    first_step_scores = scores[0]
    primary_step_logits = first_step_scores[batch_idx] if first_step_scores.dim() > 1 and batch_idx < first_step_scores.shape[0] else first_step_scores[0]
    k_val = min(top_k, primary_step_logits.shape[-1])
    topk_vals, topk_indices = torch.topk(primary_step_logits, k=k_val)
    answer_logits = [float(v.item()) for v in topk_vals]
    candidate_token_ids = [int(i.item()) for i in topk_indices]

    # For single-token answers: p = exp(logprob_0)
    # For multi-token answers: p = exp(mean(logprob))
    mean_logprob = float(np.mean(token_logprobs))
    raw_confidence = float(np.exp(mean_logprob))
    raw_confidence = max(0.0001, min(1.0, raw_confidence))

    return raw_confidence, token_logprobs, answer_logits, candidate_token_ids

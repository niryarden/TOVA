import torch
from transformers.models.mistral.modeling_mistral import rotate_half

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    new_tokens_position_indexes: torch.Tensor,
    position_encoding_indexes: torch.Tensor,
):
    # For the query, we only encode the position of the new tokens
    cos_q = cos[new_tokens_position_indexes].unsqueeze(1)
    sin_q = sin[new_tokens_position_indexes].unsqueeze(1)
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)

    # For the key, we encode the positions of all tokens (new and from cache)
    cos_k = cos[position_encoding_indexes].unsqueeze(1)
    sin_k = sin[position_encoding_indexes].unsqueeze(1)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed

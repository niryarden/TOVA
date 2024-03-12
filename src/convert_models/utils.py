import torch
from transformers.models.mistral.modeling_mistral import rotate_half

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    position_encoding: torch.Tensor,
):
    cos_q = cos[position_ids].unsqueeze(1)
    sin_q = sin[position_ids].unsqueeze(1)
    cos_k = cos[position_encoding].unsqueeze(1)
    sin_k = sin[position_encoding].unsqueeze(1)
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed

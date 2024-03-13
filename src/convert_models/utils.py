import torch
from transformers.models.mistral.modeling_mistral import rotate_half
from ..tova_cache import TOVACache

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_position_indexes: torch.Tensor,
    key_position_indexes: torch.Tensor,
):
    # For the query, we only encode the position of the new tokens
    cos_q = cos[query_position_indexes].unsqueeze(1)
    sin_q = sin[query_position_indexes].unsqueeze(1)
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)

    # For the key, we encode the positions of all tokens (new and from cache)
    cos_k = cos[key_position_indexes].unsqueeze(1)
    sin_k = sin[key_position_indexes].unsqueeze(1)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed

def get_positional_encoding_indexes(past_key_value: TOVACache, position_ids: torch.Tensor, layer_idx: int, is_input_tokens_round: bool):
    if is_input_tokens_round:
        return position_ids, position_ids
    key_position_indexes =  past_key_value.position_encoding_indexes.get_position_indexes(
            past_key_value, layer_idx
        ).detach().clone().reshape(1, -1)
    query_position_indexes = key_position_indexes[0, -1].reshape(1, -1)
    return query_position_indexes, key_position_indexes

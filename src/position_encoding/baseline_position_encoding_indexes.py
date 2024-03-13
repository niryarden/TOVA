import torch
from transformers.cache_utils import Cache
from TOVA.src.position_encoding.position_encoding_indexes import PositionEncodingIndexes

class BaselinePositionEncodingIndexes(PositionEncodingIndexes):
    def get_position_indexes(self, past_key_value: Cache, position_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
        # if there's no cache for this layer yet
        if len(past_key_value.cached_input_indices) <= layer_idx:
            return position_ids
        
        existing_ids = past_key_value.cached_input_indices[layer_idx].detach().clone()
        new_ids = position_ids.detach().clone().reshape(1)
        return torch.cat([existing_ids, new_ids], dim=-1).reshape(1, -1)

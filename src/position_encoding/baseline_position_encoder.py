import torch
from transformers.cache_utils import Cache
from position_encoder import PositionEncoder

class BaselinePositionEncoder(PositionEncoder):
    def get_positions(self, past_key_value: Cache, position_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
        # if there's no cache for this layer yet
        if len(past_key_value.saved_input_indices) <= layer_idx:
            return position_ids
        
        existing_ids = past_key_value.saved_input_indices[layer_idx].detach().clone()
        new_ids = position_ids.detach().clone().reshape(1)
        return torch.cat([existing_ids, new_ids], dim=-1).reshape(1, -1)

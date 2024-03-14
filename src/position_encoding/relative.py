import torch
from TOVA.src.position_encoding.position_encoding_indexes import PositionEncodingIndexes

class RelativePositionEncodingIndexes(PositionEncodingIndexes):
    def get_position_indexes(self, past_key_value, layer_idx, context_limit):
        cached_indexes = past_key_value.cached_input_indexes[layer_idx]
        last_index = cached_indexes[-1].item()
        relational_coefficient =  (context_limit - 1) / last_index
        return torch.ceil(cached_indexes * relational_coefficient).int()

import torch
from TOVA.src.position_encoding.position_encoding_indexes import PositionEncodingIndexes

class ContinuesPositionEncodingIndexes(PositionEncodingIndexes):
    def get_position_indexes(self, past_key_value, layer_idx):
        cached_tokens_amount = past_key_value.cached_input_indexes[layer_idx].shape[-1]
        return torch.arange(cached_tokens_amount)

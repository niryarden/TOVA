import torch
from TOVA.src.position_encoding.position_encoding_indexes import PositionEncodingIndexes

class GappedPositionEncodingIndexes(PositionEncodingIndexes):
    def get_position_indexes(self, past_key_value, layer_idx, context_limit):
        cached_indexes = past_key_value.cached_input_indexes[layer_idx]
        positions = []
        for i in range(cached_indexes.shape[-1]):
            if i == 0:
                positions.append(0)
                continue
            if cached_indexes[i].item() == cached_indexes[i - 1].item() + 1:
                positions.append(positions[-1] + 1)
                continue
            positions.append(positions[-1] + 2)
        return torch.tensor(positions, dtype=torch.int64)

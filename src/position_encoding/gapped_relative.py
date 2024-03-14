import math
import torch
from TOVA.src.position_encoding.position_encoding_indexes import PositionEncodingIndexes

class GappedRelativePositionEncodingIndexes(PositionEncodingIndexes):
    def get_position_indexes(self, past_key_value, layer_idx, context_limit):
        cached_indexes = past_key_value.cached_input_indexes[layer_idx] 
        last_index = cached_indexes[-1].item()
        relational_coefficient =  (context_limit - 1) / last_index
        positions = []
        for i in range(cached_indexes.shape[-1]):
            if i == 0:
                positions.append(0)
                continue
            current_item = cached_indexes[i].item()
            previous_item = cached_indexes[i - 1].item()
            gap = 1 if current_item == previous_item + 1 else 2
            gapped = previous_item + gap
            relative = math.ceil(current_item * relational_coefficient)
            position = max([gapped, relative])
            positions.append(position)
        return torch.tensor(positions, dtype=torch.int64)

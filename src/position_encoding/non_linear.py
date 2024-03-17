import torch
import math
from TOVA.src.position_encoding.position_encoding_indexes import PositionEncodingIndexes


K = 10

def g(x):
    if x < K:
        return x
    return math.ceil(math.sqrt(x))


class NonLinearPositionEncodingIndexes(PositionEncodingIndexes):
    def get_position_indexes(self, past_key_value, layer_idx, context_limit):
        cached_indexes = past_key_value.cached_input_indexes[layer_idx]
        positions = []
        for j in range(cached_indexes.shape[0]):
            i_j = cached_indexes[j].item()
            i_j_minus1 = 0 if j == 0 else cached_indexes[j - 1].item()
            p_j_minus1 = 0 if j == 0 else positions[j - 1]
            p_j = p_j_minus1 + g(i_j - i_j_minus1)
            positions.append(p_j)
        return torch.tensor(positions, dtype=torch.int64)

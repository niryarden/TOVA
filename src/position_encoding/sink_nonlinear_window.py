import math
import torch
from TOVA.src.position_encoding.position_encoding_indexes import PositionEncodingIndexes

K = 20
T = 10

def g(x):
    if x <= T:
        return x
    return T + math.sqrt(x - T)


class SinkNonlinearWindowPositionEncodingIndexes(PositionEncodingIndexes):
    def get_position_indexes(self, past_key_value, layer_idx, context_limit):
        cached_indexes = past_key_value.cached_input_indexes[layer_idx]
        M = cached_indexes.shape[0]
        R = cached_indexes[-1].item() - M

        positions = []
        for j in range(cached_indexes.shape[0]):
            i_j = cached_indexes[j].item()
            i_j_minus1 = 0 if j == 0 else cached_indexes[j - 1].item()
            p_j_minus1 = 0 if j == 0 else positions[j - 1]
            if K <= i_j <= R:
                p_j = p_j_minus1 + math.ceil(g(i_j - i_j_minus1))
                positions.append(p_j)
            else:
                p_j = p_j_minus1 + (i_j - i_j_minus1)
                positions.append(p_j)
        return torch.tensor(positions, dtype=torch.int64).to("cuda" if torch.cuda.is_available() else "cpu")

import torch
import math
from TOVA.src.position_encoding.position_encoding_indexes import PositionEncodingIndexes


def g(x, f, T):
    if x <= T:
        return x
    return T + f(x - T)


class NonLinearPositionEncodingIndexes(PositionEncodingIndexes):
    def __init__(self, f=math.sqrt, T=10) -> None:
        super().__init__()
        self.f = f
        self.T = T

    def get_position_indexes(self, past_key_value, layer_idx, context_limit):
        cached_indexes = past_key_value.cached_input_indexes[layer_idx]
        positions = []
        for j in range(cached_indexes.shape[0]):
            i_j = cached_indexes[j].item()
            i_j_minus1 = 0 if j == 0 else cached_indexes[j - 1].item()
            p_j_minus1 = 0 if j == 0 else positions[j - 1]
            p_j = p_j_minus1 + math.ceil(g(i_j - i_j_minus1, self.f, self.T))
            positions.append(p_j)
        return torch.tensor(positions, dtype=torch.int64).to("cuda" if torch.cuda.is_available() else "cpu")

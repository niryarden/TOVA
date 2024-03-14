import math
import torch
from TOVA.src.position_encoding.position_encoding_indexes import PositionEncodingIndexes

K = 20

class SinkRelativeWindowPositionEncodingIndexes(PositionEncodingIndexes):
    def get_position_indexes(self, past_key_value, layer_idx, context_limit):
        cached_indexes = past_key_value.cached_input_indexes[layer_idx]
        sink, sink_indexes = self.calc_sink(cached_indexes)
        window, window_indexes = self.calc_window(cached_indexes, context_limit)
        p_j_top_plus1 = window[0].item()
        p_j_bottom_minus1 = sink[-1].item()
        relative_center = self.calc_relative_center(
            cached_indexes, sink_indexes, window_indexes, p_j_top_plus1, p_j_bottom_minus1)
        return torch.cat([sink, relative_center, window]).int()


    def calc_sink(cached_indexes):
        sink_mask = cached_indexes < K
        sink = cached_indexes[sink_mask]
        sink_indexes = torch.nonzero(sink_mask)
        sink_indexes = sink_indexes.reshape(sink_indexes.shape[0])
        return sink, sink_indexes

    def calc_window(cached_indexes, N):
        M = cached_indexes.shape[0]
        r = cached_indexes[-1] - M
        window_mask = cached_indexes > r
        window_indexes = torch.nonzero(window_mask)
        window_indexes = window_indexes.reshape(window_indexes.shape[0])
        window = cached_indexes[window_mask]
        window = window - (cached_indexes[-1] - (N - 1))
        return window, window_indexes

    def calc_relative_center(
        cached_indexes, sink_indexes, window_indexes, p_j_top_plus1, p_j_bottom_minus1):
        M = cached_indexes.shape[0]
        j_top = M - window_indexes.shape[0]
        i_j_top = cached_indexes[j_top].item()
        j_bottom = sink_indexes.shape[0]
        i_j_bottom = cached_indexes[j_bottom].item()
        
        mask = torch.ones_like(cached_indexes, dtype=torch.bool)
        mask[sink_indexes], mask[window_indexes] = False, False
        center = cached_indexes[mask]
        
        relative_center = []
        # please note the j by -j_bottom is shifted in comparison to definition
        for j in range(center.shape[0]):
            i_j_minus1 = p_j_bottom_minus1 if j == 0 else center[j - 1].item()
            i_j = center[j].item()
            p_j_tag = i_j_bottom + (p_j_top_plus1 - i_j_bottom) * (i_j - i_j_bottom) / i_j_top

            gap_j = 1 if i_j == i_j_minus1 + 1 else 2
            p_j_minus1 = p_j_bottom_minus1 if j == 0 else relative_center[j - 1]
            p_j = max([math.ceil(p_j_tag), p_j_minus1 + gap_j])
            relative_center.append(p_j)
        return torch.tensor(relative_center)

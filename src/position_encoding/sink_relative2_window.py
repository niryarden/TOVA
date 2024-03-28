import math
import torch
from TOVA.src.position_encoding.position_encoding_indexes import PositionEncodingIndexes

K = 20

class SinkRelative2WindowPositionEncodingIndexes(PositionEncodingIndexes):
    def get_position_indexes(self, past_key_value, layer_idx, context_limit):
        cached_indexes = past_key_value.cached_input_indexes[layer_idx]
        sink = self.calc_sink(cached_indexes)
        window = self.calc_window(cached_indexes, context_limit)
        p_j_top = window[0].item()
        relative_center = self.calc_relative_center(cached_indexes, sink.shape[0], window.shape[0], p_j_top)
        return torch.cat([sink, relative_center, window]).int()


    def calc_sink(self, cached_indexes):
        sink_mask = cached_indexes < K
        sink = cached_indexes[sink_mask]
        return sink

    def calc_window(self, cached_indexes, N):
        M = cached_indexes.shape[0]
        R = cached_indexes[-1] - M
        window_mask = cached_indexes > R
        window = cached_indexes[window_mask]
        shifted_window = window - (cached_indexes[-1] - (N - 1))
        return shifted_window

    def calc_relative_center(
        self, cached_indexes, sink_size, window_size, p_j_top):
        j_top = cached_indexes.shape[0] - window_size
        i_j_top = cached_indexes[j_top].item()
        if sink_size == 0:
            i_j_bottom = p_j_bottom = 0
        else:
            j_bottom = sink_size - 1
            p_j_bottom = i_j_bottom = cached_indexes[j_bottom].item()
        center = cached_indexes[sink_size:-window_size]
        
        relative_center = []
        # please note the j by -j_bottom is shifted in comparison to definition
        for j in range(center.shape[0]):
            i_j = center[j].item()
            p_j = math.ceil(p_j_bottom + i_j * (p_j_top - p_j_bottom) / (i_j_top - i_j_bottom))
            relative_center.append(p_j)
        return torch.tensor(relative_center).to("cuda" if torch.cuda.is_available() else "cpu")

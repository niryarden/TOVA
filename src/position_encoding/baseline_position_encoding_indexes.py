from TOVA.src.position_encoding.position_encoding_indexes import PositionEncodingIndexes

class BaselinePositionEncodingIndexes(PositionEncodingIndexes):
    def get_position_indexes(self, past_key_value, layer_idx, context_limit):
        return past_key_value.cached_input_indices[layer_idx]

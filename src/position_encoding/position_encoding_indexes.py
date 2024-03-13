from abc import ABC, abstractmethod

class PositionEncodingIndexes(ABC):
    @abstractmethod
    def get_position_indexes(self, past_key_value, layer_idx):
        pass

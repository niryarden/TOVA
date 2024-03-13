from abc import ABC, abstractmethod

import torch
from transformers.cache_utils import Cache

class PositionEncodingIndexes(ABC):
    @abstractmethod
    def get_position_indexes(self, past_key_value: Cache, position_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
        pass

from abc import ABC, abstractmethod
from typing import Optional, List


class EmbeddingBase(ABC):
    def __init__(self):
        self.embed_version = "local_v0.0.1_20230525_6d4019f1559aef84abc2ab8257e1ad4c"
        self._max_batchsz = 0

    @abstractmethod
    def get_embedding(self, sentences: List[str], max_length: int) -> List:
        "get embedding for sentences"

    @property
    def max_batchsz(self) -> int:
        return self._max_batchsz

    @property
    def ModelVersion(self) -> str:
        return self.embed_version

    @max_batchsz.setter
    def max_batchsz(self, value):
        self._max_batchsz = value

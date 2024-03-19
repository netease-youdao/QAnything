from abc import ABC, abstractmethod
from typing import Optional, List
from copy import deepcopy


class RerankBase(ABC):

    def __init__(self):
        self.overlap_tokens = 80
        self.spe_id = 0

    @abstractmethod
    def inference(self, batch) -> List:
        "do inference"

    def merge_inputs(self, chunk1_raw, chunk2):
        chunk1 = deepcopy(chunk1_raw)
        chunk1['input_ids'].extend(chunk2['input_ids'])
        chunk1['input_ids'].append(self.spe_id)
        chunk1['attention_mask'].extend(chunk2['attention_mask'])
        chunk1['attention_mask'].append(chunk2['attention_mask'][0])
        if 'token_type_ids' in chunk1:
            token_type_ids = [1 for _ in range(len(chunk2['token_type_ids']) + 1)]
            chunk1['token_type_ids'].extend(token_type_ids)
        return chunk1

    def tokenize_preproc(self, query: str, passages: List[str]):
        "tokenizer preprocessing"

    @abstractmethod
    def predict(self, query: str, passages: List[str]):
        "predict"


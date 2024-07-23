"""Wrapper around YouDao embedding models."""
from typing import List
from qanything_kernel.configs.model_config import LOCAL_EMBED_MAX_LENGTH, LOCAL_EMBED_BATCH, \
    LOCAL_EMBED_PATH, LOCAL_EMBED_WORKERS
from qanything_kernel.utils.general_utils import get_time
from qanything_kernel.utils.custom_log import debug_logger
from transformers import AutoTokenizer
import concurrent.futures
from tqdm import tqdm
from abc import ABC, abstractmethod


class EmbeddingBackend(ABC):
    embed_version = "local_v0.0.1_20230525_6d4019f1559aef84abc2ab8257e1ad4c"

    def __init__(self, use_cpu: bool = False):
        self.use_cpu = use_cpu
        self._tokenizer = AutoTokenizer.from_pretrained(LOCAL_EMBED_PATH)
        self.workers = LOCAL_EMBED_WORKERS

    @abstractmethod
    def get_embedding(self, sentences, max_length) -> List:
        pass

    @get_time
    def get_len_safe_embeddings(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        batch_size = LOCAL_EMBED_BATCH

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                future = executor.submit(self.get_embedding, batch, LOCAL_EMBED_MAX_LENGTH)
                futures.append(future)
            debug_logger.info(f'embedding number: {len(futures)}')
            for future in tqdm(futures):
                embeddings = future.result()
                all_embeddings += embeddings
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs using multithreading, maintaining the original order."""
        return self.get_len_safe_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    @property
    def getModelVersion(self):
        return self.embed_version

from abc import ABC, abstractmethod
from typing import Optional, List
from langchain_core.embeddings import Embeddings
import concurrent.futures
from qanything_kernel.utils.custom_log import debug_logger
from tqdm import tqdm
from qanything_kernel.configs.model_config import LOCAL_EMBED_MODEL_PATH, LOCAL_EMBED_MAX_LENGTH, LOCAL_EMBED_BATCH, \
    LOCAL_EMBED_PATH, LOCAL_EMBED_REPO


class EmbeddingBase(Embeddings):
    def __init__(self):
        self.embed_version = "local_v0.0.1_20230525_6d4019f1559aef84abc2ab8257e1ad4c"
        self._max_batchsz = 0

    @abstractmethod
    def get_embedding(self, sentences: List[str], max_length: int) -> List:
        "get embedding for sentences"

    def get_len_safe_embeddings(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        batch_size = self.max_batchsz

        with concurrent.futures.ThreadPoolExecutor() as executor:
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
    def max_batchsz(self) -> int:
        return self._max_batchsz

    @property
    def ModelVersion(self) -> str:
        return self.embed_version

    @max_batchsz.setter
    def max_batchsz(self, value):
        self._max_batchsz = value

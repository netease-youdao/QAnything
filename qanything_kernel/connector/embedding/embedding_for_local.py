"""Wrapper around YouDao embedding models."""
from typing import List

from qanything_kernel.connector.embedding.embedding_client import EmbeddingClient
import concurrent.futures

embedding_client = EmbeddingClient(
    server_url='localhost:10001',
    model_name='embed',
    model_version='1',
    resp_wait_s=120,
    tokenizer_path='qanything_kernel/connector/embedding/embedding_model_0630')


class YouDaoLocalEmbeddings:
    def __init__(self):
        pass

    def _get_embedding(self, queries):
        embeddings = embedding_client.get_embedding(queries)
        return embeddings

    def _get_len_safe_embeddings(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        batch_size = 16
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                future = executor.submit(self._get_embedding, batch)
                futures.append(future)
            for future in futures:
                embeddings = future.result()
                all_embeddings += embeddings
        return all_embeddings

    @property
    def embed_version(self):
        return embedding_client.getModelVersion()

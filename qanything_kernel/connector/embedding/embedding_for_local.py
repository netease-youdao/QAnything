"""Wrapper around YouDao embedding models."""
import platform
from typing import List
from qanything_kernel.configs.model_config import LOCAL_EMBED_MODEL_PATH, LOCAL_EMBED_MAX_LENGTH, LOCAL_EMBED_BATCH, LOCAL_EMBED_PATH, LOCAL_EMBED_REPO
from qanything_kernel.utils.custom_log import debug_logger
import concurrent.futures
from tqdm import tqdm
import qanything_kernel.connector.gpuinfo.global_vars as global_vars
# from huggingface_hub import snapshot_download

gpu_type = global_vars.get_gpu_type()

if gpu_type == "nvidia":
    from qanything_kernel.connector.embedding.embedding_client_onnx import EmbeddingClientONNX
elif gpu_type == "metal":
    from qanything_kernel.connector.embedding.embedding_client_torch_mps import EmbeddingClientTorchMPS
elif gpu_type == 'intel':
    from qanything_kernel.connector.embedding.embedding_client_torch_xpu import EmbeddingClientTorchXPU


class YouDaoLocalEmbeddings:
    def __init__(self):
        if gpu_type == "nvidia":
            self.embedding_client: EmbeddingClientONNX = EmbeddingClientONNX(model_path=LOCAL_EMBED_MODEL_PATH,
                                                                             tokenizer_path=LOCAL_EMBED_PATH)
        elif gpu_type == "metal":
            self.embedding_client: EmbeddingClientTorchMPS = EmbeddingClientTorchMPS()
        elif gpu_type == 'intel':
            self.embedding_client: EmbeddingClientTorchXPU = EmbeddingClientTorchXPU()

    def _get_embedding(self, queries):
        embeddings = self.embedding_client.get_embedding(queries, max_length=LOCAL_EMBED_MAX_LENGTH)
        return embeddings

    def _get_len_safe_embeddings(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        batch_size = self.embedding_client.max_batchsz

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                future = executor.submit(self._get_embedding, batch)
                futures.append(future)
            debug_logger.info(f'embedding number: {len(futures)}')
            for future in tqdm(futures):
                embeddings = future.result()
                all_embeddings += embeddings
        return all_embeddings

    @property
    def embed_version(self):
        return self.embedding_client.ModelVersion

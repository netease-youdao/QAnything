"""Wrapper around YouDao embedding models."""
from typing import List

from qanything_kernel.connector.embedding.embedding_client import EmbeddingClient
from qanything_kernel.configs.model_config import LOCAL_EMBED_MODEL_PATH, LOCAL_EMBED_MAX_LENGTH, LOCAL_EMBED_BATCH, LOCAL_EMBED_PATH, LOCAL_EMBED_REPO
from qanything_kernel.utils.custom_log import debug_logger
import concurrent.futures
from tqdm import tqdm 
# from huggingface_hub import snapshot_download
from modelscope import snapshot_download
import subprocess

import os

# 如果模型不存在, 下载模型
if not os.path.exists(LOCAL_EMBED_MODEL_PATH):
    # snapshot_download(repo_id=LOCAL_EMBED_REPO, local_dir=LOCAL_EMBED_PATH, local_dir_use_symlinks="auto")
    debug_logger.info(f"开始下载embedding模型：{LOCAL_EMBED_REPO}")
    cache_dir = snapshot_download(model_id=LOCAL_EMBED_REPO)
    # 如果存在的话，删除LOCAL_EMBED_PATH
    os.system(f"rm -rf {LOCAL_EMBED_PATH}")
    output = subprocess.check_output(['ln', '-s', cache_dir, LOCAL_EMBED_PATH], text=True)
    debug_logger.info(f"模型下载完毕！cache地址：{cache_dir}, 软链接地址：{LOCAL_EMBED_PATH}")



embedding_client = EmbeddingClient(
    model_path=LOCAL_EMBED_MODEL_PATH,
    tokenizer_path=LOCAL_EMBED_PATH)


class YouDaoLocalEmbeddings:
    def __init__(self):
        pass

    def _get_embedding(self, queries):
        embeddings = embedding_client.get_embedding(queries, max_length=LOCAL_EMBED_MAX_LENGTH)
        return embeddings

    def _get_len_safe_embeddings(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        batch_size = LOCAL_EMBED_BATCH

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
        return embedding_client.getModelVersion()

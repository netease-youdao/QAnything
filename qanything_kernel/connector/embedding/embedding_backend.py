"""Wrapper around YouDao embedding models."""
from typing import List
from qanything_kernel.configs.model_config import LOCAL_EMBED_MODEL_PATH, LOCAL_EMBED_MAX_LENGTH, LOCAL_EMBED_BATCH, \
    LOCAL_EMBED_PATH, LOCAL_EMBED_REPO, LOCAL_EMBED_WORKERS
from qanything_kernel.utils.general_utils import get_time
from qanything_kernel.utils.custom_log import debug_logger
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer
import concurrent.futures
from tqdm import tqdm
from modelscope import snapshot_download
import subprocess
from abc import ABC, abstractmethod

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


class EmbeddingBackend(Embeddings):
    embed_version = "local_v0.0.1_20230525_6d4019f1559aef84abc2ab8257e1ad4c"

    def __init__(self, use_cpu):
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

"""Wrapper around YouDao embedding models."""
from typing import (
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

from qanything_kernel.utils.custom_log import debug_logger, qa_logger
from qanything_kernel.utils.general_utils import get_time_async, get_time
from langchain_core.embeddings import Embeddings
from qanything_kernel.configs.model_config import LOCAL_EMBED_SERVICE_URL, LOCAL_EMBED_WORKERS
import json
import urllib.request
import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed


class YouDaoEmbeddings(Embeddings):
    model_name: str = "text-embedding-youdao-001"
    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Tuple[()]] = "all"

    def __init__(self):
        self.model_version = None
        self.url = f"http://{LOCAL_EMBED_SERVICE_URL}/embedding"
        super().__init__()

    def _get_embedding(self, queries):
        for idx, query in enumerate(queries):
            # 去除query中所有以![figure]或![equation]开头的行
            queries[idx] = '\n'.join([line for line in query.split('\n') if
                                      not line.strip().startswith('![figure]') and not line.strip().startswith(
                                          '![equation]')])

        data = {
            'texts': queries,
        }
        headers = {"content-type": "application/json"}
        req = urllib.request.Request(
            url=self.url,
            headers=headers,
            data=json.dumps(data).encode("utf-8")
        )
        try:
            f = urllib.request.urlopen(
                req
            )
            js = json.loads(f.read().decode())
            return js
        except Exception as e:
            debug_logger.info('embedding data length:', sum(len(s) for s in queries))
            debug_logger.error('embedding error:', traceback.format_exc())
            return None

    @get_time
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs using multithreading, maintaining the original order."""
        batch_size = 16
        all_embeddings = [None for _ in range(len(texts))]  # 预先分配空间
        with ThreadPoolExecutor(max_workers=LOCAL_EMBED_WORKERS) as executor:
            # 创建一个未来对象字典，键为索引，值为线程执行的结果
            future_to_index = {executor.submit(self._get_embedding, texts[i:i + batch_size]): i // batch_size
                               for i in range(0, len(texts), batch_size)}

            for future in as_completed(future_to_index):
                result_index = future_to_index[future]
                res = future.result()
                start_index = result_index * batch_size
                # 将结果按原始顺序存储到 all_embeddings
                all_embeddings[start_index:start_index + batch_size] = res['embeddings']
                if self.model_version is None:
                    self.model_version = self.model_version

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self._get_embedding([text])['embeddings'][0]

    @property
    def embed_version(self):
        return self.model_version

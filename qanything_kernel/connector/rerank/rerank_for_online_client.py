import json
import urllib
import urllib.parse
import urllib.request
from urllib import request
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.utils.general_utils import get_time_async, get_time
from qanything_kernel.configs.model_config import LOCAL_RERANK_SERVICE_URL, LOCAL_RERANK_WORKERS
from langchain.schema import Document
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List


class YouDaoRerank:
    def __init__(self):
        self.url = f"http://{LOCAL_RERANK_SERVICE_URL}/rerank"

    def _get_rerank_res(self, query, passages):
        data = {
            'query': query,
            'passages': passages
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
            scores = json.loads(f.read().decode())
            return scores
        except Exception as e:
            debug_logger.info(f'rerank query: {query}, rerank passages length: {len(passages)}')
            debug_logger.error(f'rerank error: {traceback.format_exc()}')
            return None

    @get_time
    def rerank_documents(self, query: str, source_documents: List[Document]) -> List[Document]:
        """Embed search docs using multithreading, maintaining the original order."""
        batch_size = 16
        all_scores = [None for _ in range(len(source_documents))]
        passages = [doc.page_content for doc in source_documents]
        with ThreadPoolExecutor(max_workers=LOCAL_RERANK_WORKERS) as executor:
            # 创建一个未来对象字典，键为索引，值为线程执行的结果
            future_to_index = {executor.submit(self._get_rerank_res, query, passages[i:i + batch_size]): i // batch_size
                               for i in range(0, len(passages), batch_size)}

            for future in as_completed(future_to_index):
                result_index = future_to_index[future]
                res = future.result()
                if res is None:
                    return source_documents
                start_index = result_index * batch_size
                # 将结果按原始顺序存储到 all_embeddings
                all_scores[start_index:start_index + batch_size] = res

        # debug_logger.info(f"rerank scores: {all_scores}")
        for idx, score in enumerate(all_scores):
            source_documents[idx].metadata['score'] = score
        source_documents = sorted(source_documents, key=lambda x: x.metadata['score'], reverse=True)

        return source_documents

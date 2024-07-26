import asyncio
import aiohttp
from typing import List
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.utils.general_utils import get_time_async
from qanything_kernel.configs.model_config import LOCAL_RERANK_SERVICE_URL
from langchain.schema import Document
import traceback


class YouDaoRerank:
    def __init__(self):
        self.url = f"http://{LOCAL_RERANK_SERVICE_URL}/rerank"

    async def _get_rerank_res(self, query, passages):
        data = {
            'query': query,
            'passages': passages
        }
        headers = {"content-type": "application/json"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=data, headers=headers) as response:
                    if response.status == 200:
                        scores = await response.json()
                        return scores
                    else:
                        debug_logger.error(f'Rerank request failed with status {response.status}')
                        return None
        except Exception as e:
            debug_logger.info(f'rerank query: {query}, rerank passages length: {len(passages)}')
            debug_logger.error(f'rerank error: {traceback.format_exc()}')
            return None

    @get_time_async
    async def arerank_documents(self, query: str, source_documents: List[Document]) -> List[Document]:
        """Embed search docs using async calls, maintaining the original order."""
        batch_size = 64  # 增大客户端批处理大小
        all_scores = [None for _ in range(len(source_documents))]
        passages = [doc.page_content for doc in source_documents]

        tasks = []
        for i in range(0, len(passages), batch_size):
            task = asyncio.create_task(self._get_rerank_res(query, passages[i:i + batch_size]))
            tasks.append((i, task))

        for start_index, task in tasks:
            res = await task
            if res is None:
                return source_documents
            all_scores[start_index:start_index + batch_size] = res

        for idx, score in enumerate(all_scores):
            source_documents[idx].metadata['score'] = score
        source_documents = sorted(source_documents, key=lambda x: x.metadata['score'], reverse=True)

        return source_documents


# 使用示例
# async def main():
#     reranker = YouDaoRerank()
#     query = "Your query here"
#     documents = [Document(page_content="content1"), Document(page_content="content2")]  # 示例文档
#     reranked_docs = await reranker.rerank_documents(query, documents)
#     return reranked_docs
#
#
# # 运行异步主函数
# if __name__ == "__main__":
#     reranked_docs = asyncio.run(main())

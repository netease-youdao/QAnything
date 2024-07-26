"""Wrapper around YouDao embedding models."""
from typing import List
from qanything_kernel.utils.custom_log import debug_logger, qa_logger
from qanything_kernel.utils.general_utils import get_time_async, get_time
from langchain_core.embeddings import Embeddings
from qanything_kernel.configs.model_config import LOCAL_EMBED_SERVICE_URL
import traceback
import aiohttp
import asyncio
import requests


def _process_query(query):
    return '\n'.join([line for line in query.split('\n') if
                      not line.strip().startswith('![figure]') and
                      not line.strip().startswith('![equation]')])


class YouDaoEmbeddings(Embeddings):
    def __init__(self):
        self.model_version = 'local_v20240725'
        self.url = f"http://{LOCAL_EMBED_SERVICE_URL}/embedding"
        self.session = requests.Session()
        super().__init__()

    async def _get_embedding_async(self, session, queries):
        data = {'texts': queries}
        async with session.post(self.url, json=data) as response:
            return await response.json()

    @get_time_async
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 64  # 增大客户端批处理大小
        all_embeddings = []
        async with aiohttp.ClientSession() as session:
            tasks = [self._get_embedding_async(session, texts[i:i + batch_size])
                     for i in range(0, len(texts), batch_size)]
            results = await asyncio.gather(*tasks)
            for result in results:
                all_embeddings.extend(result)
        debug_logger.info(f'success embedding number: {len(all_embeddings)}')
        return all_embeddings

    async def aembed_query(self, text: str) -> List[float]:
        return (await self.aembed_documents([text]))[0]

    def _get_embedding_sync(self, texts):
        data = {'texts': [_process_query(text) for text in texts]}
        try:
            response = self.session.post(self.url, json=data)
            response.raise_for_status()
            result = response.json()
            return result
        except Exception as e:
            debug_logger.error(f'sync embedding error: {traceback.format_exc()}')
            return None

    @get_time
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_embedding_sync(texts)

    @get_time
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        # return self._get_embedding([text])['embeddings'][0]
        return self._get_embedding_sync([text])[0]

    @property
    def embed_version(self):
        return self.model_version

# 使用示例
# async def main():
#     embedder = YouDaoEmbeddings()
#     query = "Your query here"
#     texts = ["text1", "text2"]  # 示例文本
#     embeddings = await embedder.aembed_documents(texts)
#     return embeddings

# if __name__ == '__main__':
#     embeddings = asyncio.run(main())

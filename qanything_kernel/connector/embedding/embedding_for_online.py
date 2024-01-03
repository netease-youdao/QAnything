from typing import (
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import json
import urllib.request
import traceback

import concurrent.futures


class YouDaoEmbeddings:
    model_name: str = "text-embedding-youdao-001"
    deployment: str = model_name  # to support Azure OpenAI Service custom deployment names
    embedding_ctx_length: int = 416
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch"""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    base_url: str = 'https://embedding.corp.youdao.com'

    def __init__(self):
        pass

    def _get_embedding(self, queries):
        data = {
            'queries': queries,
        }
        print('embedding data length:', sum(len(s) for s in queries), flush=True)
        headers = {"content-type": "application/json"}
        url = self.base_url + "/embedding"
        req = urllib.request.Request(
            url=url,
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
            print('embedding error:', traceback.format_exc())
            return None

    def getModelVersion(self):
        data = ''
        headers = {"content-type": "application/json"}

        url = self.base_url + "/getModelVersion"
        req = urllib.request.Request(
            url=url,
            headers=headers,
            data=json.dumps(data).encode("utf-8")
        )

        f = urllib.request.urlopen(
            req
        )
        js = json.loads(f.read().decode())

        return js

    def _get_len_safe_embeddings(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        batch_size = 16
        total_texts = sum(len(s) for s in texts)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                future = executor.submit(self._get_embedding, batch)
                futures.append(future)
            for future in futures:
                embd_js = future.result()
                if embd_js:
                    embeddings = embd_js["embeddings"]
                    model_version = embd_js["model_version"]
                    print(model_version)
                    all_embeddings += embeddings
                else:
                    raise Exception("embedding error, data length: %d" % total_texts)
        return all_embeddings

    @property
    def embed_version(self):
        return self.getModelVersion()['model_version']

import os
import math
import numpy as np
import time
import json
import requests

from typing import Optional

from transformers import AutoTokenizer

class EmbeddingClient:
    DEFAULT_MAX_RESP_WAIT_S = 120
    embed_version = "local_v0.0.1_20230525_6d4019f1559aef84abc2ab8257e1ad4c"

    def __init__(
        self,
        server_url: str,
        model_name: str,
        model_version: str,
        tokenizer_path: str,
        resp_wait_s: Optional[float] = None,
    ):
        self._server_url = server_url
        self._model_name = model_name
        self._model_version = model_version
        self._response_wait_t = self.DEFAULT_MAX_RESP_WAIT_S if resp_wait_s is None else resp_wait_s
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def get_embedding(self, sentences, max_length=512):
        # Setting up client

        sentences_data = {"sentences": sentences}
        headers = {
            'Content-Type': 'application/json'
        }

        resp = requests.request("POST", "https://region-101.seetacloud.com:51501/embeddings", headers=headers, data=json.dumps(sentences_data), verify=False)
        result = json.loads(resp.text)
        embeddings = np.array(result["embeddings"])
        norm_arr = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / norm_arr
        return embeddings_normalized.tolist()

    def getModelVersion(self):
        return self.embed_version


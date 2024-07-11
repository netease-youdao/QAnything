#!/usr/bin/python
#-*- coding:utf-8 -*-
#################################
# File Name: rerank_client.py
# Author: renzihui
# Date: 2023/10/12 13:48:07
# Description: I can do anything.
#################################
import os
import requests
import json
import urllib
import urllib.parse
import urllib.request
from urllib import request
from dotenv import load_dotenv
from qanything_kernel.utils.custom_log import debug_logger

load_dotenv()


def rerank_cohere_client(query, docs):
    debug_logger.info(f'##cohere Reranker## - Info - start!')
    url = "https://api.cohere.ai/v1/rerank"
    key = os.getenv("COHERE_APPKEY") 
    payload = {
        "return_documents": False,
        "max_chunks_per_doc": 10,
        "model": "rerank-multilingual-v2.0",
        "query": query,
        "documents": docs
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {key}"
    }

    proxies = {
        "http": "http://dy9cf10c35.f.yodao.net:30022",
        "https": "http://dy9cf10c35.f.yodao.net:30022"
    }

    try:
        # response = requests.post(url, json=payload, headers=headers, proxies=proxies)
        response = requests.post(url, json=payload, headers=headers, timeout=120)

        results = response.json()['results']
        ranks = []
        for res in results:
            index = res["index"]
            relevance_score = res["relevance_score"]
            if relevance_score < 0.28 and len(ranks) > 0:
                continue
            rank = {"index": index, "relevance_score": relevance_score}
            ranks.append(rank)
    except requests.exceptions.Timeout:
        # 请求超时处理
        debug_logger.warning("cohere rerank请求超时，返回空列表")
        return []
    except Exception as e:
        debug_logger.error(f'cohere rerank出错: {e}')
        return []
    return ranks


def rerank_bce_client(query, docs):
    debug_logger.info(f'##BCE Reranker## - Info - start!')
    url = "https://embedding.corp.youdao.com/reranker/rerank"
    data = {'query': query, 'passages': docs}
    headers = {"content-type": "application/json"}
    req = request.Request(
        url=url,
        headers=headers,
        data=json.dumps(data).encode("utf-8")
    )
    rerank_res = None
    try_num = 3
    while try_num > 0:
        try:
            f = urllib.request.urlopen(req)
            rerank_res = json.loads(f.read().decode())
            try_num = -1
        except Exception as e:
            debug_logger.error(f'##BCE Reranker## - Error - {e}')
            try_num -= 1
            rerank_res = None

    if rerank_res is None:
        return [{"index": 0, "relevance_score": 0.5}]

    # debug_logger.info(f"rerank res: {rerank_res}")

    ranks = []
    for index, relevance_score in zip(rerank_res['rerank_indices'], rerank_res['rerank_scores']):
        if relevance_score < 0.28 and len(ranks) > 0:
            continue
        rank = {"index": index, "relevance_score": relevance_score}
        ranks.append(rank)

    # debug_logger.info(f'##BCE Reranker## - Info - finished !')
    return ranks



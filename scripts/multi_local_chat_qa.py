import os
import json
import requests
import time
import random
import string
import hashlib
import argparse
import concurrent.futures
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import threading

lock = threading.Lock()

def write_to_file_safe(file_name, data):
    # 获取锁
    with lock:
        with open(file_name, 'a') as f:
            f.write(data + '\n')
    # 释放锁会在with语句块结束时自动进行

def stream_requests(ques, output_file):
    base_url = "http://0.0.0.0:8777"
    URL = base_url + "/api/local_doc_qa/local_doc_chat" #流式
    data = {
            "kb_ids": [
            "KBf46828db208c4289a120a34f0fc96147",
            "KBc2440f13e98f4736b5ef81cfaebef3a9",
            "KBb78af28c73f74fb4ae6ad44b3c53302f",
            "KB6c2b097d83be430ab809e361fa8dcc8b",
            "KB69331d593f5b4b5bb555a0ea1b145e5b",
            "KB3cdc79f8c8d24a14bffd27e6570c33da"
            ],
            "question": ques,
            "user_id": "liujx_265",
            "streaming": False,
            "rerank": True,
            "history": []
    }
    response = requests.post(
        URL,
        json=data,
        timeout=60,
        stream=True
    )
    print("response", response)
    print(response.iter_lines)
    for line in response.iter_lines(decode_unicode=False, delimiter=b"\n\n"):
    # for line in response.iter_lines():
        if line:
            yield line

def no_stream_requests(ques, output_file):
    url = 'https://qanything-local-test-265.site.youdao.com/api/local_doc_qa/local_doc_chat'
    headers = {'content-type': 'application/json'}
    data = {
            "kb_ids": [
            "KBf46828db208c4289a120a34f0fc96147",
            "KBc2440f13e98f4736b5ef81cfaebef3a9",
            "KBb78af28c73f74fb4ae6ad44b3c53302f",
            "KB6c2b097d83be430ab809e361fa8dcc8b",
            "KB69331d593f5b4b5bb555a0ea1b145e5b",
            "KB3cdc79f8c8d24a14bffd27e6570c33da"
            ],
            "question": ques,
            "user_id": "liujx_265",
            "streaming": False,
            "rerank": True,
            "history": []
    }
    try:
        response = requests.post(url=url, headers=headers, json=data, timeout=60)
        res = response.json()
        res = data['question'] + '::' + res['response']
        print(res)
        write_to_file_safe(output_file, res)
    except Exception as e:
        print(f"请求发送失败: {e}")
   
def test_stream():
    data_raw = {
        "kb_ids": [
           "KBf46828db208c4289a120a34f0fc96147",
           "KBc2440f13e98f4736b5ef81cfaebef3a9",
           "KBb78af28c73f74fb4ae6ad44b3c53302f",
           "KB6c2b097d83be430ab809e361fa8dcc8b",
           "KB69331d593f5b4b5bb555a0ea1b145e5b",
           "KB3cdc79f8c8d24a14bffd27e6570c33da"
        ],
        "question": "西南交通大学是211院校吗",
        "user_id": "liujx_265",
        "streaming": True,
        "rerank": True,
        "history": []
    }
    for i, chunk in enumerate(stream_requests(data_raw)):
        if chunk:
            chunkstr = chunk.decode("utf-8")[6:]
            chunkjs = json.loads(chunkstr)
            print(chunkjs)

def test():
    data_raw = {
        "kb_ids": [
           "KBf46828db208c4289a120a34f0fc96147",
           "KBc2440f13e98f4736b5ef81cfaebef3a9",
           "KBb78af28c73f74fb4ae6ad44b3c53302f",
           "KB6c2b097d83be430ab809e361fa8dcc8b",
           "KB69331d593f5b4b5bb555a0ea1b145e5b",
           "KB3cdc79f8c8d24a14bffd27e6570c33da"
        ],
        "question": "西南交通大学是211院校吗",
        "user_id": "liujx_265",
        "rerank": True,
        "history": []
    }
    print(type(no_stream_requests))
    no_stream_requests(data_raw) 
    

def measure_latency(ques, output_file, is_stream=False):
    start_time = time.time()
    if is_stream:
        _ = list(stream_requests(ques, output_file))
    else:
        no_stream_requests(ques, output_file)
    end_time = time.time()
    return end_time - start_time

def perform_load_test(concurrency, total_requests, questions, output_file, is_stream=False):
    latencies = []
    questions = ["什么是三大专项", "江苏高三物生地，军校能不能报，哪些专业不能报", "山东文科在江苏怎么选学校", "东南大学化学工程与工艺，生物科学，制药工程分流哪个好？", "男生高三物化地，辽宁，学日语好选学校吗"]
    #questions = ["什么是三大专项"] * 5
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_request = {executor.submit(measure_latency, random.choice(questions), output_file, is_stream): i for i in range(total_requests)}
        for future in concurrent.futures.as_completed(future_to_request):
            try:
                latency = future.result()
                latencies.append(latency)
            except Exception as e:
                print(f"请求执行异常: {e}")

    # 计算统计数据
    p99 = np.percentile(latencies, 99)
    p95 = np.percentile(latencies, 95)
    total_time = sum(latencies)
    qps = total_requests / total_time

    return latencies, p99, p95, qps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='I can do anything.') 
    parser.add_argument('-i', '--input_file', default='升学百科benchmark_20231120.xlsx')
    parser.add_argument('-o', '--output_file', default='output_res.txt')
    parser.add_argument('-c', '--concurrency', type=int, default=10, help='并发数量')
    parser.add_argument('-n', '--total_requests', type=int, default=100, help='总请求数量')
    parser.add_argument('--stream', action='store_true', help='是否进行流式请求')
    args = parser.parse_args()

    df = pd.read_excel(args.input_file)
    output_file = args.output_file
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    output_file = f"{output_file}_{args.concurrency}_{args.total_requests}_{args.stream}_{timestamp}.txt"
    questions = []
    for index, row in tqdm(df.iterrows()):
        question = row['问题']
        questions.append(question)

    # 执行压测
    latencies, p99, p95, qps = perform_load_test(args.concurrency, args.total_requests, questions, output_file, args.stream)

    # 打印统计结果
    print(f"延迟P99: {p99} 秒")
    print(f"延迟P95: {p95} 秒")
    print(f"QPS: {qps} 请求/秒")
    write_to_file_safe(output_file, f"延迟P99: {p99} 秒")
    write_to_file_safe(output_file, f"延迟P95: {p95} 秒")
    write_to_file_safe(output_file, f"QPS: {qps} 请求/秒")

    # test()


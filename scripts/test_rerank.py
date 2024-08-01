import random
import psutil
import asyncio
import time
import numpy as np
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
import logging
import string
import gc
from copy import deepcopy
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 测试配置
TEST_DURATION = 10  # 测试持续时间（秒）
BATCH_SIZES = [1, 2, 4, 8, 16]
NUM_THREADS = [1, 2, 4]
MODEL_PATH = "/root/models/linux_onnx/rerank_model_configs_v0.0.1/rerank.onnx"  # 请替换为实际的模型路径
LOCAL_RERANK_PATH = "/root/models/linux_onnx/rerank_model_configs_v0.0.1"  # 请替换为实际的分词器路径
LOCAL_RERANK_MAX_LENGTH = 512
LOCAL_RERANK_BATCH = 4
LOCAL_RERANK_THREADS = 1


class RerankAsyncBackend:
    def __init__(self, model_path, use_cpu=True, num_threads=4, batch_size=16):
        self.use_cpu = use_cpu
        self.overlap_tokens = 80
        self.max_length = LOCAL_RERANK_MAX_LENGTH
        self.return_tensors = "np"
        self.batch_size = batch_size
        # 创建一个ONNX Runtime会话设置，使用GPU执行
        sess_options = SessionOptions()
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        if use_cpu:
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.executor = ThreadPoolExecutor(max_workers=num_threads)

        self.session = InferenceSession(model_path, sess_options, providers=providers)
        self._tokenizer = AutoTokenizer.from_pretrained(LOCAL_RERANK_PATH, use_fast=True)
        self.spe_id = self._tokenizer.sep_token_id

        self.queue = asyncio.Queue()
        asyncio.create_task(self.process_queue())

    def inference(self, batch):
        logger.info(f"rerank shape: {batch['attention_mask'].shape}")
        # 准备输入数据
        inputs = {self.session.get_inputs()[i].name: batch[name]
                  for i, name in enumerate(['input_ids', 'attention_mask', 'token_type_ids'])
                  if name in batch}

        # 执行推理 输出为logits
        result = self.session.run(None, inputs)  # None表示获取所有输出
        # debug_logger.info(f"rerank result: {result}")

        # 应用sigmoid函数
        sigmoid_scores = 1 / (1 + np.exp(-np.array(result[0])))

        return sigmoid_scores.reshape(-1).tolist()

    def merge_inputs(self, chunk1_raw, chunk2):
        chunk1 = deepcopy(chunk1_raw)
        chunk1['input_ids'].extend(chunk2['input_ids'])
        chunk1['input_ids'].append(self.spe_id)
        chunk1['attention_mask'].extend(chunk2['attention_mask'])
        chunk1['attention_mask'].append(chunk2['attention_mask'][0])
        if 'token_type_ids' in chunk1:
            token_type_ids = [1 for _ in range(len(chunk2['token_type_ids']) + 1)]
            chunk1['token_type_ids'].extend(token_type_ids)
        return chunk1

    def tokenize_preproc(self, query: str, passages: List[str]):
        query_inputs = self._tokenizer(query, add_special_tokens=False, truncation=False, padding=False)
        max_passage_inputs_length = self.max_length - len(query_inputs['input_ids']) - 1
        assert max_passage_inputs_length > 10
        overlap_tokens = min(self.overlap_tokens, max_passage_inputs_length * 2 // 7)

        merge_inputs, merge_inputs_idxs = [], []
        for pid, passage in enumerate(passages):
            passage_inputs = self._tokenizer(passage, add_special_tokens=False, truncation=False, padding=False)
            passage_inputs_length = len(passage_inputs['input_ids'])

            if passage_inputs_length <= max_passage_inputs_length:
                if not passage_inputs['attention_mask']:
                    continue
                qp_merge_inputs = self.merge_inputs(query_inputs, passage_inputs)
                merge_inputs.append(qp_merge_inputs)
                merge_inputs_idxs.append(pid)
            else:
                start_id = 0
                while start_id < passage_inputs_length:
                    end_id = start_id + max_passage_inputs_length
                    sub_passage_inputs = {k: v[start_id:end_id] for k, v in passage_inputs.items()}
                    start_id = end_id - overlap_tokens if end_id < passage_inputs_length else end_id

                    qp_merge_inputs = self.merge_inputs(query_inputs, sub_passage_inputs)
                    merge_inputs.append(qp_merge_inputs)
                    merge_inputs_idxs.append(pid)

        return merge_inputs, merge_inputs_idxs

    async def get_rerank_async(self, query: str, passages: List[str]):
        tot_batches, merge_inputs_idxs_sort = self.tokenize_preproc(query, passages)

        futures = []
        mini_batch = 1  # 设置mini_batch为1
        for i in range(0, len(tot_batches), mini_batch):
            future = asyncio.Future()
            futures.append(future)
            await self.queue.put((tot_batches[i:i + mini_batch], future))

        results = await asyncio.gather(*futures)
        tot_scores = [score for batch_scores in results for score in batch_scores]

        merge_tot_scores = [0 for _ in range(len(passages))]
        for pid, score in zip(merge_inputs_idxs_sort, tot_scores):
            merge_tot_scores[pid] = max(merge_tot_scores[pid], score)

        return merge_tot_scores

    async def process_queue(self):
        while True:
            batch_items = []
            futures = []

            try:
                while len(batch_items) < self.batch_size:
                    batch, future = await asyncio.wait_for(self.queue.get(), timeout=0.01)
                    batch_items.extend(batch)
                    futures.append((future, len(batch)))
            except asyncio.TimeoutError:
                pass

            if batch_items:
                loop = asyncio.get_running_loop()
                input_batch = self._tokenizer.pad(
                    batch_items,
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=None,
                    return_tensors=self.return_tensors
                )
                result = await loop.run_in_executor(self.executor, self.inference, input_batch)

                start = 0
                for future, item_count in futures:
                    end = start + item_count
                    future.set_result(result[start:end])
                    start = end
            else:
                await asyncio.sleep(0.1)

    async def get_rerank(self, query: str, passages: List[str]):
        return await self.get_rerank_async(query, passages)

    async def close(self):
        self.is_running = False
        if hasattr(self, 'queue_task'):
            self.queue_task.cancel()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        large_attributes = ['session', '_tokenizer', 'queue', 'executor']
        for attr in large_attributes:
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()


def generate_test_data(num_samples=1000, max_length=512):
    all_chars = string.ascii_letters + string.digits + " "

    def generate_random_string(length):
        return ''.join(np.random.choice(list(all_chars), size=length))

    return [generate_random_string(np.random.randint(10, max_length + 1)) for _ in range(num_samples)]


async def client(backend, query: str, passages: List[str], request_count):
    for _ in range(request_count):
        start_time = time.time()
        await backend.get_rerank(query, passages)
        latency = time.time() - start_time
        return latency


async def run_test(batch_size, num_threads):
    backend = RerankAsyncBackend(MODEL_PATH, use_cpu=True, num_threads=num_threads, batch_size=batch_size)
    test_queries = generate_test_data(num_samples=100, max_length=50)  # 生成查询
    test_passages = generate_test_data(num_samples=1000, max_length=1024)  # 生成段落

    start_time = time.time()
    end_time = start_time + TEST_DURATION
    request_count = 0
    latencies = []

    process = psutil.Process()
    max_memory = 0
    memory_usage = []
    total_passages = 0

    while time.time() < end_time:
        tasks = []
        for _ in range(10):  # 模拟10个并发客户端
            query = random.choice(test_queries)
            passages = random.sample(test_passages, random.randint(5, 20))  # 每次随机选择5-20个段落
            tasks.append(asyncio.create_task(client(backend, query, passages, 1)))
            total_passages += len(passages)

        batch_latencies = await asyncio.gather(*tasks)
        latencies.extend(batch_latencies)
        request_count += len(tasks)

        current_memory = process.memory_info().rss / (1024 * 1024)  # 转换为MB
        memory_usage.append(current_memory)
        max_memory = max(max_memory, current_memory)

    logger.info(f"Processed {total_passages} passages in {request_count} requests")

    total_time = time.time() - start_time
    qps = total_passages / total_time
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    avg_memory = np.mean(memory_usage)

    await backend.close()  # 释放资源
    del backend  # 释放内存
    return {
        "batch_size": batch_size,
        "num_threads": num_threads,
        "qps": qps,
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
        "max_memory_mb": max_memory,
        "avg_memory_mb": avg_memory,
    }


async def main():
    results = []
    for batch_size in BATCH_SIZES:
        for num_threads in NUM_THREADS:
            logger.info(f"Testing with batch_size={batch_size}, num_threads={num_threads}")
            result = await run_test(batch_size, num_threads)
            results.append(result)
            logger.info(f"Results: {result}")

    print("\nTest Results:")
    print("-------------")
    for result in results:
        print(f"Batch Size: {result['batch_size']}, Threads: {result['num_threads']}")
        print(f"  QPS: {result['qps']:.2f}")
        print(f"  Avg Latency: {result['avg_latency'] * 1000:.2f} ms")
        print(f"  P95 Latency: {result['p95_latency'] * 1000:.2f} ms")
        print(f"  Max Memory: {result['max_memory_mb']:.2f} MB")
        print(f"  Avg Memory: {result['avg_memory_mb']:.2f} MB")
        print()

    best_qps = max(results, key=lambda x: x['qps'])
    best_latency = min(results, key=lambda x: x['avg_latency'])
    best_memory = min(results, key=lambda x: x['max_memory_mb'])

    print("Best Configurations:")
    print(
        f"Best QPS: Batch Size {best_qps['batch_size']}, Threads {best_qps['num_threads']} (QPS: {best_qps['qps']:.2f})")
    print(
        f"Best Latency: Batch Size {best_latency['batch_size']}, Threads {best_latency['num_threads']} (Avg Latency: {best_latency['avg_latency'] * 1000:.2f} ms)")
    print(
        f"Best Memory Usage: Batch Size {best_memory['batch_size']}, Threads {best_memory['num_threads']} (Max Memory: {best_memory['max_memory_mb']:.2f} MB)")


if __name__ == "__main__":
    asyncio.run(main())

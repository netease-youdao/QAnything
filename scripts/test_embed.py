import random
import psutil
import asyncio
import time
import numpy as np
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from transformers import AutoTokenizer
import logging
import string
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 测试配置
TEST_DURATION = 10  # 测试持续时间（秒）
BATCH_SIZES = [1, 2, 4, 8, 16]
NUM_THREADS = [1, 2, 4]
MODEL_PATH = "/root/models/linux_onnx/embedding_model_configs_v0.0.1/embed.onnx"  # 请替换为实际的模型路径

LOCAL_EMBED_MAX_LENGTH = 512
LOCAL_EMBED_PATH = "/root/models/linux_onnx/embedding_model_configs_v0.0.1"
LOCAL_EMBED_BATCH = 16


class EmbeddingAsyncBackend:
    def __init__(self, model_path, use_cpu=True, num_threads=4, batch_size=16):
        self.use_cpu = use_cpu
        self.return_tensors = "np"
        self.batch_size = batch_size
        sess_options = SessionOptions()
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        if use_cpu:
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.executor = ThreadPoolExecutor(max_workers=num_threads)

        self.session = InferenceSession(model_path, sess_options=sess_options, providers=providers)
        self._tokenizer = AutoTokenizer.from_pretrained(LOCAL_EMBED_PATH)  # 请根据实际使用的模型调整

        self.queue = asyncio.Queue()
        self.is_running = True
        asyncio.create_task(self.process_queue())

    def release_resources(self):
        self.is_running = False
        # 取消 process_queue 任务
        if hasattr(self, 'queue_task'):
            self.queue_task.cancel()

        # 关闭执行器
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

        # 清理主要的大型对象
        large_attributes = ['session', '_tokenizer', 'queue', 'executor']
        for attr in large_attributes:
            if hasattr(self, attr):
                setattr(self, attr, None)

        # 强制进行垃圾回收
        gc.collect()

    async def embed_documents_async(self, texts):
        futures = []
        # 设置mini_batch=1，每次处理1个文本
        mini_batch = 1
        for i in range(0, len(texts), mini_batch):
            future = asyncio.Future()
            futures.append(future)
            await self.queue.put((texts[i:i + mini_batch], future))

        results = await asyncio.gather(*futures)
        return [item for sublist in results for item in sublist]

    def embed_documents(self, texts):
        logger.info(f"embed_documents number: {len(texts)}")
        inputs_onnx = self._tokenizer(texts, padding=True, truncation=True, max_length=LOCAL_EMBED_MAX_LENGTH,
                                      return_tensors=self.return_tensors)
        inputs_onnx = {k: v for k, v in inputs_onnx.items()}

        # start_time = time.time()
        outputs_onnx = self.session.run(output_names=['output'], input_feed=inputs_onnx)
        # debug_logger.info(f"onnx infer time: {time.time() - start_time}")

        embedding = outputs_onnx[0][:, 0]
        # logger.info(f'embedding shape: {embedding.shape}')

        norm_arr = np.linalg.norm(embedding, axis=1, keepdims=True)
        embeddings_normalized = embedding / norm_arr

        return embeddings_normalized.tolist()

    async def process_queue(self):
        while self.is_running:

            if not hasattr(self, 'queue') or self.queue is None:
                # 如果队列不存在或为None，等待一段时间后退出
                await asyncio.sleep(0.1)
                return

            batch_texts = []
            futures = []

            try:
                while len(batch_texts) < self.batch_size:
                    texts, future = await asyncio.wait_for(self.queue.get(), timeout=0.01)
                    batch_texts.extend(texts)
                    futures.append((future, len(texts)))
            except asyncio.TimeoutError:
                pass

            # logger.info(f"process_queue embedding texts number: {len(batch_texts)}")
            if batch_texts:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(self.executor, self.embed_documents, batch_texts)

                start = 0
                for future, text_count in futures:
                    end = start + text_count
                    future.set_result(result[start:end])
                    start = end
            else:
                await asyncio.sleep(0.1)  # 如果没有文本要处理，短暂休眠


def generate_test_data(num_samples=1000, max_length=2048):
    # 创建一个包含所有可能字符的字符串
    all_chars = string.ascii_letters + string.digits  # 包含大小写字母和数字，共62个字符

    # 生成随机字符串的函数
    def generate_random_string(length):
        return ''.join(np.random.choice(list(all_chars), size=length))

    # 生成指定数量的随机长度字符串
    return [generate_random_string(np.random.randint(1, max_length + 1)) for _ in range(num_samples)]


# 模拟客户端请求
async def client(backend, texts, request_count):
    for _ in range(request_count):
        start_time = time.time()
        await backend.embed_documents_async(texts)
        latency = time.time() - start_time
        return latency


# 运行测试
async def run_test(batch_size, num_threads):
    backend = EmbeddingAsyncBackend(MODEL_PATH, use_cpu=True, num_threads=num_threads, batch_size=batch_size)
    test_data = generate_test_data()

    start_time = time.time()
    end_time = start_time + TEST_DURATION
    request_count = 0
    latencies = []

    process = psutil.Process()
    max_memory = 0
    memory_usage = []
    total_texts = 0

    while time.time() < end_time:
        tasks = []
        for _ in range(10):  # 模拟10个并发客户端
            texts = [test_data[np.random.randint(0, len(test_data))] for _ in
                     range(random.randint(batch_size, batch_size * 8))]
            tasks.append(asyncio.create_task(client(backend, texts, 1)))
            total_texts += len(texts)

        batch_latencies = await asyncio.gather(*tasks)
        latencies.extend(batch_latencies)
        request_count += len(tasks)
        # 更新最大内存使用
        current_memory = process.memory_info().rss / (1024 * 1024)  # 转换为MB
        memory_usage.append(current_memory)
        max_memory = max(max_memory, current_memory)

    logger.info(f"Processed {total_texts} texts in {request_count} requests")

    total_time = time.time() - start_time
    qps = total_texts / total_time
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    avg_memory = np.mean(memory_usage)

    backend.release_resources()  # 释放资源
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


# 主测试函数
async def main():
    results = []
    for batch_size in BATCH_SIZES:
        for num_threads in NUM_THREADS:
            logger.info(f"Testing with batch_size={batch_size}, num_threads={num_threads}")
            result = await run_test(batch_size, num_threads)
            results.append(result)
            logger.info(f"Results: {result}")

    # 输出结果
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

    # 找出最佳配置
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

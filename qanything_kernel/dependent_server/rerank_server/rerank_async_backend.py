from transformers import AutoTokenizer
from copy import deepcopy
from typing import List
import asyncio
from qanything_kernel.configs.model_config import LOCAL_RERANK_MAX_LENGTH, \
    LOCAL_RERANK_BATCH, LOCAL_RERANK_PATH
from qanything_kernel.utils.general_utils import get_time
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession
from concurrent.futures import ThreadPoolExecutor
from qanything_kernel.utils.custom_log import debug_logger
import numpy as np


class RerankAsyncBackend:
    def __init__(self, model_path, use_cpu=True, num_threads=4):
        self.use_cpu = use_cpu
        self.overlap_tokens = 80
        self.max_length = LOCAL_RERANK_MAX_LENGTH
        self.return_tensors = "np"
        # 创建一个ONNX Runtime会话设置，使用GPU执行
        sess_options = SessionOptions()
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        if use_cpu:
            providers = ['CPUExecutionProvider']
            self.executor = ThreadPoolExecutor(max_workers=num_threads)
            self.batch_size = LOCAL_RERANK_BATCH  # CPU批处理大小
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.executor = ThreadPoolExecutor(max_workers=num_threads)
            self.batch_size = 16  # GPU批处理大小固定为16

        self.session = InferenceSession(model_path, sess_options, providers=providers)
        self._tokenizer = AutoTokenizer.from_pretrained(LOCAL_RERANK_PATH)
        self.spe_id = self._tokenizer.sep_token_id

        self.queue = asyncio.Queue()
        asyncio.create_task(self.process_queue())

    @get_time
    def inference(self, batch):
        debug_logger.info(f"rerank shape: {batch['attention_mask'].shape}")
        # 准备输入数据
        inputs = {self.session.get_inputs()[0].name: batch['input_ids'],
                  self.session.get_inputs()[1].name: batch['attention_mask']}

        if 'token_type_ids' in batch:
            inputs[self.session.get_inputs()[2].name] = batch['token_type_ids']

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

    def tokenize_preproc(self,
                         query: str,
                         passages: List[str],
                         ):
        query_inputs = self._tokenizer.encode_plus(query, truncation=False, padding=False)
        max_passage_inputs_length = self.max_length - len(query_inputs['input_ids']) - 1
        assert max_passage_inputs_length > 10
        overlap_tokens = min(self.overlap_tokens, max_passage_inputs_length * 2 // 7)

        # 组[query, passage]对
        merge_inputs = []
        merge_inputs_idxs = []
        for pid, passage in enumerate(passages):
            passage_inputs = self._tokenizer.encode_plus(passage, truncation=False, padding=False,
                                                         add_special_tokens=False)
            passage_inputs_length = len(passage_inputs['input_ids'])

            if passage_inputs_length <= max_passage_inputs_length:
                if passage_inputs['attention_mask'] is None or len(passage_inputs['attention_mask']) == 0:
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
            batch = self._tokenizer.pad(
                tot_batches[i:i + mini_batch],
                padding=True,
                max_length=None,
                pad_to_multiple_of=None,
                return_tensors=self.return_tensors
            )
            future = asyncio.Future()
            futures.append(future)
            await self.queue.put((batch, future))

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
                    batch_items.append(batch)
                    futures.append(future)
            except asyncio.TimeoutError:
                pass

            if batch_items:
                loop = asyncio.get_running_loop()
                combined_batch = {
                    'input_ids': np.concatenate([item['input_ids'] for item in batch_items]),
                    'attention_mask': np.concatenate([item['attention_mask'] for item in batch_items])
                }
                if 'token_type_ids' in batch_items[0]:
                    combined_batch['token_type_ids'] = np.concatenate([item['token_type_ids'] for item in batch_items])

                result = await loop.run_in_executor(self.executor, self.inference, combined_batch)

                start = 0
                for future, batch in zip(futures, batch_items):
                    end = start + len(batch['input_ids'])
                    future.set_result(result[start:end])
                    start = end
            else:
                await asyncio.sleep(0.1)

    async def get_rerank(self, query: str, passages: List[str]):
        return await self.get_rerank_async(query, passages)

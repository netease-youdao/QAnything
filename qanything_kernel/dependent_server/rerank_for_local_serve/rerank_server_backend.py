import time
from transformers import AutoTokenizer
from copy import deepcopy
from typing import List
from tritonclient import grpc as grpcclient
from qanything_kernel.configs.model_config import LOCAL_RERANK_SERVICE_URL, LOCAL_RERANK_MAX_LENGTH, LOCAL_RERANK_MODEL_NAME, \
    LOCAL_RERANK_BATCH
import numpy as np


class LocalRerankBackend:
    def __init__(self):
        tokenizer_path = 'qanything_kernel/dependent_server/rerank_for_local_serve/reranker_model_yd_1225'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.overlap_tokens = 80
        self.spe_id = self.tokenizer.sep_token_id

        self.batch_size = LOCAL_RERANK_BATCH
        self.max_length = LOCAL_RERANK_MAX_LENGTH
        self.model_name = LOCAL_RERANK_MODEL_NAME
        # 创建Triton客户端实例
        self.triton_client = grpcclient.InferenceServerClient(url=LOCAL_RERANK_SERVICE_URL)

    def inference(self, serialized_inputs):
        # 准备输入数据
        inputs = []
        for input_name, data in serialized_inputs.items():
            infer_input = grpcclient.InferInput(input_name, data.shape, grpcclient.np_to_triton_dtype(data.dtype))
            infer_input.set_data_from_numpy(data)
            inputs.append(infer_input)

        # 准备输出
        outputs = []
        output_name = "logits"
        outputs.append(grpcclient.InferRequestedOutput(output_name))

        # 发送推理请求
        start_time = time.time()
        response = self.triton_client.infer(self.model_name, inputs, outputs=outputs)
        print('local rerank infer time: {} s'.format(time.time() - start_time), flush=True)

        # 获取响应数据
        result_data = response.as_numpy(output_name)
        print('rerank res:', result_data, flush=True)

        # 应用sigmoid函数
        sigmoid_scores = 1 / (1 + np.exp(-result_data))

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
        query_inputs = self.tokenizer.encode_plus(query, truncation=False, padding=False)
        max_passage_inputs_length = self.max_length - len(query_inputs['input_ids']) - 1
        assert max_passage_inputs_length > 10
        overlap_tokens = min(self.overlap_tokens, max_passage_inputs_length * 2 // 7)

        # 组[query, passage]对
        merge_inputs = []
        merge_inputs_idxs = []
        for pid, passage in enumerate(passages):
            passage_inputs = self.tokenizer.encode_plus(passage, truncation=False, padding=False,
                                                        add_special_tokens=False)
            passage_inputs_length = len(passage_inputs['input_ids'])

            if passage_inputs_length <= max_passage_inputs_length:
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

    def predict(self,
                query: str,
                passages: List[str],
                ):
        tot_batches, merge_inputs_idxs_sort = self.tokenize_preproc(query, passages)

        tot_scores = []
        for k in range(0, len(tot_batches), self.batch_size):
            batch = self.tokenizer.pad(
                tot_batches[k:k + self.batch_size],
                padding=True,
                max_length=None,
                pad_to_multiple_of=None,
                return_tensors="np"
            )
            scores = self.inference(batch)
            tot_scores.extend(scores)

        merge_tot_scores = [0 for _ in range(len(passages))]
        for pid, score in zip(merge_inputs_idxs_sort, tot_scores):
            merge_tot_scores[pid] = max(merge_tot_scores[pid], score)
        print("merge_tot_scores:", merge_tot_scores, flush=True)
        return merge_tot_scores

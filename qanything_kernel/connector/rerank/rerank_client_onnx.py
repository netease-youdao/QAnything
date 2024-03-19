import onnxruntime
import time
from transformers import AutoTokenizer
from .base import RerankBase
from copy import deepcopy
from typing import List
from qanything_kernel.configs.model_config import LOCAL_RERANK_MODEL_PATH, LOCAL_RERANK_MAX_LENGTH, LOCAL_RERANK_MODEL_NAME, \
    LOCAL_RERANK_BATCH, LOCAL_RERANK_PATH, LOCAL_RERANK_REPO 
from qanything_kernel.utils.custom_log import debug_logger
import numpy as np
# from huggingface_hub import snapshot_download
from modelscope import snapshot_download
import subprocess
import os

# 如果模型不存在, 下载模型
if not os.path.exists(LOCAL_RERANK_MODEL_PATH):
    # snapshot_download(repo_id=LOCAL_RERANK_REPO, local_dir=LOCAL_RERANK_PATH, local_dir_use_symlinks="auto")
    debug_logger.info(f"开始下载rerank模型：{LOCAL_RERANK_REPO}")
    cache_dir = snapshot_download(model_id=LOCAL_RERANK_REPO)
    # 如果存在的话，删除LOCAL_EMBED_PATH
    os.system(f"rm -rf {LOCAL_RERANK_PATH}")
    output = subprocess.check_output(['ln', '-s', cache_dir, LOCAL_RERANK_PATH], text=True)
    debug_logger.info(f"模型下载完毕！cache地址：{cache_dir}, 软链接地址：{LOCAL_RERANK_PATH}")


class RerankONNXBackend(RerankBase):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_RERANK_PATH)
        self.overlap_tokens = 80
        self.spe_id = self.tokenizer.sep_token_id

        self.batch_size = LOCAL_RERANK_BATCH
        self.max_length = LOCAL_RERANK_MAX_LENGTH
        self.model_name = LOCAL_RERANK_MODEL_NAME
        # 加载ONNX模型
        # 创建一个ONNX Runtime会话设置，使用GPU执行
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(LOCAL_RERANK_MODEL_PATH, sess_options, providers=providers)

    def inference(self, batch):
        # 准备输入数据
        inputs = {self.session.get_inputs()[0].name: batch['input_ids'],
                  self.session.get_inputs()[1].name: batch['attention_mask']}

        if 'token_type_ids' in batch:
            inputs[self.session.get_inputs()[2].name] = batch['token_type_ids']
        
        # 执行推理 输出为logits
        start_time = time.time()
        result = self.session.run(None, inputs)  # None表示获取所有输出
        debug_logger.info(f"rerank infer time: {time.time() - start_time}")
        # debug_logger.info(f"rerank result: {result}")
        
        # 应用sigmoid函数
        sigmoid_scores = 1 / (1 + np.exp(-np.array(result[0])))
        
        return sigmoid_scores.reshape(-1).tolist()

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

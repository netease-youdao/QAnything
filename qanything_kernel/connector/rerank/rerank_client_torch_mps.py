import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .base import RerankBase
from copy import deepcopy
from typing import List
from qanything_kernel.utils.custom_log import debug_logger
import numpy as np
from qanything_kernel.configs.model_config import LOCAL_RERANK_MAX_LENGTH
import os
import torch


class RerankTorchMPSBackend(RerankBase):
    def __init__(self):
        super().__init__()
        self.overlap_tokens = 80
        self.batch_size = 8
        self.max_length = LOCAL_RERANK_MAX_LENGTH
        self._get_model()

    def _get_model(self):
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(current_script_path + '/rerank_torch_models', exist_ok=True)
        self._tokenizer = AutoTokenizer.from_pretrained('maidalun1020/bce-reranker-base_v1',
                                                        cache_dir=current_script_path + '/rerank_torch_models')
        self._model = AutoModelForSequenceClassification.from_pretrained('maidalun1020/bce-reranker-base_v1',
                                                                         cache_dir=current_script_path + '/rerank_torch_models',
                                                                         return_dict=False)
        self._model = self._model.half().to('mps')
        self.spe_id = self._tokenizer.sep_token_id

    def inference(self, batch):
        # 准备输入数据
        inputs = {k: v.to('mps') for k, v in batch.items()}

        # 执行推理 输出为logits
        start_time = time.time()
        result = self._model(**inputs, return_dict=True)

        debug_logger.info(f"rerank infer time: {time.time() - start_time}")
        sigmoid_scores = torch.sigmoid(result.logits.view(-1, )).cpu().detach().numpy()

        return sigmoid_scores.tolist()

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
            batch = self._tokenizer.pad(
                tot_batches[k:k + self.batch_size],
                padding=True,
                max_length=None,
                pad_to_multiple_of=None,
                return_tensors="pt"
            )
            scores = self.inference(batch)
            tot_scores.extend(scores)

        merge_tot_scores = [0 for _ in range(len(passages))]
        for pid, score in zip(merge_inputs_idxs_sort, tot_scores):
            merge_tot_scores[pid] = max(merge_tot_scores[pid], score)
        print("merge_tot_scores:", merge_tot_scores, flush=True)
        return merge_tot_scores

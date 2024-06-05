from transformers import AutoTokenizer
from copy import deepcopy
from typing import List
from qanything_kernel.configs.model_config import LOCAL_RERANK_MODEL_PATH, LOCAL_RERANK_MAX_LENGTH, \
    LOCAL_RERANK_MODEL_NAME, \
    LOCAL_RERANK_BATCH, LOCAL_RERANK_PATH, LOCAL_RERANK_REPO, LOCAL_RERANK_WORKERS
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.utils.general_utils import get_time
from modelscope import snapshot_download
from abc import ABC, abstractmethod
import subprocess
import os
import concurrent.futures

# 如果模型不存在, 下载模型
if not os.path.exists(LOCAL_RERANK_MODEL_PATH):
    # snapshot_download(repo_id=LOCAL_RERANK_REPO, local_dir=LOCAL_RERANK_PATH, local_dir_use_symlinks="auto")
    debug_logger.info(f"开始下载rerank模型：{LOCAL_RERANK_REPO}")
    cache_dir = snapshot_download(model_id=LOCAL_RERANK_REPO)
    # 如果存在的话，删除LOCAL_EMBED_PATH
    os.system(f"rm -rf {LOCAL_RERANK_PATH}")
    output = subprocess.check_output(['ln', '-s', cache_dir, LOCAL_RERANK_PATH], text=True)
    debug_logger.info(f"模型下载完毕！cache地址：{cache_dir}, 软链接地址：{LOCAL_RERANK_PATH}")


class RerankBackend(ABC):
    def __init__(self, use_cpu):
        self.use_cpu = use_cpu
        self._tokenizer = AutoTokenizer.from_pretrained(LOCAL_RERANK_PATH)
        self.spe_id = self._tokenizer.sep_token_id
        self.overlap_tokens = 80
        self.batch_size = LOCAL_RERANK_BATCH
        self.max_length = LOCAL_RERANK_MAX_LENGTH
        self.return_tensors = None
        self.workers = LOCAL_RERANK_WORKERS
    
    @abstractmethod
    def inference(self, batch) -> List:
        pass
         
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

    @get_time
    def get_rerank(self, query: str, passages: List[str]):
        tot_batches, merge_inputs_idxs_sort = self.tokenize_preproc(query, passages)

        tot_scores = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for k in range(0, len(tot_batches), self.batch_size):
                batch = self._tokenizer.pad(
                    tot_batches[k:k + self.batch_size],
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=None,
                    return_tensors=self.return_tensors
                )
                future = executor.submit(self.inference, batch)
                futures.append(future)
            debug_logger.info(f'rerank number: {len(futures)}')
            for future in futures:
                scores = future.result()
                tot_scores.extend(scores)

        merge_tot_scores = [0 for _ in range(len(passages))]
        for pid, score in zip(merge_inputs_idxs_sort, tot_scores):
            merge_tot_scores[pid] = max(merge_tot_scores[pid], score)
        # print("merge_tot_scores:", merge_tot_scores, flush=True)
        return merge_tot_scores

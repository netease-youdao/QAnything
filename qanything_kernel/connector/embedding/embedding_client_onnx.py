import numpy as np
import time
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel 
from transformers import AutoTokenizer
from qanything_kernel.utils.custom_log import debug_logger
from .base import EmbeddingBase
from qanything_kernel.configs.model_config import LOCAL_EMBED_MODEL_PATH, LOCAL_EMBED_MAX_LENGTH, LOCAL_EMBED_BATCH, LOCAL_EMBED_PATH, LOCAL_EMBED_REPO
from modelscope import snapshot_download
import subprocess
import os

# 如果模型不存在, 下载模型
if not os.path.exists(LOCAL_EMBED_MODEL_PATH):
    # snapshot_download(repo_id=LOCAL_EMBED_REPO, local_dir=LOCAL_EMBED_PATH, local_dir_use_symlinks="auto")
    debug_logger.info(f"开始下载embedding模型：{LOCAL_EMBED_REPO}")
    cache_dir = snapshot_download(model_id=LOCAL_EMBED_REPO)
    # 如果存在的话，删除LOCAL_EMBED_PATH
    os.system(f"rm -rf {LOCAL_EMBED_PATH}")
    output = subprocess.check_output(['ln', '-s', cache_dir, LOCAL_EMBED_PATH], text=True)
    debug_logger.info(f"模型下载完毕！cache地址：{cache_dir}, 软链接地址：{LOCAL_EMBED_PATH}")

class EmbeddingClientONNX(EmbeddingBase):
    # embed_version = "local_v0.0.1_20230525_6d4019f1559aef84abc2ab8257e1ad4c"

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
    ):
        super().__init__()
        self.max_batchsz = LOCAL_EMBED_BATCH
        self._model_path = model_path
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        sess_options = SessionOptions()
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = InferenceSession(model_path, sess_options=sess_options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        debug_logger.info(f"EmbeddingClient: model_path: {model_path}, tokenizer_path: {tokenizer_path}")

    def get_embedding(self, sentences, max_length=512):
        inputs_onnx = self._tokenizer(sentences, padding=True, truncation=True, max_length=max_length, return_tensors='np')
        inputs_onnx = {k: v for k, v in inputs_onnx.items()}
        print(inputs_onnx['input_ids'].shape)
        print(inputs_onnx['attention_mask'].shape)
        start_time = time.time()
        outputs_onnx = self._session.run(output_names=['output'], input_feed=inputs_onnx)
        debug_logger.info(f"embedding infer time: {time.time() - start_time}")
        embedding = outputs_onnx[0][:,0]
        print(embedding.dtype)
        debug_logger.info(f'embedding shape: {embedding.shape}')
        norm_arr = np.linalg.norm(embedding, axis=1, keepdims=True)
        embeddings_normalized = embedding / norm_arr

        return embeddings_normalized.tolist()




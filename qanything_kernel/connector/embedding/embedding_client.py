import numpy as np
import time
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel 
from transformers import AutoTokenizer
from qanything_kernel.utils.custom_log import debug_logger


class EmbeddingClient:
    embed_version = "local_v0.0.1_20230525_6d4019f1559aef84abc2ab8257e1ad4c"

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
    ):
        self._model_path = model_path
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        sess_options = SessionOptions()
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = InferenceSession(model_path, sess_options=sess_options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        debug_logger.info(f"EmbeddingClient: model_path: {model_path}, tokenizer_path: {tokenizer_path}")

    def get_embedding(self, sentences, max_length=512):
        inputs_onnx = self._tokenizer(sentences, padding=True, truncation=True, max_length=max_length, return_tensors='np')
        inputs_onnx = {k: v for k, v in inputs_onnx.items()}
        start_time = time.time()
        outputs_onnx = self._session.run(output_names=['output'], input_feed=inputs_onnx)
        debug_logger.info(f"infer time: {time.time() - start_time}")
        embedding = outputs_onnx[0][:,0]
        debug_logger.info(f'embedding shape: {embedding.shape}')
        norm_arr = np.linalg.norm(embedding, axis=1, keepdims=True)
        embeddings_normalized = embedding / norm_arr

        return embeddings_normalized.tolist()
    
    def getModelVersion(self):
        return self.embed_version


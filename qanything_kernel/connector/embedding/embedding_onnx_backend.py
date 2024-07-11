import numpy as np
import time
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel 
from qanything_kernel.configs.model_config import LOCAL_EMBED_MODEL_PATH
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.connector.embedding.embedding_backend import EmbeddingBackend


class EmbeddingOnnxBackend(EmbeddingBackend):
    def __init__(self, use_cpu: bool = False):
        super().__init__(use_cpu)
        self.return_tensors = "np"
        sess_options = SessionOptions()
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        if use_cpu:
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self._session = InferenceSession(LOCAL_EMBED_MODEL_PATH, sess_options=sess_options, providers=providers)
        debug_logger.info(f"EmbeddingClient: model_path: {LOCAL_EMBED_MODEL_PATH}")

    def get_embedding(self, sentences, max_length):
        inputs_onnx = self._tokenizer(sentences, padding=True, truncation=True, max_length=max_length, return_tensors=self.return_tensors)
        debug_logger.info(f'embedding input shape: {inputs_onnx["input_ids"].shape}')
        inputs_onnx = {k: v for k, v in inputs_onnx.items()}
        start_time = time.time()
        outputs_onnx = self._session.run(output_names=['output'], input_feed=inputs_onnx)
        debug_logger.info(f"onnx infer time: {time.time() - start_time}")
        embedding = outputs_onnx[0][:,0]
        debug_logger.info(f'embedding shape: {embedding.shape}')
        norm_arr = np.linalg.norm(embedding, axis=1, keepdims=True)
        embeddings_normalized = embedding / norm_arr

        return embeddings_normalized.tolist()

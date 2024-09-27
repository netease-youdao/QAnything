import numpy as np
import time
from typing import List, Union
from numpy import ndarray
import torch
from torch import Tensor
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from qanything_kernel.configs.model_config import LOCAL_EMBED_MODEL_PATH, LOCAL_EMBED_PATH, LOCAL_EMBED_BATCH, LOCAL_RERANK_MAX_LENGTH
from qanything_kernel.utils.custom_log import debug_logger
from transformers import AutoTokenizer
from qanything_kernel.dependent_server.embedding_server.embedding_backend import EmbeddingBackend


class EmbeddingOnnxBackend:
    def __init__(self, use_cpu: bool = False):
        self._tokenizer = AutoTokenizer.from_pretrained(LOCAL_EMBED_PATH)
        self.return_tensors = "np"
        self.batch_size = LOCAL_EMBED_BATCH
        self.max_length = LOCAL_RERANK_MAX_LENGTH
        sess_options = SessionOptions()
        sess_options.intra_op_num_threads = 0
        sess_options.inter_op_num_threads = 0
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        if use_cpu:
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self._session = InferenceSession(LOCAL_EMBED_MODEL_PATH, sess_options=sess_options, providers=providers)
        debug_logger.info(f"EmbeddingClient: model_path: {LOCAL_EMBED_MODEL_PATH}")

    def get_embedding(self, sentences, max_length):
        inputs_onnx = self._tokenizer(sentences, padding=True, truncation=True, max_length=max_length, return_tensors=self.return_tensors)
        inputs_onnx = {k: v for k, v in inputs_onnx.items()}
        start_time = time.time()
        outputs_onnx = self._session.run(output_names=['output'], input_feed=inputs_onnx)
        debug_logger.info(f"onnx infer time: {time.time() - start_time}")
        embedding = outputs_onnx[0][:,0]
        debug_logger.info(f'embedding shape: {embedding.shape}')
        norm_arr = np.linalg.norm(embedding, axis=1, keepdims=True)
        embeddings_normalized = embedding / norm_arr

        return embeddings_normalized.tolist()

    def inference(self, inputs):
        outputs_onnx = None
        try_num = 2
        while outputs_onnx is None and try_num > 0:
            try:
                io_binding = self._session.io_binding()
                for k, v in inputs.items():
                    io_binding.bind_cpu_input(k, v)
                io_binding.synchronize_inputs()
                io_binding.bind_output('output')

                self._session.run_with_iobinding(io_binding)

                io_binding.synchronize_outputs()
                outputs_onnx = io_binding.copy_outputs_to_cpu()
                io_binding.clear_binding_inputs()
                io_binding.clear_binding_outputs()
            except:
                outputs_onnx = None
            try_num -= 1

        return outputs_onnx

    def encode(self, sentence: Union[str, List[str]],
               return_numpy: bool = False,
               normalize_to_unit: bool = True,
               keepdim: bool = True,
               batch_size: int = 64,
               max_length: int = 384,
               tokenizer=None,
               return_tokens_num=False,
               return_time_log=False) -> Union[ndarray, Tensor]:

        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = []

        tokens_num = 0
        using_time_tokenizer = 0
        using_time_model = 0

        total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
        for batch_id in range(total_batch):
            start_time_tokenizer = time.time()
            if tokenizer is not None:
                inputs = tokenizer(
                    sentence[batch_id * batch_size:(batch_id + 1) * batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="np"
                )
            else:
                inputs = self._tokenizer(
                    sentence[batch_id * batch_size:(batch_id + 1) * batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="np"
                )
            using_time_tokenizer += (time.time() - start_time_tokenizer)
            if return_tokens_num:
                tokens_num += (inputs['attention_mask'].sum().item() - 2 * inputs['attention_mask'].shape[0])

            inputs = {k: v for k, v in inputs.items()}

            start_time_model = time.time()
            outputs_onnx = self.inference(inputs)
            using_time_model += (time.time() - start_time_model)

            embeddings = np.asarray(outputs_onnx[0][:, 0])
            if normalize_to_unit:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            embedding_list.append(embeddings)

        embeddings = np.concatenate(embedding_list, axis=0)

        if single_sentence and not keepdim:
            embeddings = embeddings[0]

        if not return_numpy and isinstance(embeddings, ndarray):
            embeddings = torch.from_numpy(embeddings)

        if return_tokens_num and return_time_log:
            return embeddings, tokens_num, using_time_tokenizer, using_time_model
        elif return_tokens_num:
            return embeddings, tokens_num
        elif return_time_log:
            return embeddings, using_time_tokenizer, using_time_model
        else:
            return embeddings

    def predict(self, queries, return_tokens_num=False):
        embeddings = self.encode(
            queries, batch_size=self.batch_size, normalize_to_unit=True, return_numpy=True, max_length=self.max_length,
            tokenizer=self._tokenizer,
            return_tokens_num=return_tokens_num
        )

        return embeddings.tolist()
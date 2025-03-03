import traceback
import numpy as np
import time
from typing import List, Union
from numpy import ndarray
import torch
from torch import Tensor
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from qanything_kernel.configs.model_config import LOCAL_EMBED_MODEL_PATH, LOCAL_EMBED_PATH, LOCAL_EMBED_BATCH, LOCAL_EMBED_MAX_LENGTH
from qanything_kernel.utils.custom_log import embed_logger
from transformers import AutoTokenizer, PretrainedConfig


class EmbeddingOnnxBackend:
    def __init__(self, use_cpu: bool = False):
        # 使用 Jina 的分词器和配置
        self.use_cpu = use_cpu
        self._tokenizer = AutoTokenizer.from_pretrained(LOCAL_EMBED_PATH, local_files_only=True)
        self._config = PretrainedConfig.from_pretrained(LOCAL_EMBED_PATH, local_files_only=True)
        self.return_tensors = "np"
        self.batch_size = LOCAL_EMBED_BATCH
        self.max_length = LOCAL_EMBED_MAX_LENGTH
        self.default_task_type = 'text-matching'  # 可以根据需求修改
        self.task_id = np.array(self._config.lora_adaptations.index(self.default_task_type), dtype=np.int64)
        self.io_binding = None

        sess_options = SessionOptions()
        sess_options.intra_op_num_threads = 0
        sess_options.inter_op_num_threads = 0
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        if use_cpu:
            providers = ['CPUExecutionProvider']
            provider_options = [
                {
                    'intra_op_num_threads': 4,  # 根据你的 CPU 核心数调整
                }
            ]
            sess_options.enable_cpu_mem_arena = True
            sess_options.intra_op_num_threads = 4  # 根据你的 CPU 核心数调整
        else:
            providers = ['CUDAExecutionProvider']
            provider_options = [
                {
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB GPU 内存限制
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True
                }
            ]
            sess_options.enable_mem_pattern = True
            sess_options.enable_mem_reuse = True

        # 加载 Jina 的 ONNX 模型
        self._session = InferenceSession(LOCAL_EMBED_MODEL_PATH, sess_options=sess_options,
                                         providers=providers, provider_options=provider_options)
        if not use_cpu:
            self.io_binding = self._session.io_binding()
        # debug_logger.info(f"EmbeddingClient: model_path: {LOCAL_EMBED_MODEL_PATH}")

    def inference(self, inputs):
        output_info = self._session.get_outputs()
        output_name = output_info[0].name

        outputs_onnx = None
        try_num = 2
        while outputs_onnx is None and try_num > 0:
            try:
                if self.use_cpu:
                    embed_logger.info(f"Using CPU to run inference")
                    outputs_onnx = self._session.run([output_name], inputs)
                else:
                    embed_logger.info(f"Using GPU to run inference")
                    io_binding = self.io_binding  # 重用 io_binding
                    io_binding.clear_binding_inputs()  # 清理上次的输入
                    io_binding.clear_binding_outputs()  # 清理上次的输出

                    for k, v in inputs.items():
                        io_binding.bind_cpu_input(k, v)
                    io_binding.synchronize_inputs()
                    io_binding.bind_output(output_name)

                    self._session.run_with_iobinding(io_binding)

                    io_binding.synchronize_outputs()
                    outputs_onnx = io_binding.copy_outputs_to_cpu()
            except:
                embed_logger.error(f'Inference failed {traceback.format_exc()}, retrying...')
                outputs_onnx = None
            try_num -= 1
        return outputs_onnx

    def encode(self, sentence: Union[str, List[str]],
               return_numpy: bool = False,
               normalize_to_unit: bool = True,
               keepdim: bool = True,
               batch_size: int = 64,
               max_length: int = 384,
               truncate_dim: int = 768,
               tokenizer=None,
               return_tokens_num=False,
               return_time_log=False,
               task_type=None) -> Union[ndarray, Tensor]:

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
                input_text = tokenizer(
                    sentence[batch_id * batch_size:(batch_id + 1) * batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="np"
                )
            else:
                input_text = self._tokenizer(
                    sentence[batch_id * batch_size:(batch_id + 1) * batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="np"
                )
            if task_type is None:
                task_type = self.default_task_type
            task_id = np.array(self._config.lora_adaptations.index(task_type), dtype=np.int64)
            inputs = {
                'input_ids': input_text['input_ids'],
                'attention_mask': input_text['attention_mask'],
                'task_id': task_id,
            }
            # 打印输入的形状和 query 的总长度
            embed_logger.info(f"query num: {len(sentence)}, max_length: {max_length}, batch_size: {batch_size}")
            embed_logger.info(f"input shape: {inputs['input_ids'].shape}, query total char_lens: {inputs['attention_mask'].sum()}")
            using_time_tokenizer += (time.time() - start_time_tokenizer)
            if return_tokens_num:
                tokens_num += (inputs['attention_mask'].sum().item() - 2 * inputs['attention_mask'].shape[0])

            # 添加 task_id 到输入中
            if task_type is None:
                task_type = self.default_task_type

            start_time_model = time.time()
            embed_logger.info(f"task_type: {task_type}")
            outputs_onnx = self.inference(inputs)
            using_time_model += (time.time() - start_time_model)

            embeddings = np.asarray(outputs_onnx[0][:, 0])
            if normalize_to_unit:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            # 如果指定了 truncate_dim，则对嵌入向量进行截断
            if truncate_dim is not None:
                embeddings = embeddings[:, :truncate_dim]

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

    def predict(self, queries, task_type=None, return_tokens_num=False):
        embeddings = self.encode(
            queries, batch_size=self.batch_size, normalize_to_unit=True, return_numpy=True, max_length=self.max_length,
            tokenizer=self._tokenizer, return_tokens_num=return_tokens_num, task_type=task_type
        )

        return embeddings.tolist()

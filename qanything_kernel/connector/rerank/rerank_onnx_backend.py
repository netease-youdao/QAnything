import onnxruntime
from qanything_kernel.connector.rerank.rerank_backend import RerankBackend
from qanything_kernel.configs.model_config import LOCAL_RERANK_MODEL_PATH
from qanything_kernel.utils.custom_log import debug_logger
import numpy as np


class RerankOnnxBackend(RerankBackend):
    def __init__(self, use_cpu: bool = False):
        super().__init__(use_cpu)
        self.return_tensors = "np"
        # 创建一个ONNX Runtime会话设置，使用GPU执行
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        if use_cpu:
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(LOCAL_RERANK_MODEL_PATH, sess_options, providers=providers)

    def inference(self, batch):
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

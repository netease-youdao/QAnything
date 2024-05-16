import time
from transformers import AutoModelForSequenceClassification
from qanything_kernel.connector.rerank.rerank_backend import RerankBackend
from qanything_kernel.configs.model_config import LOCAL_RERANK_PATH
from qanything_kernel.utils.custom_log import debug_logger
import torch


class RerankTorchBackend(RerankBackend):
    def __init__(self, use_cpu: bool = False):
        super().__init__(use_cpu)
        self.return_tensors = "pt"
        self._model = AutoModelForSequenceClassification.from_pretrained(LOCAL_RERANK_PATH,
                                                                         return_dict=False)
        # if use_cpu or not torch.backends.mps.is_available():
        #     self.device = torch.device('cpu')
        #     self._model = self._model.to(self.device)
        # else:
        #     self.device = torch.device('mps')
        #     self._model = self._model.to(self.device)
        self.device = torch.device('cpu')
        self._model = self._model.to(self.device)
        print("rerank device:", self.device)

    def inference(self, batch):
        # 准备输入数据
        inputs = {k: v.to(self.device) for k, v in batch.items()}

        # 执行推理 输出为logits
        start_time = time.time()
        result = self._model(**inputs, return_dict=True)

        debug_logger.info(f"rerank infer time: {time.time() - start_time}")
        sigmoid_scores = torch.sigmoid(result.logits.view(-1, )).cpu().detach().numpy()

        return sigmoid_scores.tolist()

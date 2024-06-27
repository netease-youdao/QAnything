import time
from transformers import AutoModelForSequenceClassification
from qanything_kernel.connector.rerank.rerank_backend import RerankBackend
from qanything_kernel.configs.model_config import LOCAL_RERANK_PATH
from qanything_kernel.utils.custom_log import debug_logger
import torch


class RerankTorchBackend(RerankBackend):
    def __init__(self, use_cpu: bool = False, device: str = "cpu"):
        super().__init__(use_cpu)
        self.return_tensors = "pt"
        self.device = device
        self._model = AutoModelForSequenceClassification.from_pretrained(LOCAL_RERANK_PATH,
                                                                         return_dict=False)
        if use_cpu:
            self.device = torch.device('cpu')
            self._model = self._model.to(self.device)
        elif 'cuda' in device:
            self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(self.device)
        elif 'npu' in device:
            import torch_npu
            self.device = device
            torch_npu.npu.set_device(self.device)
            self._model = self._model.to(self.device)
        print("rerank device:", self.device)

    def inference(self, batch):
        if 'npu' in self.device:
            import torch_npu
            torch_npu.npu.set_device(self.device)
        # 准备输入数据
        inputs = {k: v.to(self.device) for k, v in batch.items()}

        # 执行推理 输出为logits
        start_time = time.time()
        result = self._model(**inputs, return_dict=True)

        debug_logger.info(f"rerank infer time: {time.time() - start_time}")
        sigmoid_scores = torch.sigmoid(result.logits.view(-1, )).cpu().detach().numpy()

        return sigmoid_scores.tolist()

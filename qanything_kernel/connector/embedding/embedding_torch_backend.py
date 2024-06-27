import numpy as np
import time
from qanything_kernel.utils.custom_log import debug_logger
from transformers import AutoModel, AutoTokenizer
from qanything_kernel.connector.embedding.embedding_backend import EmbeddingBackend
from qanything_kernel.configs.model_config import LOCAL_EMBED_PATH
import torch


class EmbeddingTorchBackend(EmbeddingBackend):
    def __init__(self, use_cpu: bool = False, device: str = "cpu"):
        super().__init__(use_cpu)
        self.return_tensors = "pt"
        self.device = device
        self._model = AutoModel.from_pretrained(LOCAL_EMBED_PATH, return_dict=False)

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
        print("embedding device:", self.device)

    def get_embedding(self, sentences, max_length):
        if 'npu' in self.device:
            import torch_npu
            torch_npu.npu.set_device(self.device)
        inputs_pt = self._tokenizer(sentences, padding=True, truncation=True, max_length=max_length,
                                    return_tensors=self.return_tensors)
        inputs_pt = {k: v.to(self.device) for k, v in inputs_pt.items()}
        start_time = time.time()
        outputs_pt = self._model(**inputs_pt)
        debug_logger.info(f"torch embedding infer time: {time.time() - start_time}")
        embedding = outputs_pt[0][:, 0].cpu().detach().numpy()
        debug_logger.info(f'embedding shape: {embedding.shape}')
        norm_arr = np.linalg.norm(embedding, axis=1, keepdims=True)
        embeddings_normalized = embedding / norm_arr

        return embeddings_normalized.tolist()

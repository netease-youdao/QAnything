from PIL import Image
import numpy as np
import torch
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from torchvision import transforms
from qanything_kernel.configs.model_config import PDF_MODEL_PATH
import os

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
cls = {0: 'wired', 1: 'wireless'}


class TableCls():
    def __init__(self, device=torch.device("cpu")):
        sess_options = SessionOptions()
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        # 输出当前文件绝对路径
        cls_model = os.path.join(PDF_MODEL_PATH, 'checkpoints/table/table_cls_l.onnx')
        if device == torch.device("cuda"):
            self.table_cls = InferenceSession(cls_model, sess_options, providers=['CUDAExecutionProvider'])
        else:
            self.table_cls = InferenceSession(cls_model, sess_options, providers=['CPUExecutionProvider'])

    def process(self, image):
        img = Image.fromarray(np.uint8(image))
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        output = self.table_cls.run(None, {'input': img.numpy()})
        predict = torch.softmax(torch.from_numpy(output[0]), dim=1)
        predict_cla = torch.argmax(predict, dim=1).numpy()[0]
        return cls[predict_cla]

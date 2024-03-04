from sanic import Sanic, response
from paddleocr import PaddleOCR
import base64
import threading 
import numpy as np
from sanic.worker.manager import WorkerManager
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.configs.model_config import CUDA_DEVICE
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

WorkerManager.THRESHOLD = 6000

# 创建 Sanic 应用
app = Sanic("OCRService")

# 初始化 PaddleOCR 引擎
t1 = time.time()
ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True, show_log=False)
debug_logger.info("ocr engine init time: {}".format(time.time() - t1))


# # 定义 OCR API 路由
@app.route("/ocr", methods=["POST"])
async def ocr(request):
    # 获取上传的文件
    input = request.json
    img_file = input['img64']
    height = input['height']
    width = input['width']
    channels = input['channels']

    binary_data = base64.b64decode(img_file)
    img_array = np.frombuffer(binary_data, dtype=np.uint8).reshape((height, width, channels))
    debug_logger.info("shape: {}".format(img_array.shape))

    # 无文件上传，返回错误
    if not img_file:
        return response.json({'error': 'No file was uploaded.'}, status=400)

    # 调用 PaddleOCR 进行识别
    res = ocr_engine.ocr(img_array)
    debug_logger.info("ocr result: {}".format(res))

    # 返回识别结果
    return response.json({'results': res})

class OcrServer:
    def __init__(self):
        self._stop_event = threading.Event()
    
    def start(self):
        app.run(host='0.0.0.0', port=8010, workers=1, access_log=False)
    
    def stop(self):
        self._stop_event.set()

ocr_server = OcrServer()

@app.middleware('request')
async def check_stop_flag(request):
    if ocr_server._stop_event.is_set():
        app.stop()
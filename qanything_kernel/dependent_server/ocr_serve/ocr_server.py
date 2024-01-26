from sanic import Sanic, response
from paddleocr import PaddleOCR
import base64
import numpy as np
from sanic.worker.manager import WorkerManager
import argparse
import logging

WorkerManager.THRESHOLD = 6000
logging.basicConfig(level=logging.INFO, filename='ocr.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')


parser = argparse.ArgumentParser(description='启动 OCR 服务')
parser.add_argument('OCR_USE_GPU', type=str, nargs='?', default='True', help='是否使用 GPU (True/False)')

args = parser.parse_args()
use_gpu = args.OCR_USE_GPU.lower() in ('true', '1', 't', 'y', 'yes')

logging.info(f"OCR_USE_GPU parameter is set to {args.OCR_USE_GPU}, use_gpu={use_gpu}")

# 创建 Sanic 应用
app = Sanic("OCRService")

# 初始化 PaddleOCR 引擎
ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=use_gpu, show_log=False)


# 定义 OCR API 路由
@app.post("/ocr")
async def ocr_request(request):
    # 获取上传的文件
    input = request.json
    img_file = input['img64']
    height = input['height']
    width = input['width']
    channels = input['channels']

    binary_data = base64.b64decode(img_file)
    img_array = np.frombuffer(binary_data, dtype=np.uint8).reshape((height, width, channels))
    logging.info("shape: {}".format(img_array.shape))

    # 无文件上传，返回错误
    if not img_file:
        return response.json({'error': 'No file was uploaded.'}, status=400)

    # 调用 PaddleOCR 进行识别
    res = ocr_engine.ocr(img_array)
    logging.info("ocr result: {}".format(res))

    # 返回识别结果
    return response.json({'results': res})


# 启动服务
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8010, workers=4, access_log=False)

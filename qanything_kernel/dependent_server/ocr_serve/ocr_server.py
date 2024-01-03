from sanic import Sanic, response
from paddleocr import PaddleOCR
import sys

# 创建 Sanic 应用
app = Sanic("OCRService")

# 初始化 PaddleOCR 引擎
ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True, show_log=False)

# 定义 OCR API 路由
@app.post("/ocr")
async def ocr_request(request):
    # 获取上传的文件
    img_file = request.files.get('file')
    
    # 无文件上传，返回错误
    if not img_file:
        return response.json({'error': 'No file was uploaded.'}, status=400)

    # 调用 PaddleOCR 进行识别
    res = ocr_engine.ocr(img_file.body)
    
    # 返回识别结果
    return response.json({'results': res})

if __name__ == "__main__":
    # 运行 Sanic 服务，使用 4 个 workers
    app.run(host="0.0.0.0", port=8010, workers=4)

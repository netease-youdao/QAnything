import sys
import os

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))

sys.path.append(root_dir)
print(root_dir)


from qanything_kernel.utils.general_utils import safe_get
from sanic import Sanic, response
from sanic.request import Request
from sanic.response import json
from qanything_kernel.dependent_server.pdf_parser_server.pdf_parser_backend import PdfLoader
import time
import torch
import argparse

# 接收外部参数mode
parser = argparse.ArgumentParser()
# mode必须是local或online
parser.add_argument('--use_gpu', action="store_true", help='use gpu or not')
parser.add_argument('--workers', type=int, default=1, help='workers')
# 检查是否是local或online，不是则报错
args = parser.parse_args()
print("args:", args)


app = Sanic("pdf_parser_server")


@app.before_server_start
async def init_pdf_parser(app, loop):
    start = time.time()
    app.ctx.pdf_parser = PdfLoader(device=torch.device('cpu') if not args.use_gpu else torch.device('cuda'))
    end = time.time()
    print(f'init pdf_parser cost {end - start}s', flush=True)


@app.post("/pdfparser")
async def pdf_parser(request: Request):
    filename = safe_get(request, 'filename')
    save_dir = safe_get(request, 'save_dir')

    pdf_parser_: PdfLoader = request.app.ctx.pdf_parser
    markdown_file = pdf_parser_.load_to_markdown(filename, save_dir)

    return json({"markdown_file": markdown_file})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9009, workers=args.workers)

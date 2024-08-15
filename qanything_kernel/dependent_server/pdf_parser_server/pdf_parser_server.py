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


app = Sanic("pdf_parser_server")

@app.post("/pdfparser")
async def pdf_parser(request: Request):
    filename = safe_get(request, 'filename')
    save_dir = safe_get(request, 'save_dir')

    loader = PdfLoader(filename=filename, save_dir=save_dir)
    markdown_file = loader.load_to_markdown()

    return json({"markdown_file": markdown_file})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9009, workers=1)

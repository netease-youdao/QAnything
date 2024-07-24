import sys
import os
import platform

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))

sys.path.append(root_dir)
print(root_dir)

from sanic import Sanic
from sanic.response import json
from qanything_kernel.dependent_server.rerank_server.rerank_onnx_backend import RerankOnnxBackend

app = Sanic("rerank_server")


@app.route("/rerank", methods=["POST"])
async def rerank(request):
    data = request.json
    query = data.get('query')
    passages = data.get('passages')
    rerank_backend: app.ctx.rerank_backend = request.app.ctx.rerank_backend
    print("local rerank query:", query, flush=True)
    print("local rerank passages number:", len(passages), flush=True)
    result_data = rerank_backend.get_rerank(query, passages)

    return json(result_data)


@app.before_server_start
async def init_local_doc_qa(app, loop):
    rerank_backend = RerankOnnxBackend(use_cpu=False)
    app.ctx.rerank_backend = rerank_backend


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, workers=1)

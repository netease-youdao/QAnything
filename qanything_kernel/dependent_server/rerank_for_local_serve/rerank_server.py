import sys
import os

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))

sys.path.append(root_dir)
print(root_dir)

from sanic import Sanic
from sanic.response import json
from qanything_kernel.dependent_server.rerank_for_local_serve.rerank_server_backend import LocalRerankBackend

app = Sanic("rerank_server")


@app.route("/rerank", methods=["POST"])
async def rerank(request):
    query = request.json.get("query")
    passgaes = request.json.get("passages")
    local_rerank_backend: LocalRerankBackend = request.app.ctx.local_rerank_backend
    print("local rerank query:", query, flush=True)
    print("local rerank passages number:", len(passgaes), flush=True)
    print("local rerank passages:", passgaes, flush=True)
    result_data = local_rerank_backend.predict(query, passgaes)

    return json(result_data)


@app.before_server_start
async def init_local_doc_qa(app, loop):
    app.ctx.local_rerank_backend = LocalRerankBackend()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8776, workers=4)

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
from qanything_kernel.dependent_server.rerank_server.rerank_async_backend import RerankAsyncBackend
from qanything_kernel.configs.model_config import LOCAL_RERANK_MODEL_PATH, LOCAL_RERANK_THREADS
from qanything_kernel.utils.general_utils import get_time_async

app = Sanic("rerank_server")


@get_time_async
@app.route("/rerank", methods=["POST"])
async def rerank(request):
    data = request.json
    query = data.get('query')
    passages = data.get('passages')
    onnx_backend: RerankAsyncBackend = request.app.ctx.onnx_backend
    result_data = await onnx_backend.get_rerank_async(query, passages)
    # print("local rerank query:", query, flush=True)
    # print("local rerank passages number:", len(passages), flush=True)

    return json(result_data)


@app.listener('before_server_start')
async def setup_onnx_backend(app, loop):
    app.ctx.onnx_backend = RerankAsyncBackend(model_path=LOCAL_RERANK_MODEL_PATH, use_cpu=True,
                                              num_threads=LOCAL_RERANK_THREADS)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, workers=1)

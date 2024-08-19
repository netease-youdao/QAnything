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
from qanything_kernel.dependent_server.embedding_server.embedding_async_backend import EmbeddingAsyncBackend
from qanything_kernel.configs.model_config import LOCAL_EMBED_MODEL_PATH, LOCAL_EMBED_THREADS
from qanything_kernel.utils.general_utils import get_time_async
import argparse

# 接收外部参数mode
parser = argparse.ArgumentParser()
# mode必须是local或online
parser.add_argument('--use_gpu', action="store_true", help='use gpu or not')
parser.add_argument('--workers', type=int, default=1, help='workers')
# 检查是否是local或online，不是则报错
args = parser.parse_args()
print("args:", args)

app = Sanic("embedding_server")


@get_time_async
@app.route("/embedding", methods=["POST"])
async def embedding(request):
    data = request.json
    texts = data.get('texts')
    # print("local embedding texts number:", len(texts), flush=True)

    onnx_backend: EmbeddingAsyncBackend = request.app.ctx.onnx_backend
    result_data = await onnx_backend.embed_documents_async(texts)
    # print("local embedding result number:", len(result_data), flush=True)
    # print("local embedding result:", result_data, flush=True)

    return json(result_data)


@app.listener('before_server_start')
async def setup_onnx_backend(app, loop):
    app.ctx.onnx_backend = EmbeddingAsyncBackend(model_path=LOCAL_EMBED_MODEL_PATH, 
                                                 use_cpu=not args.use_gpu, num_threads=LOCAL_EMBED_THREADS)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9001, workers=args.workers)

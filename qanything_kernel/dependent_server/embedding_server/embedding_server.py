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
from qanything_kernel.dependent_server.embedding_server.embedding_onnx_backend import EmbeddingOnnxBackend
from qanything_kernel.dependent_server.embedding_server.embedding_async_backend import EmbeddingAsyncBackend
from qanything_kernel.configs.model_config import LOCAL_EMBED_MODEL_PATH, LOCAL_EMBED_WORKERS
from qanything_kernel.utils.general_utils import get_time_async

app = Sanic("embedding_server")


@get_time_async
@app.route("/embedding", methods=["POST"])
async def embedding(request):
    data = request.json
    texts = data.get('texts')
    print("local embedding texts number:", len(texts), flush=True)

    onnx_backend: EmbeddingAsyncBackend = request.app.ctx.onnx_backend
    result_data = await onnx_backend.embed_documents_async(texts)
    # print("local embedding result number:", len(result_data), flush=True)
    # print("local embedding result:", result_data, flush=True)

    return json({"embeddings": result_data})


# @app.before_server_start
# async def init_local_doc_qa(app, loop):
#     embedding_backend = EmbeddingOnnxBackend(use_cpu=False)
#     app.ctx.embedding_backend = embedding_backend


@app.listener('before_server_start')
async def setup_onnx_backend(app, loop):
    app.ctx.onnx_backend = EmbeddingAsyncBackend(model_path=LOCAL_EMBED_MODEL_PATH, use_cpu=True, num_threads=LOCAL_EMBED_WORKERS)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9001, workers=1)

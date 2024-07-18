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
# from qanything_kernel.dependent_server.embedding_server.embedding_backend import EmbeddingBackend
from qanything_kernel.utils.general_utils import safe_get

app = Sanic("embedding_server")


@app.route("/embedding", methods=["POST"])
async def rerank(request):
    texts = safe_get(request, "texts")
    embedding_backend: app.ctx.embedding_backend = request.app.ctx.embedding_backend
    print("local embedding texts number:", len(texts), flush=True)
    result_data = embedding_backend.embed_documents(texts)

    return json({"embeddings": result_data, "model_version": embedding_backend.embed_version})


@app.before_server_start
async def init_local_doc_qa(app, loop):
    # if platform.system() == 'Linux':
    #     from qanything_kernel.dependent_server.embedding_server.embedding_onnx_backend import EmbeddingOnnxBackend
    #     embeddingBackend = EmbeddingOnnxBackend(use_cpu=False)
    #
    # else:
    #     from qanything_kernel.dependent_server.embedding_server.embedding_torch_backend import EmbeddingTorchBackend
    #     embeddingBackend = EmbeddingTorchBackend(use_cpu=False)
    from qanything_kernel.dependent_server.embedding_server.embedding_torch_backend import EmbeddingTorchBackend
    embeddingBackend = EmbeddingTorchBackend(use_cpu=False)
    app.ctx.embedding_backend = embeddingBackend


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9001, workers=2)

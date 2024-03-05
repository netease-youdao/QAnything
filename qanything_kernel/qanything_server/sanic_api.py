import sys
import os

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 获取当前脚本的父目录的路径，即`qanything_server`目录
current_dir = os.path.dirname(current_script_path)

# 获取`qanything_server`目录的父目录，即`qanything_kernel`
parent_dir = os.path.dirname(current_dir)

# 获取根目录：`qanything_kernel`的父目录
root_dir = os.path.dirname(parent_dir)

# 将项目根目录添加到sys.path
sys.path.append(root_dir)

from qanything_kernel.configs.model_config import MILVUS_LITE_LOCATION, CUDA_DEVICE, VM_3B_MODEL, VM_7B_MODEL 
from milvus import default_server
from .handler import *
from qanything_kernel.core.local_doc_qa import LocalDocQA
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.utils.general_utils import download_file, check_onnx_version
from sanic import Sanic
from sanic import response as sanic_response
from argparse import ArgumentParser, Action
from sanic.worker.manager import WorkerManager
import signal
from vllm.engine.arg_utils import AsyncEngineArgs
import time
import requests
import platform
import torch
import math 

parser = ArgumentParser()
parser = AsyncEngineArgs.add_cli_args(parser)
parser.add_argument('--host', dest='host', default='0.0.0.0', help='set host for qanything server')
parser.add_argument('--port', dest='port', default=8777, type=int, help='set port for qanything server')
#  必填参数
parser.add_argument('--model_size', dest='model_size', default=
'3B', help='set LLM model size for qanything server')
# --model "GeneZC/MiniChat-2-3B"
# --gpu-memory-utilization 0.5

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

WorkerManager.THRESHOLD = 6000

app = Sanic("QAnything")
# 设置请求体最大为 400MB
app.config.REQUEST_MAX_SIZE = 400 * 1024 * 1024


# 将 /static 路径映射到 static 文件夹
app.static('/static', './static')

# 启动Milvus Lite服务
@app.main_process_start
async def start_dependent_services(app, loop):
    debug_logger.info(f"default_server: {default_server.running}")
    if not default_server.running:
        start = time.time() 
        default_server.set_base_dir(MILVUS_LITE_LOCATION)
        default_server.start()
        print(f"Milvus Lite started at {default_server.listen_port}", flush=True)
        debug_logger.info(f"Milvus Lite started at {default_server.listen_port} in {time.time() - start} seconds.")


# 关闭依赖的服务
@app.main_process_stop
async def end_dependent_services(app, loop):
    if default_server.running:
        default_server.stop()


# CORS中间件，用于在每个响应中添加必要的头信息
@app.middleware("response")
async def add_cors_headers(request, response):
    # response.headers["Access-Control-Allow-Origin"] = "http://10.234.10.144:5052"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"  # 如果需要的话


@app.middleware("request")
async def handle_options_request(request):
    if request.method == "OPTIONS":
        headers = {
            # "Access-Control-Allow-Origin": "http://10.234.10.144:5052",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Allow-Credentials": "true"  # 如果需要的话
        }
        return sanic_response.text("", headers=headers)

@app.before_server_start
async def init_local_doc_qa(app, loop):
    start = time.time()
    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(mode='local', parser=parser)
    debug_logger.info(f"LocalDocQA started in {time.time() - start} seconds.")
    app.ctx.local_doc_qa = local_doc_qa

# @app.after_server_stop
# async def close_milvus_lite(app, loop):
#     if default_server.running:
#         default_server.stop()


app.add_route(document, "/api/docs", methods=['GET'])
app.add_route(new_knowledge_base, "/api/local_doc_qa/new_knowledge_base", methods=['POST'])  # tags=["新建知识库"]
app.add_route(upload_weblink, "/api/local_doc_qa/upload_weblink", methods=['POST'])  # tags=["上传网页链接"]
app.add_route(upload_files, "/api/local_doc_qa/upload_files", methods=['POST'])  # tags=["上传文件"] 
app.add_route(local_doc_chat, "/api/local_doc_qa/local_doc_chat", methods=['POST'])  # tags=["问答接口"] 
app.add_route(list_kbs, "/api/local_doc_qa/list_knowledge_base", methods=['POST'])  # tags=["知识库列表"] 
app.add_route(list_docs, "/api/local_doc_qa/list_files", methods=['POST'])  # tags=["文件列表"]
app.add_route(get_total_status, "/api/local_doc_qa/get_total_status", methods=['POST'])  # tags=["获取所有知识库状态"]
app.add_route(clean_files_by_status, "/api/local_doc_qa/clean_files_by_status", methods=['POST'])  # tags=["清理数据库"]
app.add_route(delete_docs, "/api/local_doc_qa/delete_files", methods=['POST'])  # tags=["删除文件"] 
app.add_route(delete_knowledge_base, "/api/local_doc_qa/delete_knowledge_base", methods=['POST'])  # tags=["删除知识库"] 
app.add_route(rename_knowledge_base, "/api/local_doc_qa/rename_knowledge_base", methods=['POST'])  # tags=["重命名知识库"] 

@app.route('/stop', methods=['GET'])
async def stop(request):
    if default_server.running:
        default_server.stop()
    request.app.stop()
    return sanic_response.text("Server is stopping.")

class LocalDocQAServer:
    def __init__(self, host='0.0.0.0', port=8777):
        self.host = host
        self.port = port

    def start(self):
        app.run(host=self.host, port=self.port, single_process=True, access_log=False)

    def stop(self):
        res = requests.get('http://{self.host}:{self.port}/stop'.format(self.host, self.port))
        debug_logger.info(f"Stop qanything server: {res.text}")


def get_gpu_memory_utilization(model_size):
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available: torch.cuda.is_available(): return False")
    gpu_memory = torch.cuda.get_device_properties(int(CUDA_DEVICE)).total_memory
    gpu_memory_in_GB = math.ceil(gpu_memory / (1024 ** 3))  # 将字节转换为GB
    debug_logger.info(f"GPU memory: {gpu_memory_in_GB} GB")
    if model_size == '3B':
        if gpu_memory_in_GB < 10:  # 显存最低需要10GB
            raise ValueError(f"GPU memory is not enough: {gpu_memory_in_GB} GB, at least 10GB is required with 3B Model.")
        gpu_memory_utilization = round(7 / gpu_memory_in_GB, 2)
    elif model_size == '7B':
        if gpu_memory_in_GB < 20:  # 显存最低需要20GB
            raise ValueError(f"GPU memory is not enough: {gpu_memory_in_GB} GB, at least 20GB is required with 7B Model.")
        gpu_memory_utilization = round(16 / gpu_memory_in_GB, 2)
    else:
        raise ValueError(f"Unsupported model size: {model_size}, supported model size: 3B, 7B")
    return gpu_memory_utilization


def main():
    debug_logger.info(f"default_server: {default_server.running}")
    if not default_server.running:
        start = time.time() 
        default_server.set_base_dir(MILVUS_LITE_LOCATION)
        default_server.start()
        debug_logger.info(f"Milvus Lite started at {default_server.listen_port} in {time.time() - start} seconds.")


    args = parser.parse_args()
    model_size = args.model_size
    
    args.gpu_memory_utilization = get_gpu_memory_utilization(model_size)
    if model_size == '3B':
        args.model = VM_3B_MODEL
    elif model_size == '7B':
        args.model = VM_7B_MODEL
    else:
        raise ValueError(f"Unsupported model size: {model_size}, supported model size: 3B, 7B")
    
    # 根据命令行参数启动服务器
    qanything_server = LocalDocQAServer(host=args.host, port=args.port)

    signal.signal(signal.SIGINT, lambda sig, frame: qanything_server.stop())
    signal.signal(signal.SIGTERM, lambda sig, frame: qanything_server.stop())

    try:
        qanything_server.start()
    except TimeoutError:
        print('Wait for qanything server started timeout.')
    except RuntimeError:
        print('QAnything server already stopped.')

if __name__ == "__main__":
    main()


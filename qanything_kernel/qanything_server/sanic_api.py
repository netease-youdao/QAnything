import sys
import os
import time
import webbrowser

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

from qanything_kernel.configs.model_config import DT_7B_MODEL_PATH, \
    DT_7B_DOWNLOAD_PARAMS, DT_3B_MODEL_PATH, DT_3B_DOWNLOAD_PARAMS, PDF_MODEL_PATH
import qanything_kernel.configs.model_config as model_config
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.utils.general_utils import download_file, get_gpu_memory_utilization, check_package_version
import torch
import platform
from argparse import ArgumentParser

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os_system = platform.system()
parser = ArgumentParser()

from sanic import Sanic
from sanic import response as sanic_response
from sanic.worker.manager import WorkerManager
import signal
import requests
from modelscope import snapshot_download
from modelscope.hub.file_download import model_file_download
import subprocess

parser.add_argument('--host', dest='host', default='0.0.0.0', help='set host for qanything server')
parser.add_argument('--port', dest='port', default=8777, type=int, help='set port for qanything server')
parser.add_argument('--workers', dest='workers', default=4, type=int, help='sanic server workers number')
# 是否使用GPU
parser.add_argument('--use_cpu', dest='use_cpu', action='store_true', help='use gpu')
# 是否使用Openai API
parser.add_argument('--use_openai_api', dest='use_openai_api', action='store_true', help='use openai api')
parser.add_argument('--openai_api_base', dest='openai_api_base', default='https://api.openai.com/v1', type=str,
                    help='openai api base url')
parser.add_argument('--openai_api_key', dest='openai_api_key', default='your-api-key-here', type=str,
                    help='openai api key')
parser.add_argument('--openai_api_model_name', dest='openai_api_model_name', default='gpt-3.5-turbo-1106', type=str,
                    help='openai api model name')
parser.add_argument('--openai_api_context_length', dest='openai_api_context_length', default='4096', type=str,
                    help='openai api content length')
#  必填参数
parser.add_argument('--model_size', dest='model_size', default='7B', help='set LLM model size for qanything server')
parser.add_argument('--device_id', dest='device_id', default='0', help='cuda device id for qanything server')
args = parser.parse_args()

print('use_cpu:', args.use_cpu, flush=True)
print('use_openai_api:', args.use_openai_api, flush=True)

# 输出用户启动的端口
print(f"The server is starting on port: {args.port}")

if os_system != 'Darwin':
    if not args.use_cpu:
        cuda_version = torch.version.cuda
        if cuda_version is None:
            raise ValueError("CUDA is not installed.")
        elif float(cuda_version) < 12:
            raise ValueError("CUDA version must be 12.0 or higher.")

    python_version = platform.python_version()
    python3_version = python_version.split('.')[1]
    os_system = platform.system()
    if os_system == "Windows":
        raise ValueError("The project must be run in the WSL environment on Windows system.")
    if os_system != "Linux":
        raise ValueError(f"Unsupported system: {os_system}")
    system_name = 'manylinux_2_28_x86_64'
    glibc_info = platform.libc_ver()
    if glibc_info[0] != 'glibc':
        raise ValueError(f"Unsupported libc: {glibc_info[0]}, 请确认系统是否为Linux系统。")
    glibc_version = float(glibc_info[1])
    if glibc_version < 2.28:
        if not check_package_version("onnxruntime", "1.16.3"):
            print(f"当前系统glibc版本为{glibc_version}<2.28，无法使用onnxruntime-gpu(cuda12.x)，将安装onnxruntime来代替", flush=True)
            os.system("pip install onnxruntime")
    elif not args.use_cpu:
        # 官方发布的1.17.1不支持cuda12以上的系统，需要根据官方文档:https://onnxruntime.ai/docs/install/里提到的地址手动下载whl
        if not check_package_version("onnxruntime-gpu", "1.17.1"):
            download_url = f"https://aiinfra.pkgs.visualstudio.com/PublicPackages/_apis/packaging/feeds/9387c3aa-d9ad-4513-968c-383f6f7f53b8/pypi/packages/onnxruntime-gpu/versions/1.17.1/onnxruntime_gpu-1.17.1-cp3{python3_version}-cp3{python3_version}-{system_name}.whl/content"
            debug_logger.info(f'开始从{download_url}下载onnxruntime，也可以手动下载并通过pip install *.whl安装')
            whl_name = f'onnxruntime_gpu-1.17.1-cp3{python3_version}-cp3{python3_version}-{system_name}.whl'
            download_file(download_url, whl_name)
            exit_status = os.system(f"pip install {whl_name}")
            if exit_status != 0:
                # raise ValueError(f"安装onnxruntime失败，请手动安装{whl_name}")
                debug_logger.warning(f"安装onnxruntime-gpu失败，将安装onnxruntime来代替")
                print(f"安装onnxruntime-gpu失败，将安装onnxruntime来代替", flush=True)
                os.system("pip install onnxruntime")
    if not args.use_openai_api:
        if not check_package_version("vllm", "0.2.7"):
            os.system(f"pip install vllm==0.2.7 -i https://pypi.mirrors.ustc.edu.cn/simple/ --trusted-host pypi.mirrors.ustc.edu.cn")

        from vllm.engine.arg_utils import AsyncEngineArgs

        parser = AsyncEngineArgs.add_cli_args(parser)
        args = parser.parse_args()

else:
    # mac下ocr依赖onnxruntime
    if not check_package_version("onnxruntime", "1.17.1"):
        os.system("pip install onnxruntime -i https://pypi.mirrors.ustc.edu.cn/simple/ --trusted-host pypi.mirrors.ustc.edu.cn")
    # torch==2.1.2
    # torchvision==0.16.2
    # os.system("pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu -i https://pypi.mirrors.ustc.edu.cn/simple/ --trusted-host pypi.mirrors.ustc.edu.cn")
    # os.system("pip install torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu -i https://pypi.mirrors.ustc.edu.cn/simple/ --trusted-host pypi.mirrors.ustc.edu.cn")
    if not args.use_openai_api:
        if not check_package_version("llama_cpp_python", "0.2.60"):
            os.system(f'CMAKE_ARGS="-DLLAMA_METAL_EMBED_LIBRARY=ON -DLLAMA_METAL=on" pip install -U llama-cpp-python --no-cache-dir -i https://pypi.mirrors.ustc.edu.cn/simple/ --trusted-host pypi.mirrors.ustc.edu.cn')

model_download_params = None
if not args.use_openai_api:
    model_size = args.model_size
    if os_system == "Linux" and not args.use_cpu:
        model_config.CUDA_DEVICE = args.device_id
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
        args.gpu_memory_utilization = get_gpu_memory_utilization(model_size, args.device_id)
        debug_logger.info(f"GPU memory utilization: {args.gpu_memory_utilization}")
    if model_size == '3B':
        args.model = DT_3B_MODEL_PATH
        model_download_params = DT_3B_DOWNLOAD_PARAMS
    elif model_size == '7B':
        args.model = DT_7B_MODEL_PATH
        model_download_params = DT_7B_DOWNLOAD_PARAMS
    else:
        raise ValueError(f"Unsupported model size: {model_size}, supported model size: 3B, 7B")

# 如果模型不存在, 下载模型
if args.use_openai_api:
    debug_logger.info(f'使用openai api {args.openai_api_model_name} 无需下载大模型')
elif not os.path.exists(args.model):
    debug_logger.info(f'开始下载大模型：{model_download_params}')
    # if os_system == 'Darwin':
    #     cache_dir = model_file_download(**model_download_params)
    #     debug_logger.info(f'模型下载完毕！{cache_dir}')
    # else:
    #     cache_dir = snapshot_download(**model_download_params)
    cache_dir = snapshot_download(**model_download_params)
    debug_logger.info(f'模型下载完毕！{cache_dir}')
    # output = subprocess.check_output(['ln', '-s', cache_dir, args.model], text=True)
    # debug_logger.info(f'模型下载完毕！cache地址：{cache_dir}, 软链接地址：{args.model}')
    debug_logger.info(f"CUDA_DEVICE: {model_config.CUDA_DEVICE}")
else:
    debug_logger.info(f'{args.model}路径已存在，不再重复下载大模型（如果下载出错可手动删除此目录）')
    debug_logger.info(f"CUDA_DEVICE: {model_config.CUDA_DEVICE}")

from .handler import *
from qanything_kernel.core.local_doc_qa import LocalDocQA

WorkerManager.THRESHOLD = 6000

app = Sanic("QAnything")
# 设置请求体最大为 400MB
app.config.REQUEST_MAX_SIZE = 400 * 1024 * 1024

# 将 /qanything 路径映射到 ./dist/qanything 文件夹，并指定路由名称
app.static('/qanything/', 'qanything_kernel/qanything_server/dist/qanything/', name='qanything', index="index.html")

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
    local_doc_qa.init_cfg(args=args)
    debug_logger.info(f"LocalDocQA started in {time.time() - start} seconds.")
    app.ctx.local_doc_qa = local_doc_qa

@app.after_server_start
async def print_info(app, loop):
    print(f"已启动后端服务，请复制[  http://0.0.0.0:{args.port}/qanything/  ]到浏览器进行测试。", flush=True)
    print("详细调试日志请查看 logs/debug_logs/debug.log，问答日志请查看logs/qa_logs/qa.log", flush=True)
    # os.system("tail -f logs/debug_logs/debug.log")

# 启动服务器并尝试自动打开浏览器的函数
@app.after_server_start
async def start_server_and_open_browser(app, loop):
    try:
        print(f"Opening browser at http://{args.host}:{args.port}/qanything/")
        webbrowser.open(f"http://{args.host}:{args.port}/qanything/")
    except Exception as e:
        # 记录或处理任何异常
        print(f"Failed to open browser: {e}")


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
app.add_route(new_bot, "/api/local_doc_qa/new_bot", methods=['POST'])  # tags=["新建Bot"]
app.add_route(delete_bot, "/api/local_doc_qa/delete_bot", methods=['POST'])  # tags=["删除Bot"]
app.add_route(update_bot, "/api/local_doc_qa/update_bot", methods=['POST'])  # tags=["更新Bot"]
app.add_route(get_bot_info, "/api/local_doc_qa/get_bot_info", methods=['POST'])  # tags=["获取Bot信息"]
app.add_route(upload_faqs, "/api/local_doc_qa/upload_faqs", methods=['POST'])  # tags=["上传FAQ"]
app.add_route(get_file_base64, "/api/local_doc_qa/get_file_base64", methods=['POST'])  # tags=["获取文件base64"]
app.add_route(get_qa_info, "/api/local_doc_qa/get_qa_info", methods=['POST'])  # tags=["获取QA信息"]


if __name__ == "__main__":
    # if args.use_openai_api:
    #     try:
    #         # 尝试以指定的workers数量启动应用
    #         app.run(host=args.host, port=args.port, workers=args.workers, access_log=False)
    #     except Exception as e:
    #         debug_logger.info(f"启动多worker模式失败: {e}，尝试以单进程模式启动。")
    #         # 如果出现异常，则退回到单进程模式
    #         app.run(host=args.host, port=args.port, single_process=True, access_log=False)
    # else:
    #     # 模型占用显存大，多个worker显存不够用
    #     app.run(host=args.host, port=args.port, single_process=True, access_log=False)
    # 由于有用户启动时上下文环境报错，使用单进程模式：
    app.run(host=args.host, port=args.port, single_process=True, access_log=False)


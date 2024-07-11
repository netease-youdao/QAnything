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

from handler import *
from qanything_kernel.core.local_doc_qa import LocalDocQA
from qanything_kernel.utils.custom_log import debug_logger, qa_logger
from sanic.worker.manager import WorkerManager
from sanic import Sanic
from sanic_ext import Extend
import time
import argparse

WorkerManager.THRESHOLD = 6000

# 接收外部参数mode
parser = argparse.ArgumentParser()
# mode必须是local或online
parser.add_argument('--mode', type=str, default='online', help='local or online')
parser.add_argument('--port', type=int, default=8777, help='port')
parser.add_argument('--workers', type=int, default=4, help='workers')
# 检查是否是local或online，不是则报错
args = parser.parse_args()
if args.mode not in ['local', 'online']:
    raise ValueError('mode must be local or online')

start_time = time.time()
app = Sanic("QAnything")
app.config.CORS_ORIGINS = "*"
Extend(app)
# 设置请求体最大为 128MB
app.config.REQUEST_MAX_SIZE = 128 * 1024 * 1024

# 将 /qanything 路径映射到 ./dist/qanything 文件夹，并指定路由名称
app.static('/qanything/', 'qanything_kernel/qanything_server/dist/qanything/', name='qanything', index="index.html")


@app.before_server_start
async def init_local_doc_qa(app, loop):
    start = time.time()
    local_doc_qa = LocalDocQA(args.port)
    local_doc_qa.init_cfg()
    end = time.time()
    print(f'init local_doc_qa cost {end - start}s', flush=True)
    app.ctx.local_doc_qa = local_doc_qa
    
@app.after_server_start
async def notify_server_started(app, loop):
    print(f"Server Start Cost {time.time() - start_time} seconds", flush=True)

# app.add_route(lambda req: response.redirect('/api/docs'), '/')
# tags=["新建知识库"]
app.add_route(document, "/api/docs", methods=['GET'])
app.add_route(health_check, "/api/health_check", methods=['GET'])  # tags=["健康检查"]
app.add_route(new_knowledge_base, "/api/local_doc_qa/new_knowledge_base", methods=['POST'])  # tags=["新建知识库"]
app.add_route(upload_weblink, "/api/local_doc_qa/upload_weblink", methods=['POST'])  # tags=["上传网页链接"]
app.add_route(upload_files, "/api/local_doc_qa/upload_files", methods=['POST'])  # tags=["上传文件"]
app.add_route(upload_faqs, "/api/local_doc_qa/upload_faqs", methods=['POST'])  # tags=["上传FAQ"]
app.add_route(local_doc_chat, "/api/local_doc_qa/local_doc_chat", methods=['POST'])  # tags=["问答接口"] 
app.add_route(list_kbs, "/api/local_doc_qa/list_knowledge_base", methods=['POST'])  # tags=["知识库列表"] 
app.add_route(list_docs, "/api/local_doc_qa/list_files", methods=['POST'])  # tags=["文件列表"]
app.add_route(get_total_status, "/api/local_doc_qa/get_total_status", methods=['POST'])  # tags=["获取所有知识库状态数据库"]
app.add_route(clean_files_by_status, "/api/local_doc_qa/clean_files_by_status", methods=['POST'])  # tags=["清理数据库"]
app.add_route(delete_docs, "/api/local_doc_qa/delete_files", methods=['POST'])  # tags=["删除文件"] 
app.add_route(delete_knowledge_base, "/api/local_doc_qa/delete_knowledge_base", methods=['POST'])  # tags=["删除知识库"] 
app.add_route(rename_knowledge_base, "/api/local_doc_qa/rename_knowledge_base", methods=['POST'])  # tags=["重命名知识库"]
app.add_route(get_doc_completed, "/api/local_doc_qa/get_doc_completed", methods=['POST'])  # tags=["获取文档完整解析内容"]
app.add_route(get_qa_info, "/api/local_doc_qa/get_qa_info", methods=['POST'])  # tags=["获取QA信息"]
app.add_route(get_user_id, "/api/local_doc_qa/get_user_id", methods=['POST'])  # tags=["获取用户ID"]
app.add_route(get_doc, "/api/local_doc_qa/get_doc", methods=['POST'])  # tags=["获取doc详细内容"]
app.add_route(get_rerank_results, "/api/local_doc_qa/get_rerank_results", methods=['POST'])  # tags=["获取rerank结果"]
app.add_route(get_user_status, "/api/local_doc_qa/get_user_status", methods=['POST'])  # tags=["获取用户状态"]
app.add_route(get_random_qa, "/api/local_doc_qa/get_random_qa", methods=['POST'])  # tags=["获取随机QA"]
app.add_route(get_related_qa, "/api/local_doc_qa/get_related_qa", methods=['POST'])  # tags=["获取相关QA"]
app.add_route(new_bot, "/api/local_doc_qa/new_bot", methods=['POST'])  # tags=["新建Bot"]
app.add_route(delete_bot, "/api/local_doc_qa/delete_bot", methods=['POST'])  # tags=["删除Bot"]
app.add_route(update_bot, "/api/local_doc_qa/update_bot", methods=['POST'])  # tags=["更新Bot"]
app.add_route(get_bot_info, "/api/local_doc_qa/get_bot_info", methods=['POST'])  # tags=["获取Bot信息"]

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=args.port, workers=args.workers, access_log=False)

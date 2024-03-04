import os
import logging
import uuid
from dotenv import load_dotenv
load_dotenv()

# 默认的CUDA设备
CUDA_DEVICE = '0'

# 获取项目根目录
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
UPLOAD_ROOT_PATH = os.path.join(root_path, "QANY_DB", "content")
print("UPLOAD_ROOT_PATH:", UPLOAD_ROOT_PATH)

# LLM streaming reponse
STREAMING = True

PROMPT_TEMPLATE = """参考信息：
{context}
---
我的问题或指令：
{question}
---
请根据上述参考信息回答我的问题或回复我的指令。前面的参考信息可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复,
你的回复："""

# For LLM Chat w/o Retrieval context 
# PROMPT_TEMPLATE = """{question}"""

QUERY_PROMPT_TEMPLATE = """{question}"""

# 缓存知识库数量
CACHED_VS_NUM = 100

# 文本分句长度
SENTENCE_SIZE = 100

# 匹配后单段上下文长度
CHUNK_SIZE = 800

# 传入LLM的历史记录长度
LLM_HISTORY_LEN = 3

# 知识库检索时返回的匹配内容条数
VECTOR_SEARCH_TOP_K = 100

# embedding检索的相似度阈值，归一化后的L2距离，设置越大，召回越多，设置越小，召回越少
VECTOR_SEARCH_SCORE_THRESHOLD = 1.1

# NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")
# print('NLTK_DATA_PATH', NLTK_DATA_PATH)

# 是否开启中文标题加强，以及标题增强的相关配置
# 通过增加标题判断，判断哪些文本为标题，并在metadata中进行标记；
# 然后将文本与往上一级的标题进行拼合，实现文本信息的增强。
ZH_TITLE_ENHANCE = False

# MILVUS向量数据库地址
MILVUS_HOST_LOCAL = '127.0.0.1'
MILVUS_HOST_ONLINE = '127.0.0.1'
MILVUS_PORT = 19530
MILVUS_USER = ''
MILVUS_PASSWORD = ''
MILVUS_DB_NAME = ''

# MYSQL_HOST_LOCAL = 'mysql-container-local'
# MYSQL_HOST_ONLINE = 'mysql-container-local'
# MYSQL_PORT = 3306
# MYSQL_USER = 'root'
# MYSQL_PASSWORD = '123456'
# MYSQL_DATABASE = 'qanything'

SQLITE_DATABASE = os.path.join(root_path, "QANY_DB", "qanything.db")
MILVUS_LITE_LOCATION = os.path.join(root_path, "QANY_DB", "milvus")

llm_api_serve_model = os.getenv('LLM_API_SERVE_MODEL', 'MiniChat-2-3B')
llm_api_serve_port = os.getenv('LLM_API_SERVE_PORT', 7802)

LOCAL_LLM_SERVICE_URL = f"localhost:{llm_api_serve_port}"
LOCAL_LLM_MODEL_NAME = llm_api_serve_model
LOCAL_LLM_MAX_LENGTH = 4096

# LOCAL_RERANK_SERVICE_URL = f"localhost:{rerank_port}"
LOCAL_RERANK_MODEL_PATH = os.path.join(root_path, "onnx_models", "rerank.onnx")
LOCAL_RERANK_CONFIG_PATH = os.path.join(root_path, 'qanything_kernel/connector/rerank', 'rerank_model_configs_v0.0.1')
LOCAL_RERANK_MODEL_NAME = 'rerank'
LOCAL_RERANK_MAX_LENGTH = 512
LOCAL_RERANK_BATCH = 16

# LOCAL_EMBED_SERVICE_URL = f"localhost:{embed_port}"
LOCAL_EMBED_MODEL_PATH = os.path.join(root_path, "onnx_models", "embed.onnx")
LOCAL_EMBED_CONFIG_PATH = os.path.join(root_path, 'qanything_kernel/connector/embedding', 'embedding_model_configs_v0.0.1')
LOCAL_EMBED_MODEL_NAME = 'embed'
LOCAL_EMBED_MAX_LENGTH = 512
LOCAL_EMBED_BATCH = 16

# nohup python3 -m fastchat.serve.controller --host 0.0.0.0 --port 7800 > logs/debug_logs/fastchat_logs/fschat_controller_7800.log 2>&1 &
CONTROLLER_PORT = 7800

# nohup python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 7802 --controller-address http://0.0.0.0:7800 > logs/debug_logs/fastchat_logs/
OPENAI_API_SERVER_PORT = 7802
OPENAI_API_SERVER_CONTROLLER_ADDRESS = f"http://0.0.0.0:{CONTROLLER_PORT}"


# CUDA_VISIBLE_DEVICES=3 nohup python3 -m fastchat.serve.vllm_worker --host 0.0.0.0 --port 7801 \
#             --controller-address http://0.0.0.0:7800 --worker-address http://0.0.0.0:7801 \
#             --model-path assets/custom_models/MiniChat-2-3B --trust-remote-code --block-size 32 --tensor-parallel-size 1 \
#             --max-model-len 4096 --gpu-memory-utilization 0.81 --dtype bfloat16 --conv-template minichat > logs/debug_logs/fastchat_logs/fschat_model_worker_7801.log 2>&1 &

# VLLM PARAMS
VW_PORT = 7801
VW_CONTROLLER_ADDRESS = f"http://0.0.0.0:{CONTROLLER_PORT}"
VW_WORKER_ADDRESS = f"http://0.0.0.0:{VW_PORT}"
VW_MODEL_PATH = os.path.join(root_path, "assets", "custom_models", "MiniChat-2-3B")
VW_TRUST_REMOTE_CODE = False
VW_BLOCK_SIZE = 32
VW_TENSOR_PARALLEL_SIZE = 1
VW_MAX_MODEL_LEN = 4096
VW_GPU_MEMORY_UTILIZATION = 0.81
VW_DTYPE = "bfloat16"
VW_CONV_TEMPLATE = "minichat"

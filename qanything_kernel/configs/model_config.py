import os
import nltk
import platform
from dotenv import load_dotenv
load_dotenv()

os_system = platform.system()

# 默认的CUDA设备
CUDA_DEVICE = '0'

# 获取项目根目录
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
UPLOAD_ROOT_PATH = os.path.join(root_path, "QANY_DB", "content")
print("LOCAL DATA PATH:", UPLOAD_ROOT_PATH)
# 如果不存在则创建
if not os.path.exists(UPLOAD_ROOT_PATH):
    os.makedirs(UPLOAD_ROOT_PATH)

nltk_data_path = os.path.join(root_path, 'qanything_kernel/nltk_data')

# 将你的本地路径添加到nltk的数据路径中
nltk.data.path.append(nltk_data_path)

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

SQLITE_DATABASE = os.path.join(root_path, "QANY_DB", "qanything.db")
MILVUS_LITE_LOCATION = os.path.join(root_path, "QANY_DB", "milvus")

# llm_api_serve_model = os.getenv('LLM_API_SERVE_MODEL', 'MiniChat-2-3B')
# llm_api_serve_port = os.getenv('LLM_API_SERVE_PORT', 7802)

# LOCAL_LLM_SERVICE_URL = f"localhost:{llm_api_serve_port}"
# LOCAL_LLM_MODEL_NAME = llm_api_serve_model
# LOCAL_LLM_MAX_LENGTH = 4096

LOCAL_RERANK_PATH = os.path.join(root_path, 'qanything_kernel/connector/rerank', 'rerank_model_configs_v0.0.1')
if os_system == 'Darwin':
    LOCAL_RERANK_REPO = "maidalun/bce-reranker-base_v1"
    LOCAL_RERANK_MODEL_PATH = os.path.join(LOCAL_RERANK_PATH, "pytorch_model.bin")
else:
    LOCAL_RERANK_REPO = "netease-youdao/bce-reranker-base_v1"
    LOCAL_RERANK_MODEL_PATH = os.path.join(LOCAL_RERANK_PATH, "rerank.onnx")
print('LOCAL_RERANK_REPO:', LOCAL_RERANK_REPO)
LOCAL_RERANK_MODEL_NAME = 'rerank'
LOCAL_RERANK_MAX_LENGTH = 512
LOCAL_RERANK_BATCH = 16

LOCAL_EMBED_PATH = os.path.join(root_path, 'qanything_kernel/connector/embedding', 'embedding_model_configs_v0.0.1')
if os_system == 'Darwin':
    LOCAL_EMBED_REPO = "maidalun/bce-embedding-base_v1"
    LOCAL_EMBED_MODEL_PATH = os.path.join(LOCAL_EMBED_PATH, "pytorch_model.bin")
else:
    LOCAL_EMBED_REPO = "netease-youdao/bce-embedding-base_v1"
    LOCAL_EMBED_MODEL_PATH = os.path.join(LOCAL_EMBED_PATH, "embed.onnx")
print('LOCAL_EMBED_REPO:', LOCAL_EMBED_REPO)
LOCAL_EMBED_MODEL_NAME = 'embed'
LOCAL_EMBED_MAX_LENGTH = 512
LOCAL_EMBED_BATCH = 16

# VLLM PARAMS
model_path = os.path.join(root_path, "assets", "custom_models")
# 检查目录是否存在，如果不存在则创建
if not os.path.exists(model_path):
    os.makedirs(model_path)
if os_system == 'Darwin':
    VW_4B_MODEL_PATH = os.path.join(model_path, "qwen/Qwen1.5-4B-Chat-GGUF") + '/qwen1_5-4b-chat-q4_k_m.gguf'
    VW_7B_MODEL_PATH = os.path.join(model_path, "qwen/Qwen1.5-7B-Chat-GGUF") + '/qwen1_5-7b-chat-q4_k_m.gguf'
    VM_4B_DOWNLOAD_PARAMS = {'model_id': 'qwen/Qwen1.5-4B-Chat-GGUF', 'file_path': 'qwen1_5-4b-chat-q4_k_m.gguf', 'revision': 'master', 'cache_dir': model_path}
    VM_7B_DOWNLOAD_PARAMS = {'model_id': 'qwen/Qwen1.5-7B-Chat-GGUF', 'file_path': 'qwen1_5-7b-chat-q4_k_m.gguf', 'revision': 'master', 'cache_dir': model_path}
    VW_CONV_4B_TEMPLATE = "qwen-7b-chat"
    VW_CONV_7B_TEMPLATE = "qwen-7b-chat"
else:
    VW_4B_MODEL_PATH = os.path.join(model_path, "Qwen1.5-4B-Chat")
    VW_7B_MODEL_PATH = os.path.join(model_path, "Qwen-7B-QAnything")
    VM_4B_DOWNLOAD_PARAMS = {'model_id': 'qwen/Qwen1.5-4B-Chat',
                             'revision': 'master', 'cache_dir': VW_4B_MODEL_PATH}
    VM_7B_DOWNLOAD_PARAMS = {'model_id': 'netease-youdao/Qwen-7B-QAnything',
                             'revision': 'master', 'cache_dir': VW_7B_MODEL_PATH}
    VW_CONV_4B_TEMPLATE = "qwen-7b-chat"
    VW_CONV_7B_TEMPLATE = "qwen-7b-qanything"


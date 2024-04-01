import os
import logging
import uuid
from dotenv import load_dotenv
load_dotenv()

# LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logging.basicConfig(format=LOG_FORMAT)
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
MILVUS_HOST_LOCAL = 'milvus-standalone-local'
MILVUS_HOST_ONLINE = 'milvus-standalone-local'
MILVUS_PORT = 19530
MILVUS_USER = ''
MILVUS_PASSWORD = ''
MILVUS_DB_NAME = ''

# ES bm25检索相关配置
HYBRID_SEARCH = True  # 是否启用es的BM25检索
ES_BM25_SEARCH_SIZE = 50

ES_HOST_LOCAL = 'http://es-container-local:9200/'
ES_CLOUD_ID = None
ES_USER = None
ES_PASSWORD = None
ES_API_KEY = None
ES_CONNECT_PARAMS = None
ES_SIGNATURE = 'qanything/bm25_v240320'


MYSQL_HOST_LOCAL = 'mysql-container-local'
MYSQL_HOST_ONLINE = 'mysql-container-local'
MYSQL_PORT = 3306
MYSQL_USER = 'root'
MYSQL_PASSWORD = '123456'
MYSQL_DATABASE = 'qanything'

llm_api_serve_model = os.getenv('LLM_API_SERVE_MODEL')
llm_api_serve_port = os.getenv('LLM_API_SERVE_PORT')
rerank_port = os.getenv('RERANK_PORT')
embed_port = os.getenv('EMBED_PORT')

print("llm_api_serve_port:", llm_api_serve_port)
print("rerank_port:", rerank_port)
print("embed_port:", embed_port)


LOCAL_LLM_SERVICE_URL = f"localhost:{llm_api_serve_port}"
LOCAL_LLM_MODEL_NAME = llm_api_serve_model
LOCAL_LLM_MAX_LENGTH = 4096

LOCAL_RERANK_SERVICE_URL = f"localhost:{rerank_port}"
LOCAL_RERANK_MODEL_NAME = 'rerank'
LOCAL_RERANK_MAX_LENGTH = 512
LOCAL_RERANK_BATCH = 16

LOCAL_EMBED_SERVICE_URL = f"localhost:{embed_port}"
LOCAL_EMBED_MODEL_NAME = 'embed'
LOCAL_EMBED_MAX_LENGTH = 512
LOCAL_EMBED_BATCH = 16

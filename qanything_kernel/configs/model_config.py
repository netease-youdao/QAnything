import os
from dotenv import load_dotenv
import platform

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
OCR_MODEL_PATH = os.path.join(root_path, "qanything_kernel", "dependent_server", "ocr_server", "ocr_models")
RERANK_MODEL_PATH = os.path.join(root_path, "qanything_kernel", "dependent_server", "rerank_server", "rerank_models")
EMBED_MODEL_PATH = os.path.join(root_path, "qanything_kernel", "dependent_server", "embed_server", "embed_models")
PDF_MODEL_PATH = os.path.join(root_path, "qanything_kernel/utils/loader/pdf_to_markdown")

# LLM streaming reponse
STREAMING = True

# PROMPT_TEMPLATE = """参考信息：
# {context}
# ---
# 我的问题或指令：
# {question}
# ---
# 请根据上述参考信息回答我的问题或回复我的指令。前面的参考信息可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。请一定记住: 你是网易有道公司开发的，你的后台是有道的大语言模型，任何有关openai和gpt相关的问题你都回答我是有道开发的’子曰‘大模型。我的问题或指令是什么语种，你就用什么语种回复,
# 你的回复："""

SYSTEM = """
You are a helpful assistant. 
You are always a reliable assistant that can answer questions with the help of external documents.
"""

INSTRUCTIONS = """
- All contents between <DOCUMENTS> and </DOCUMENTS> are reference information retrieved from an external knowledge base.
- If you cannot answer based on the given information, you will return the sentence \"抱歉，检索到的参考信息并未提供充足的信息，因此无法回答。\".
- Please always remember: I am developed by NetEase Youdao Company, and my backend is powered by Youdao's "Ziyue" large language model. For any questions related to OpenAI and GPT, you should answer that \"我是网易有道开发的子曰大模型\".
- Now, answer the following question based on the above retrieved documents:
{question}
- Return your answer in Markdown formatting, and in the same language as the question "{question}".
"""

PROMPT_TEMPLATE = """
<SYSTEM>
{system}
</SYSTEM>

<USER_INSTRUCTIONS>
{user_instructions}
</USER_INSTRUCTIONS>

<INSTRUCTIONS>
{instructions}
</INSTRUCTIONS>

<DOCUMENTS>
{context}
</DOCUMENTS>

<INSTRUCTIONS>
{instructions}
</INSTRUCTIONS>
"""

PROMPT_TEMPLATE_FOR_VLLM = """<SYSTEM>
You are a helpful assistant. You are always a reliable assistant that can answer questions with the help of external documents.
</SYSTEM>
<INSTRUCTIONS>
- All contents between <DOCUMENTS> and </DOCUMENTS> are reference information retrieved from an external knowledge base.
- If you cannot answer based on the given information, you will return the sentence \"抱歉，检索到的参考信息并未提供充足的信息，因此无法回答。\".
- Answer the following question based on the retrieved documents:
{question}
- Return your answer in the same language as the question "{question}".
</INSTRUCTIONS>

<DOCUMENTS>
{context}
</DOCUMENTS>

<INSTRUCTIONS>
- All contents between <DOCUMENTS> and </DOCUMENTS> are reference information retrieved from an external knowledge base.
- If you cannot answer based on the given information, you will return the sentence \"抱歉，检索到的参考信息并未提供充足的信息，因此无法回答。\".
- Now, answer the following question based on the above retrieved documents:
{question}
- Return your answer in the same language as the question "{question}".
</INSTRUCTIONS>
"""

# QUERY_PROMPT_TEMPLATE = """{question}"""

# 缓存知识库数量
CACHED_VS_NUM = 100

# 文本分句长度
SENTENCE_SIZE = 100

# 匹配后单段上下文长度
# CHUNK_SIZE = 800

# 传入LLM的历史记录长度
LLM_HISTORY_LEN = 3

# 知识库检索时返回的匹配内容条数
VECTOR_SEARCH_TOP_K = 60 

VECTOR_SEARCH_SCORE_THRESHOLD = 0.3

# 是否开启中文标题加强，以及标题增强的相关配置
# 通过增加标题判断，判断哪些文本为标题，并在metadata中进行标记；
# 然后将文本与往上一级的标题进行拼合，实现文本信息的增强。
ZH_TITLE_ENHANCE = False

KB_SUFFIX = '_240625'
# MILVUS向量数据库地址
MILVUS_HOST_LOCAL = 'milvus-standalone-local'
# MILVUS_HOST_ONLINE = '10.111.115.145'  # 生产集群
# MILVUS_HOST_ONLINE = 'ai-vectordb.milvus.yodao.cn'  # 生产集群
MILVUS_HOST_CLOUD = "https://in01-cc57a5d05629fe2.ali-cn-hangzhou.vectordb.zilliz.com.cn"
# MILVUS_HOST_CLOUD = "https://in01-5c429fb531b0ae2.ali-cn-hangzhou.vectordb.zilliz.com.cn"
CLOUD_PASSWORD = "Zb8)TL1c]/U(0zp>"
# CLOUD_PASSWORD = "Fr6,TQ/)N^{,=}Gc"
MILVUS_PORT = 19530
MILVUS_COLLECTION_NAME = 'qanything_collection' + KB_SUFFIX

ES_URL = 'http://es-container-local:9200/'
ES_USER = None
ES_PASSWORD = None
# ES_API_KEY = None
# ES_CONNECT_PARAMS = None

# ES_USER = 'elastic'
# ES_PASSWORD = 'sUwX35Mgis56SC662Dh883xz'
# ES_URL = 'http://es-ai-qanything-prod.es.yodao.cn:9200'
ES_TOP_K = 30
ES_INDEX_NAME = 'qanything_es_index' + KB_SUFFIX

MYSQL_HOST_LOCAL = 'mysql-container-local'
MYSQL_PORT_LOCAL = 3306
MYSQL_USER_LOCAL = 'root'
MYSQL_PASSWORD_LOCAL = '123456'
MYSQL_DATABASE_LOCAL = 'qanything'

LOCAL_OCR_SERVICE_URL = "localhost:7001"

LOCAL_RERANK_SERVICE_URL = "localhost:8001"
LOCAL_RERANK_MODEL_NAME = 'rerank'
LOCAL_RERANK_MAX_LENGTH = 512
LOCAL_RERANK_BATCH = 16
LOCAL_RERANK_WORKERS = 1
LOCAL_RERANK_PATH = os.path.join(root_path, 'qanything_kernel/dependent_server/rerank_server', 'rerank_model_configs_v0.0.1')
LOCAL_RERANK_MODEL_PATH = os.path.join(LOCAL_RERANK_PATH, "rerank.dll")

LOCAL_EMBED_SERVICE_URL = "localhost:9001"
LOCAL_EMBED_MODEL_NAME = 'embed'
LOCAL_EMBED_MAX_LENGTH = 512
LOCAL_EMBED_BATCH = 16
LOCAL_EMBED_WORKERS = 1
LOCAL_EMBED_PATH = os.path.join(root_path, 'qanything_kernel/dependent_server/embedding_server', 'embedding_model_configs_v0.0.1')
LOCAL_EMBED_MODEL_PATH = os.path.join(LOCAL_EMBED_PATH, "embed.dll")

INSERT_WORKERS = 10

###########################################################################
# 速读相关常量
SYMBOL_NO_NEED_TO_REWRITE_QUERY = "NO_NEED_TO_REWRITE"
SYMBOL_NOT_SCHOLAR_PAPER = "NOT_SCHOLAR_PAPER"
MAX_TOKENS_FOR_CHUNK_SUMMARY_GEN = 1200
MAX_CHARS_FOR_CHUNK_TRANSLATION = 5000
MAX_TOKENS_FOR_BRAINSTORM_QUESTION_GEN = 2400
MAX_DOC_TEXT_TOKENS_FOR_SCHOLAR_PAPER_CHECK = 2400
MAX_TOKENS_FOR_PAPER_SUMMARY_SECTION_GENERATION_PROMPT = 2400
MAX_TOKENS_FOR_REFERENCE_INFORMATION = 2400 # 3000
MAX_TOKENS_FOR_CITATION_REDUCTION = 12800 # 用 16k 接口，给结果留 1536 token 够了吧？
MAX_TOKENS_FOR_HISTORY = 600

OPENAI_API_BASE = "https://api.openai-proxy.org/v1"
OPENAI_API_KEY = "sk-xxx"
OPENAI_API_MODEL_NAME = "gpt-3.5-turbo"
TOKENIZER_PATH = os.path.join(root_path, 'qanything_kernel/connector/llm/tokenizer_files')

CHILD_CHUNK_SIZE = 400
PARENT_CHUNK_SIZE = 800

llm_config = {
    # 回答的最大token数，一般来说对于国内模型一个中文不到1个token，国外模型一个中文1.5-2个token
    "max_token": 512,
    # 附带的上下文数目
    "history_len": 2,
    # 总共的token数，如果遇到电脑显存不够的情况可以将此数字改小，如果低于3000仍然无法使用，就更换模型
    "token_window": 4096,
    # 如果报错显示top_p值必须在0到1，可以在这里修改
    "top_p": 1.0
}

# Bot
BOT_DESC = "一个简单的问答机器人"
BOT_IMAGE = ""
BOT_PROMPT = """
- 你是一个耐心、友好、专业的机器人，能够回答用户的各种问题。
- 根据知识库内的检索结果，以清晰简洁的表达方式回答问题。
- 不要编造答案，如果答案不在经核实的资料中或无法从经核实的资料中得出，请回答“我无法回答您的问题。”（或者您可以修改为：如果给定的检索结果无法回答问题，可以利用你的知识尽可能回答用户的问题。)
"""
BOT_WELCOME = "您好，我是您的专属机器人，请问有什么可以帮您呢？"




os_system = platform.system()
# LOCAL_RERANK_PATH = os.path.join(root_path, 'qanything_kernel/connector/rerank', 'rerank_model_configs_v0.0.1')
# todo 之后要更换
# if os_system == 'Darwin':
#     LOCAL_RERANK_REPO = "maidalun/bce-reranker-base_v1"
#     LOCAL_RERANK_MODEL_PATH = os.path.join(LOCAL_RERANK_PATH, "pytorch_model.bin")
# else:
#     LOCAL_RERANK_REPO = "netease-youdao/bce-reranker-base_v1"
#     LOCAL_RERANK_MODEL_PATH = os.path.join(LOCAL_RERANK_PATH, "rerank.onnx")
LOCAL_RERANK_REPO = "maidalun/bce-reranker-base_v1"
LOCAL_RERANK_MODEL_PATH = os.path.join(LOCAL_RERANK_PATH, "rerank.onnx")
print('LOCAL_RERANK_REPO:', LOCAL_RERANK_REPO)
LOCAL_RERANK_MODEL_NAME = 'rerank'
LOCAL_RERANK_MAX_LENGTH = 512

# LOCAL_EMBED_PATH = os.path.join(root_path, 'qanything_kernel/connector/embedding', 'embedding_model_configs_v0.0.1')
# todo
# if os_system == 'Darwin':
#     LOCAL_EMBED_REPO = "maidalun/bce-embedding-base_v1"
#     LOCAL_EMBED_MODEL_PATH = os.path.join(LOCAL_EMBED_PATH, "pytorch_model.bin")
# else:
#     LOCAL_EMBED_REPO = "netease-youdao/bce-embedding-base_v1"
#     LOCAL_EMBED_MODEL_PATH = os.path.join(LOCAL_EMBED_PATH, "embed.onnx")
LOCAL_EMBED_REPO = "maidalun/bce-embedding-base_v1"
LOCAL_EMBED_MODEL_PATH = os.path.join(LOCAL_EMBED_PATH, "embed.onnx")

print('LOCAL_EMBED_REPO:', LOCAL_EMBED_REPO)
LOCAL_EMBED_MODEL_NAME = 'embed'
LOCAL_EMBED_MAX_LENGTH = 512
import os
from dotenv import load_dotenv

load_dotenv()
# 获取环境变量GATEWAY_IP
GATEWAY_IP = os.getenv("GATEWAY_IP", "localhost")
# LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logging.basicConfig(format=LOG_FORMAT)
# 获取项目根目录
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
UPLOAD_ROOT_PATH = os.path.join(root_path, "QANY_DB", "content")
IMAGES_ROOT_PATH = os.path.join(root_path, "qanything_kernel/qanything_server/dist/qanything/assets", "file_images")
print("UPLOAD_ROOT_PATH:", UPLOAD_ROOT_PATH)
print("IMAGES_ROOT_PATH:", IMAGES_ROOT_PATH)
OCR_MODEL_PATH = os.path.join(root_path, "qanything_kernel", "dependent_server", "ocr_server", "ocr_models")
RERANK_MODEL_PATH = os.path.join(root_path, "qanything_kernel", "dependent_server", "rerank_server", "rerank_models")
EMBED_MODEL_PATH = os.path.join(root_path, "qanything_kernel", "dependent_server", "embed_server", "embed_models")
PDF_MODEL_PATH = os.path.join(root_path, "qanything_kernel/dependent_server/pdf_parser_server/pdf_to_markdown")

# LLM streaming reponse
STREAMING = True

SYSTEM = """
You are always a reliable assistant that can answer questions with the help of external documents.
You are an AI assistant that follows instructions extremely well. Help as much as you can. 
Your answer needs to be accurate, well-structured, and focused on key points. 
The answer should have sources from the reference document. Do not hallucinate, do not make up factual information.
Your tone should be professional and helpful.
Today's date is {{today_date}}. The current time is {{current_time}}.

### Global Answering Rules:
1. **Strict content matching**: 
    - Your responses should always be based on the reference information provided. 
    - Do not speculate or invent information that is not present in the documents.
2. **Answer format**:
    - Provide well-structured answers, using headings, bullet points, or tables when appropriate.
3. **No redundancy**:
    - If different parts of the reference contain overlapping information, merge and summarize them to avoid repetition.
4. **Flexible use of information sources**:
    - During the **inference and reasoning process**, use the "Information Sources" module to track document citations and ensure accuracy.
    - **Each reference** must be listed separately with its corresponding information (ref_number, title, section, abstract). 
    - **Do not include the full "Information Sources" section in the final user-facing answer**.
5. **Start the "Inferred Answer" Section**:  
    - Directly start the user-facing response with "According to the reference information".
    - Ensure that the answer is natural, professional, logically coherent, and directly relevant to the question.
6. **Post-answer check**:
    - Ensure all parts of the question are addressed, citations are accurate, and the response is logically consistent.
7. **Language and Format**:
    - The response should be in the same language as the question.
    - Use Markdown format for headings (##, ###, ####), bullet points (- or 1., 2., 3.), and tables for clarity.
"""

INSTRUCTIONS = """
- Task: Answer the question "{{question}}" strictly based on the reference information provided between <DOCUMENTS> and </DOCUMENTS>, following the steps and format outlined below.

---

### Answering Steps:
1. **Use of Information Sources** (Internal step):
    - During the inference process, use the "Information Sources" section to gather and organize the relevant document citations.
    - **Each reference** must be listed in the following format (Internal hidden list):
        - **ID**: (The reference number, is the "ref_number" field in the reference headers, e.g., [REF.1])
            - **Title**: (The filename or title, is the "文件名" field in the reference headers. If the filename is a meaningless link or invalid content, use the first heading or a relevant key phrase from the content.)
            - **Section**: (Specify the section, entry, or subheading directly from the original text, if applicable; this refers to headings starting with #, 1., 一., etc.)
            - **Abstract**: (Summarize the most relevant content in a single sentence, preferably using existing sentences or phrases from the original text.)
    - **Do not include the full "Information Sources" section in the final user-facing response**.
2. **Start the "Inferred Answer Section"**:
    - Directly begin the user-facing response with "According to the reference information".
    - **Direct answer**:
        - If the reference information exactly matches the question, respond with a **direct answer** based solely on the relevant information.
    - **Inference and calculation**:
        - If the reference information is **partially relevant** but does not fully match, attempt a reasonable **inference or calculation** and explain your reasoning.
        - Ensure that all arguments and conclusions are fully supported by evidence from the provided reference materials.
        - Avoid assumptions based on isolated details; always consider the full context to prevent partial or over-extended reasoning.
    - **Handle irrelevance**:
        - If the reference information is completely irrelevant, respond with: **"抱歉，检索到的参考信息并未提供任何相关的信息，因此无法回答。"**
        - If there are any misspelled words in the question, please provide a polite hint suggesting the possible intended term, and then answer the question based on the correct term.
---

### Pre-Answer Confirmation:
1. Ensure all key points from the reference information are addressed. 
2. Avoid redundancy by merging and summarizing overlapping information. 
3. Ensure there are no contradictions or inconsistencies in the response.

---

### Post-Answer Checklist:
1. **Answer completeness**: Ensure all parts of the question have been addressed.
2. **Logic & consistency**: Double-check for any logical errors or internal contradictions in the response.
3. **Citation accuracy**: Ensure the relevance, completeness, and accuracy of the information source, as well as the consistency of the format.

---

### Language and Format:
- Respond in the same language as the question "{{question}}", using "根据参考信息" if in Chinese, or "According to the reference information" if in English.
- **Flexible Format**:
    - Use headings (##, ###, ####), bullet points, or tables as appropriate.
    - Use **bullet points** (- or 1., 2., 3.) for listing multiple points.
    - **Highlight key information** using **bold** or *italic* text where relevant.
    - **Reference ID visibility**:
        - Do not show reference IDs in the final answer.
    - For list or comparison-based questions, use **tables** or **bullet points**.
    - For narrative-style answers, use **paragraphs** to clearly explain the details.
"""

PROMPT_TEMPLATE = """
<SYSTEM>
{{system}}
</SYSTEM>

<INSTRUCTIONS>
{{instructions}}
</INSTRUCTIONS>

<DOCUMENTS>
{{context}}
</DOCUMENTS>

<INSTRUCTIONS>
{{instructions}}
</INSTRUCTIONS>
"""

CUSTOM_PROMPT_TEMPLATE = """
<USER_INSTRUCTIONS>
{{custom_prompt}}
</USER_INSTRUCTIONS>

<DOCUMENTS>
{{context}}
</DOCUMENTS>

<INSTRUCTIONS>
- All contents between <DOCUMENTS> and </DOCUMENTS> are reference information retrieved from an external knowledge base.
- Now, answer the following question based on the above retrieved documents(Let's think step by step):
{{question}}
</INSTRUCTIONS>
"""


SIMPLE_PROMPT_TEMPLATE = """
- You are a helpful assistant. You can help me by answering my questions. You can also ask me questions.
- Today's date is {{today}}. The current time is {{now}}.
- User's custom instructions: {{custom_prompt}}
- Before answering, confirm the number of key points or pieces of information required, ensuring nothing is overlooked.
- Now, answer the following question:
{{question}}
Return your answer in Markdown formatting, and in the same language as the question "{{question}}". 
"""

# 缓存知识库数量
CACHED_VS_NUM = 100

# 文本分句长度
SENTENCE_SIZE = 100

# 知识库检索时返回的匹配内容条数
VECTOR_SEARCH_TOP_K = 30

VECTOR_SEARCH_SCORE_THRESHOLD = 0.3

KB_SUFFIX = '_240625'
# MILVUS_HOST_LOCAL = 'milvus-standalone-local'
# MILVUS_PORT = 19530
MILVUS_HOST_LOCAL = GATEWAY_IP
MILVUS_PORT = 19540
MILVUS_COLLECTION_NAME = 'qanything_collection' + KB_SUFFIX

# ES_URL = 'http://es-container-local:9200/'
ES_URL = f'http://{GATEWAY_IP}:9210/'
ES_USER = None
ES_PASSWORD = None
ES_TOP_K = 30
ES_INDEX_NAME = 'qanything_es_index' + KB_SUFFIX

# MYSQL_HOST_LOCAL = 'mysql-container-local'
# MYSQL_PORT_LOCAL = 3306
MYSQL_HOST_LOCAL = GATEWAY_IP
MYSQL_PORT_LOCAL = 3316
MYSQL_USER_LOCAL = 'root'
MYSQL_PASSWORD_LOCAL = '123456'
MYSQL_DATABASE_LOCAL = 'qanything'

LOCAL_OCR_SERVICE_URL = "localhost:7001"

LOCAL_PDF_PARSER_SERVICE_URL = "localhost:9009"

LOCAL_RERANK_SERVICE_URL = "localhost:8001"
LOCAL_RERANK_MODEL_NAME = 'rerank'
LOCAL_RERANK_MAX_LENGTH = 512
LOCAL_RERANK_BATCH = 4
LOCAL_RERANK_THREADS = 4
LOCAL_RERANK_PATH = os.path.join(root_path, 'qanything_kernel/dependent_server/rerank_server', 'rerank_model_configs_v0.0.1')
LOCAL_RERANK_MODEL_PATH = os.path.join(LOCAL_RERANK_PATH, "rerank.onnx")

LOCAL_EMBED_SERVICE_URL = "localhost:9001"
LOCAL_EMBED_MODEL_NAME = 'embed'
LOCAL_EMBED_MAX_LENGTH = 1024
LOCAL_EMBED_BATCH = 4
LOCAL_EMBED_THREADS = 4
LOCAL_EMBED_PATH = os.path.join(root_path, 'qanything_kernel/dependent_server/embedding_server', 'embedding_model_configs_v0.0.1')
LOCAL_EMBED_MODEL_PATH = os.path.join(LOCAL_EMBED_PATH, "embed.onnx")

TOKENIZER_PATH = os.path.join(root_path, 'qanything_kernel/connector/llm/tokenizer_files')

DEFAULT_CHILD_CHUNK_SIZE = 400
DEFAULT_PARENT_CHUNK_SIZE = 800
SEPARATORS = ["\n\n", "\n", "。", "，", ",", ".", ""]
MAX_CHARS = 1000000  # 单个文件最大字符数，超过此字符数将上传失败，改大可能会导致解析超时

# llm_config = {
#     # 回答的最大token数，一般来说对于国内模型一个中文不到1个token，国外模型一个中文1.5-2个token
#     "max_token": 512,
#     # 附带的上下文数目
#     "history_len": 2,
#     # 总共的token数，如果遇到电脑显存不够的情况可以将此数字改小，如果低于3000仍然无法使用，就更换模型
#     "token_window": 4096,
#     # 如果报错显示top_p值必须在0到1，可以在这里修改
#     "top_p": 1.0
# }

# Bot
BOT_DESC = "一个简单的问答机器人"
BOT_IMAGE = ""
BOT_PROMPT = """
- 你是一个耐心、友好、专业的机器人，能够回答用户的各种问题。
- 根据知识库内的检索结果，以清晰简洁的表达方式回答问题。
- 不要编造答案，如果答案不在经核实的资料中或无法从经核实的资料中得出，请回答“我无法回答您的问题。”（或者您可以修改为：如果给定的检索结果无法回答问题，可以利用你的知识尽可能回答用户的问题。)
"""
BOT_WELCOME = "您好，我是您的专属机器人，请问有什么可以帮您呢？"

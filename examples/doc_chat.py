import requests
import json
import pandas as pd

# ANSI转义码前缀
RESET = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

# 前景色
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"

# 加载Excel文件
excel_file = "questions.xlsx"
df = pd.read_excel(excel_file)

# API 地址
url = "http://qanything.llm.sxwl.ai:30003/api/local_doc_qa/local_doc_chat"

# 请求头
headers = {
    'Accept': 'text/event-stream,application/json, text/event-stream',
    'Content-Type': 'application/json',
    'Origin': 'http://qanything.llm.sxwl.ai:30003',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
}

# 遍历每个问题
for index, row in df.iterrows():
    question = row['问题']
    reference_answer = row['回答']
    source = row['来源']
    reference_answer_str = '\n    '.join(reference_answer.split('\n')) if isinstance(reference_answer, str) else "None"
    source_str = '\n    '.join(source.split('\n')) if isinstance(source, str) else "None"

    # 构造请求数据
    data = {
        "user_id": "zzp",
        "kb_ids": ["KBbf10352bb4a643f0aeb1951b063ce550"],
        "history": [],
        "question": question,
        "streaming": False
    }

    # 发送请求
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    response_data = response.json()

    # 获取API返回的答案
    api_answer = response_data.get('history', [[None, "无回答"]])[0][1]
    api_answer_str = '\n    '.join(api_answer.split('\n'))

    # 获取来源文件
    sources = response_data.get("source_documents", [])
    source_files = [f"{doc.get('file_name', '未知文件')}    {float(doc.get('score', 0)):.2%}" for doc in sources]
    source_files_str = '\n    '.join(source_files)

    # 打印问题、参考答案、来源、API返回的答案及来源文件
    print(f"问题: {question}")
    print(f"参考答案: \n    {YELLOW}{reference_answer_str}{RESET}")
    print(f"来源: \n    {YELLOW}{source_str}{RESET}")
    print(f"知识库答案: \n    {GREEN}{api_answer_str}{RESET}")
    print(f"知识库来源: \n    {GREEN}{source_files_str}{RESET}")
    print("=" * 50)

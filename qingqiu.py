import sys
import json
import requests

function_list=["document", "new_knowledge_base", "document_parser", "document_parser_embedding", "delete_knowledge_base", "question_rag_search",
           "list_kbs", "list_docs", "delete_docs", "get_total_status", "upload_faqs", "get_qa_info", "get_files_statu"]


# app.add_route(document, "/api/docs", methods=['GET'])   # tags=["接口文档"]
# app.add_route(new_knowledge_base, "/api/qanything/new_knowledge_base", methods=['POST'])  # tags=["新建知识库"]
# app.add_route(delete_knowledge_base, "/api/qanything/delete_knowledge_base", methods=['POST'])  # tags=["删除知识库"] 
# app.add_route(document_parser, "/api/qanything/document_parser", methods=['POST'])  # tags=["解析文件"]
# app.add_route(document_parser_embedding, "/api/qanything/document_parser_embedding", methods=['POST'])  # tags=["解析文件并保存"]
# app.add_route(delete_docs, "/api/qanything/delete_files", methods=['POST'])  # tags=["删除文件"]
# app.add_route(question_rag_search, "/api/qanything/question_rag_search", methods=['POST'])  # tags=["问答接口"]
# app.add_route(list_kbs, "/api/qanything/list_knowledge_base", methods=['POST'])  # tags=["知识库列表"] 
# app.add_route(list_docs, "/api/qanything/list_files", methods=['POST'])  # tags=["文件列表"]
# app.add_route(get_total_status, "/api/qanything/get_total_status", methods=['POST'])  # tags=["获取所有知识库状态"]
# app.add_route(upload_faqs, "/api/qanything/upload_faqs", methods=['POST'])  # tags=["上传FAQ"]
# app.add_route(get_qa_info, "/api/qanything/get_qa_info", methods=['POST'])  # tags=["获取QA信息"]


def document(host, port):
    url = f"http://{host}:{port}/api/docs"
    print("request url: ",url)
    try:
        response = requests.request("GET", url)
        print(response.text)
    except Exception as e:
        print("Error:", e)


def new_knowledge_base(host, port):
    print("new_knowledge_base")
    url = f"http://{host}:{port}/api/qanything/new_knowledge_base"
    headers = {
        "Content-Type": "application/json",
    }

    # payload = {"user_id": "zzp", "kb_id": "KB6dae785cdd5d47a997e890521acbe1c4", "kb_name": "rag2"}
    # payload = {"user_id": "zzp", "kb_id": "KBUNICOMkanjian2", "kb_name": "rag5"}
    payload = {"user_id": "tsh", "kb_id": "KB9NICOMkanjian2", "kb_name": "rag2"}
    print("prompt:", payload)

    try:
        response = requests.request("POST", url, headers=headers, json=payload)
        print(response.text)
    except Exception as e:
        print("Error:", e)


def delete_knowledge_base(host, port):
    url = f"http://{host}:{port}/api/qanything/delete_knowledge_base"
    headers = {
        "Content-Type": "application/json",
    }

    payload = {"user_id": "zzp", "kb_id": "KB6dae785cdd5d47a997e890521acbe1c4"} #, "file_ids": "123"}
    print("prompt:", payload)

    try:
        response = requests.request("POST", url, headers=headers, json=payload)
        print(response.text)
    except Exception as e:
        print("Error:", e)


def list_kbs(host, port):
    url = f"http://{host}:{port}/api/qanything/list_knowledge_base"
    headers = {
        "Content-Type": "application/json",
    }

    payload = {"user_id": "tsh"}
    print("prompt:", payload)

    try:
        response = requests.request("POST", url, headers=headers, json=payload)
        print(response.text)
    except Exception as e:
        print("Error:", e)


def list_docs(host, port):
    url = f"http://{host}:{port}/api/qanything/list_files"
    headers = {
        "Content-Type": "application/json"
    }

    payload = {"user_id": "zzp", "kb_id": "KB31a59121da2d42a1ac0518f129c4a90b"}
    print("prompt:", payload)

    try:
        response = requests.request("POST", url, headers=headers, json=payload)
        print(response.text)
    except Exception as e:
        print("Error:", e)


def get_files_statu(host, port):
    url = f"http://{host}:{port}/api/qanything/get_files_statu"
    headers = {
        "Content-Type": "application/json"
    }

    payload = {"user_id": "zzp", "kb_id": "KB6dae785cdd5d47a997e890521acbe1c5", "file_ids": ["66a71bc57eee49b2a4b61b8448bc4f18","124"]}
    print("prompt:", payload)

    try:
        response = requests.request("POST", url, headers=headers, json=payload)
        print(response.text)

    except Exception as e:
        print("Error:", e)



def document_parser(host, port):
    url = f"http://{host}:{port}/api/qanything/document_parser"

    payload = {"user_id": "zzp"}
    print("prompt:", payload)
    files=[('file', open('/home/darren/文档/联通看见调用.txt','rb'))]

    try:
        response = requests.request("POST", url, data=payload, files=files)
        print(response.text)

    except Exception as e:
        print("Error:", e)


def document_parser_embedding(host, port):
    url = f"http://{host}:{port}/api/qanything/document_parser_embedding"
    # payload = {"user_id": "zzp", "kb_id": "KB6dae785cdd5d47a997e890521acbe1c5", "mode": "soft", "file_ids": "124"}
    # files=[('files',open('./docx_data/12345日报/2024258236895.pdf','rb'))]
    payload = {"user_id": "tsh", "kb_id": "KB9NICOMkanjian2", "mode": "strong", "file_ids": "126"}
    # payload = {"user_id": "tsh", "kb_id": "KB31a59121da2d42a1ac0518f129c4a90b", "mode": "soft", "file_ids": "124"}
    # files=[('files', open('/home/darren/文档/测试langchain.docx','rb'))]
    files=[('files', open('/home/darren/文档/联通看见调用.txt','rb'))]

    print("payload:",payload)
    try:
        response = requests.post(url, data=payload, files=files)
        print(response.text)
    
    except Exception as e:
        print("Error:", e)




def delete_docs(host, port):
    url = f"http://{host}:{port}/api/qanything/delete_files"
    payload = {"user_id": "zzp", "kb_id":"KB6dae785cdd5d47a997e890521acbe1c5", "file_ids": ["124"]}
    print("prompt:", payload)

    try:
        response = requests.request("POST", url, json=payload)
        print(response.text)
    except Exception as e:
        print("Error:", e)


def question_rag_search(host, port):
    url = f"http://{host}:{port}/api/qanything/question_rag_search"

    payload = {"user_id": "zzp", "kb_ids": ["KB6dae785cdd5d47a997e890521acbe1c5"], "question": "如何使用docker"}
    # payload = {'user_id': '6c087e71-3101-48e4-aa29-07ef1a38055b', 'kb_ids': ['KB568e4887f73643a8847227c474903cf3'], 'question': '办理离休干部入户'}
    # headers = {
    #     "Content-Type": "application/json",
    # }

    print("payload:",payload)

    try:
        response = requests.request("POST", url, json=payload)
        print(response.text)
    except Exception as e:
        print("Error:", e)


def get_total_status(host, port):
    url = f"http://{host}:{port}/api/qanything/get_total_status"
    payload = {"user_id": "zzp"}
    print("prompt:", payload)

    try:
        response = requests.request("POST", url, json=payload)
        print(response.text)

    except Exception as e:
        print("Error:", e)


def upload_faqs(host, port):
    url = f"http://{host}:{port}/api/qanything/upload_faqs"
    payload = {
        "user_id": "zzp", 
        "kb_id": "KB6dae785cdd5d47a997e890521acbe1c5", 
        "faqs": [{"question": "如何使用python", "answer": "python是一种编程语言，可以用来开发各种应用程序。参考教程：https://www.runoob.com/python3/python3-tutorial.html"}, 
                 {"question": "如何使用docker", "answer": "Docker是一个开源的应用容器引擎，参考教程：https://www.runoob.com/docker/docker-tutorial.html"}]}
    print("prompt:", payload)

    try:
        response = requests.request("POST", url, json=payload)
        print(response.text)

    except Exception as e:
        print("Error:", e)



def get_qa_info(host, port):
    url = f"http://{host}:{port}/api/qanything/get_qa_info"
    payload = {"user_id": "zzp", "kb_id": "KB6dae785cdd5d47a997e890521acbe1c5"}
    print("prompt:", payload)

    try:
        response = requests.request("POST", url, json=payload)
        print(response.text)

    except Exception as e:
        print("Error:", e)



def usage():
    print("Usage:")
    print(f"python {sys.argv[0]} --api=\"api_name\" [--host=0.0.0.0 --port=8777]")
    print("可用的api：", function_list)





if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8777
    api = ""
    usage()
    
    for arg in sys.argv[1:]:
        if arg.startswith('--host='):
            host = arg.split('=')[1]
        elif arg.startswith('--port='):
            port = int(arg.split('=')[1])
        elif arg.startswith('--api='):
            api = arg.split('=')[1]
            
    print(f"host: {host}, port: {port}")
    print(f"api: {api}")

    if api in function_list:
        eval(api)(host, port)
    else:
        # raise error
        print(f"api:{api} is not exsit.")
        




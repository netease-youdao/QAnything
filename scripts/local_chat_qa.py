import sys
import requests
import time

response_times = []


def send_request(ques):
    # url = 'https://qanything-test.site.youdao.com/api/local_doc_qa/local_doc_chat'
    url = 'http://localhost:8777/api/local_doc_qa/local_doc_chat'
    headers = {
        'content-type': 'application/json'
    }
    data = {
        "user_id": "liujx_265",
        "kb_ids": ["KBf652e9e379c546f1894597dcabdc8e47"],
        "question": ques,
        "rerank": False,
        "history": []
    }
    try:
        start_time = time.time()
        response = requests.post(url=url, headers=headers, json=data, timeout=60)
        end_time = time.time()
        response_times.append(end_time - start_time)
        res = response.json()
        print(res['response'])
        print(f"响应状态码: {response.status_code}, 响应时间: {end_time - start_time}秒")
    except Exception as e:
        print(f"请求发送失败: {e}")


if __name__ == '__main__':
    ques = sys.argv[1]
    send_request(ques)

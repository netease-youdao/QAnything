import requests
import json

url = "http://0.0.0.0:8777/api/local_doc_qa/new_knowledge_base"
headers = {
    "Content-Type": "application/json"
}
data = {
    "user_id": "zzp",
    "kb_name": "kb_test"
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.status_code)
print(response.text)

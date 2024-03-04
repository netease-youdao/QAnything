import requests
import json

url = "http://0.0.0.0:8777/api/local_doc_qa/list_files"
headers = {
    "Content-Type": "application/json"
}
data = {
    "user_id": "zzp",
    "kb_id": "KBccd94e086e8d458fa1ed6ca3a93655d9"
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.status_code)
print(response.text)

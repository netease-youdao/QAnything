import requests
import json
import sys

kb_id = sys.argv[1]
url = "http://0.0.0.0:8777/api/local_doc_qa/list_files"
headers = {
    "Content-Type": "application/json"
}
data = {
    "user_id": "zzp",
    "kb_id": kb_id
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.status_code)
print(response.text)

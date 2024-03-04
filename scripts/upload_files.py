import os
import requests
import sys

url = "http://0.0.0.0:8777/api/local_doc_qa/upload_files"
data = {
    "user_id": "zzp",
    "kb_id": "KBccd94e086e8d458fa1ed6ca3a93655d9"
}

files = []
files.append(("files", open(sys.argv[1], "rb")))

response = requests.post(url, files=files, data=data)
print(response.text)

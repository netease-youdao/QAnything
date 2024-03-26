import os
import requests
import sys

url = "http://localhost:8777/api/local_doc_qa/upload_files"
data = {
    "user_id": "zzp",
    "kb_id": "KB8ee2b2ab902a4ea2b4b42b623790f3e8"
}

files = []
files.append(("files", open(sys.argv[1], "rb")))

response = requests.post(url, files=files, data=data)
print(response.text)

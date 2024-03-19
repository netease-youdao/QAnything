import requests
import sys

kb_id = sys.argv[1]
url = "http://0.0.0.0:8777/api/local_doc_qa/upload_files"
data = {
    "user_id": "zzp",
    "kb_id": kb_id
}

files = []
files.append(("files", open(sys.argv[2], "rb")))

response = requests.post(url, files=files, data=data)
print(response.text)

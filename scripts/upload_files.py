import requests
import sys

kb_id = "KBb66a202b3cfe48869763107f9b967427"
url = "http://localhost:8777/api/local_doc_qa/upload_files"
data = {
    "user_id": "zzp",
    "kb_id": kb_id,
}

files = []
files.append(("files", open(sys.argv[1], "rb")))

response = requests.post(url, files=files, data=data)
print(response.text)
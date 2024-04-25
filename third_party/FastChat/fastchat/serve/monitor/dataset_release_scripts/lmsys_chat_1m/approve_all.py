import requests

headers = {"authorization": "Bearer hf_XXX"}

url = "https://huggingface.co/api/datasets/lmsys/lmsys-chat-1m/user-access-request/pending"
a = requests.get(url, headers=headers, timeout=60)

for u in a.json():
    user = u["user"]["user"]
    url = "https://huggingface.co/api/datasets/lmsys/lmsys-chat-1m/user-access-request/grant"
    ret = requests.post(url, headers=headers, json={"user": user}, timeout=60)
    print(user, ret.status_code)
    assert ret.status_code == 200

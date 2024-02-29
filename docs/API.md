# QAnything 接口文档

- [QAnything 接口文档](#qanything-接口文档)
  - [全局参数](#全局参数)
  - [新建知识库（POST）](#新建知识库post)
    - [URL：http://{your_host}:8777/api/local_doc_qa/new_knowledge_base ](#urlhttpyour_host8777apilocal_doc_qanew_knowledge_base-)
    - [新建知识库请求参数（Body）](#新建知识库请求参数body)
    - [新建知识库请求示例](#新建知识库请求示例)
    - [新建知识库响应示例](#新建知识库响应示例)
  - [上传文件（POST）](#上传文件post)
    - [URL：http://{your_host}:8777/api/local_doc_qa/upload_files ](#urlhttpyour_host8777apilocal_doc_qaupload_files-)
    - [上传文件请求参数（Body）](#上传文件请求参数body)
    - [上传文件同步请求示例](#上传文件同步请求示例)
    - [上传文件异步请求示例](#上传文件异步请求示例)
    - [上传文件响应示例](#上传文件响应示例)
  - [上传网页文件（POST）](#上传网页文件post)
    - [URL：http://{your_host}:8777/api/local_doc_qa/upload_weblink](#urlhttpyour_host8777apilocal_doc_qaupload_weblink)
    - [上传网页文件请求参数（Body）](#上传网页文件请求参数body)
    - [上传网页文件请求示例](#上传网页文件请求示例)
    - [上传网页文件响应示例](#上传网页文件响应示例)
  - [查看知识库（POST）](#查看知识库post)
    - [URL：http://{your_host}:8777/api/local_doc_qa/list_knowledge_base](#urlhttpyour_host8777apilocal_doc_qalist_knowledge_base)
    - [查看知识库请求参数（Body）](#查看知识库请求参数body)
    - [查看知识库请求示例](#查看知识库请求示例)
    - [查看知识库响应示例](#查看知识库响应示例)
  - [获取文件列表（POST）](#获取文件列表post)
    - [URL: http://{your_host}:8777/api/local_doc_qa/list_files](#url-httpyour_host8777apilocal_doc_qalist_files)
    - [获取文件列表请求参数（Body）](#获取文件列表请求参数body)
    - [获取文件列表请求示例](#获取文件列表请求示例)
    - [获取文件列表响应示例](#获取文件列表响应示例)
  - [问答（POST）](#问答post)
    - [ URL：http://{your_host}:8777/api/local_doc_qa/local_doc_chat](#-urlhttpyour_host8777apilocal_doc_qalocal_doc_chat)
    - [ 问答请求参数（Body）](#-问答请求参数body)
    - [ 问答非流式请求示例](#-问答非流式请求示例)
    - [ 问答非流式响应示例](#-问答非流式响应示例)
    - [ 问答流式请求示例](#-问答流式请求示例)
    - [ 问答流式响应示例](#-问答流式响应示例)
  - [ 删除文件（POST）](#-删除文件post)
    - [ URL：http://{your_host}:8777/api/local_doc_qa/delete_files](#-urlhttpyour_host8777apilocal_doc_qadelete_files)
    - [ 删除文件请求参数（Body）](#-删除文件请求参数body)
    - [ 删除文件请求示例](#-删除文件请求示例)
    - [ 删除文件响应示例](#-删除文件响应示例)
  - [ 删除知识库（POST）](#-删除知识库post)
    - [ 删除知识库请求参数（Body）](#-删除知识库请求参数body)
    - [ 删除知识库请求示例](#-删除知识库请求示例)
    - [ 删除知识库响应示例](#-删除知识库响应示例)
  - [ 获取所有知识库状态（POST）](#-获取所有知识库状态post)
    - [ URL：http://{your_host}:8777/api/local_doc_qa/get_total_status](#-urlhttpyour_host8777apilocal_doc_qaget_total_status)
    - [获取所有知识库状态请求参数（Body）](#获取所有知识库状态请求参数body)
    - [获取所有知识库状态请求示例](#获取所有知识库状态请求示例)
    - [获取所有知识库状态响应示例](#获取所有知识库状态响应示例)
  - [清理知识库（POST）](#清理知识库post)
    - [URL：http://{your_host}:8777/api/local_doc_qa/clean_files_by_status](#urlhttpyour_host8777apilocal_doc_qaclean_files_by_status)
    - [清理知识库请求参数（Body）](#清理知识库请求参数body)
    - [清理知识库请求示例](#清理知识库请求示例)
    - [清理知识库响应示例](#清理知识库响应示例)
  - [重命名知识库（POST）](#重命名知识库post)
    - [URL：http://{your_host}:8777/api/local_doc_qa/rename_knowledge_base](#urlhttpyour_host8777apilocal_doc_qarename_knowledge_base)
    - [重命名知识库请求参数（Body）](#重命名知识库请求参数body)
    - [重命名知识库请求示例](#重命名知识库请求示例)
    - [重命名知识库响应示例](#重命名知识库响应示例)

## <h2><p id="全局参数">全局参数</p></h2>

我们提供用户区分的功能，每个接口中有 user_id 的参数，如果需要请传入 user_id 的值。

user_id 需要满足： 以字母开头，只允许包含字母，数字或下划线。

如果不需要区分不同用户，传入 user_id="zzp"即可

`注意当且仅当user_id="zzp"时通过API传入的信息与前端页面互通`

## <h2><p id="新建知识库post">新建知识库（POST）</p></h2>

### <h3><p id="urlhttpyour_host8777apilocal_doc_qanew_knowledge_base-">URL：<http://{your_host}:8777/api/local_doc_qa/new_knowledge_base> </p></h3>

### <h3><p id="新建知识库请求参数body">新建知识库请求参数（Body）</p></h3>

| 参数名  | 示例参数值 | 是否必填 | 参数类型 | 描述说明                                |
| ------- | ---------- | -------- | -------- | --------------------------------------- |
| user_id | "zzp"      | 是       | String   | 用户 id （如需使用前端填 zzp 不要更换） |
| kb_name | "kb_test"  | 是       | String   | 知识库名称 （可以随意指定）             |

### <h3><p id="新建知识库请求示例">新建知识库请求示例</p></h3>

```python
import requests
import json

url = "http://{your_host}:8777/api/local_doc_qa/new_knowledge_base"
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
```

### <h3><p id="新建知识库响应示例">新建知识库响应示例</p></h3>

```json
{
  "code": 200, //状态码
  "msg": "success create knowledge base KBd728811ed16b46f9a2946e28dd5c9939", //提示信息
  "data": {
    "kb_id": "KB4c50de98d6b548af9aa0bc5e10b2e3a7", //知识库id
    "kb_name": "kb_test", //知识库名称
    "timestamp": "202401251057" // 创建时间戳
  }
}
```

## <h2><p id="上传文件post">上传文件（POST）</p></h2>

### <h3><p id="urlhttpyour_host8777apilocal_doc_qaupload_files-">URL：<http://{your_host}:8777/api/local_doc_qa/upload_files></p> </h3>

Content-Type: multipart/form-data

### <h3><p id="上传文件请求参数body">上传文件请求参数（Body）</p></h3>

| 参数名  | 参数值                             | 是否必填 | 参数类型 | 描述说明                                                                                                |
| ------- | ---------------------------------- | -------- | -------- | ------------------------------------------------------------------------------------------------------- |
| files   | 文件二进制                         | 是       | File     | 需要上传的文件，可多选，目前仅支持[md,txt,pdf,jpg,png,jpeg,docx,xlsx,pptx,eml,csv]                      |
| user_id | zzp                                | 是       | String   | 用户 id                                                                                                 |
| kb_id   | KBb1dd58e8485443ce81166d24f6febda7 | 是       | String   | 知识库 id                                                                                               |
| mode    | soft                               | 否       | String   | 上传模式，soft：知识库内存在同名文件时当前文件不再上传，strong：文件名重复的文件强制上传，默认值为 soft |

### <h3><p id="上传文件同步请求示例">上传文件同步请求示例</p></h3>

```python
import os
import requests

url = "http://{your_host}:8777/api/local_doc_qa/upload_files"
folder_path = "./docx_data"  # 文件所在文件夹，注意是文件夹！！
data = {
    "user_id": "zzp",
    "kb_id": "KB6dae785cdd5d47a997e890521acbe1c9",
		"mode": "soft"
}

files = []
for root, dirs, file_names in os.walk(folder_path):
    for file_name in file_names:
        if file_name.endswith(".md"):  # 这里只上传后缀是md的文件，请按需修改，支持类型：
            file_path = os.path.join(root, file_name)
            files.append(("files", open(file_path, "rb")))

response = requests.post(url, files=files, data=data)
print(response.text)
```

### <h3><p id="上传文件异步请求示例">上传文件异步请求示例</p></h3>

```python
import argparse
import os
import sys
import json
import aiohttp
import asyncio
import time
import random
import string

files = []
for root, dirs, file_names in os.walk("./docx_data"):  # 文件夹
    for file_name in file_names:
        if file_name.endswith(".docx"):  # 只上传docx文件
            file_path = os.path.join(root, file_name)
            files.append(file_path)
print(len(files))
response_times = []

async def send_request(round_, files):
    print(len(files))
    url = 'http://{your_host}:8777/api/local_doc_qa/upload_files'
    data = aiohttp.FormData()
    data.add_field('user_id', 'zzp')
    data.add_field('kb_id', 'KBf1dafefdb08742f89530acb7e9ed66dd')
    data.add_field('mode', 'soft')

    total_size = 0
    for file_path in files:
        file_size = os.path.getsize(file_path)
        total_size += file_size
        data.add_field('files', open(file_path, 'rb'))
    print('size:', total_size / (1024 * 1024))
    try:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                end_time = time.time()
                response_times.append(end_time - start_time)
                print(f"round_:{round_}, 响应状态码: {response.status}, 响应时间: {end_time - start_time}秒")
    except Exception as e:
        print(f"请求发送失败: {e}")

async def main():
    start_time = time.time()
    num = int(sys.argv[1])  // 一次上传数量，http协议限制一次请求data不能大于100M，请自行控制数量
    round_ = 0
    r_files = files[:num]
    tasks = []
    task = asyncio.create_task(send_request(round_, r_files))
    tasks.append(task)
    await asyncio.gather(*tasks)

    print(f"请求完成")
    end_time = time.time()
    total_requests = len(response_times)
    total_time = end_time - start_time
    qps = total_requests / total_time
    print(f"total_time:{total_time}")

if __name__ == '__main__':
    asyncio.run(main())
```

### <h3><p id="上传文件响应示例">上传文件响应示例</p></h3>

```json
{
  "code": 200, //状态码
  "msg": "success，后台正在飞速上传文件，请耐心等待", //提示信息
  "data": [
    {
      "file_id": "1b6c0781fb9245b2973504cb031cc2f3", //文件id
      "file_name": "网易有道智云平台产品介绍2023.6.ppt", //文件名
      "status": "gray", //文件状态（red：入库失败-切分失败，green，成功入库，yellow：入库失败-milvus失败，gray：正在入库）
      "bytes": 17925, //文件大小（字节数）
      "timestamp": "202401251056" // 上传时间
    },
    {
      "file_id": "aeaec708c7a34952b7de484fb3374f5d",
      "file_name": "有道知识库问答产品介绍.pptx",
      "status": "gray",
      "bytes": 12928, //文件大小（字节数）
      "timestamp": "202401251056" // 上传时间
    }
  ] //文件列表
}
```

## <h2><p id="上传网页文件post">上传网页文件（POST）</p></h2>

### <h3><p id="urlhttpyour_host8777apilocal_doc_qaupload_weblink">URL：<http://{your_host}:8777/api/local_doc_qa/upload_weblink></p></h3>

### <h3><p id="上传网页文件请求参数body">上传网页文件请求参数（Body）</p></h3>

| 参数名  | 参数值                                                          | 是否必填 | 参数类型 | 描述说明                                                                                                |
| ------- | --------------------------------------------------------------- | -------- | -------- | ------------------------------------------------------------------------------------------------------- |
| url     | "https://ai.youdao.com/DOCSIRMA/html/trans/api/wbfy/index.html" | 是       | String   | html 网址，只支持无需登录的网站                                                                         |
| user_id | zzp                                                             | 是       | String   | 用户 id                                                                                                 |
| kb_id   | KBb1dd58e8485443ce81166d24f6febda7                              | 是       | String   | 知识库 id                                                                                               |
| mode    | soft                                                            | 否       | String   | 上传模式，soft：知识库内存在同名文件时当前文件不再上传，strong：文件名重复的文件强制上传，默认值为 soft |

### <h3><p id="上传网页文件请求示例">上传网页文件请求示例</p></h3>

```python
import requests
import json

url = "http://{your_host}:8777/api/local_doc_qa/upload_weblink"
headers = {
    "Content-Type": "application/json"
}
data = {
    "user_id": "zzp",
		"kb_id": "KBb1dd58e8485443ce81166d24f6febda7",
		"url": "https://ai.youdao.com/DOCSIRMA/html/trans/api/wbfy/index.html"
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.status_code)
print(response.text)
```

### <h3><p id="上传网页文件响应示例">上传网页文件响应示例</p></h3>

```json
{
  "code": 200,
  "msg": "success，后台正在飞速上传文件，请耐心等待",
  "data": [
    {
      "file_id": "9a49392e633d4c6f87e0af51e8c80a86",
      "file_name": "https://ai.youdao.com/DOCSIRMA/html/trans/api/wbfy/index.html",
      "status": "gray",
      "bytes": 0, // 网页文件无法显示大小
      "timestamp": "202401261809"
    }
  ]
}
```

## <h2><p id="查看知识库post">查看知识库（POST）</p></h2>

### <h3><p id="urlhttpyour_host8777apilocal_doc_qalist_knowledge_base">URL：<http://{your_host}:8777/api/local_doc_qa/list_knowledge_base></p></h3>

### <h3><p id="查看知识库请求参数body">查看知识库请求参数（Body）</p></h3>

| 参数名  | 示例参数值 | 是否必填 | 参数类型 | 描述说明 |
| ------- | ---------- | -------- | -------- | -------- |
| user_id | "zzp"      | 是       | String   | 用户 id  |

### <h3><p id="查看知识库请求示例">查看知识库请求示例</p></h3>

```python
import requests
import json

url = "http://{your_host}:8777/api/local_doc_qa/list_knowledge_base"
headers = {
    "Content-Type": "application/json"
}
data = {
    "user_id": "zzp"
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.status_code)
print(response.text)
```

### <h3><p id="查看知识库响应示例">查看知识库响应示例</p></h3>

```json
{
  "code": 200, //状态码
  "data": [
    {
      "kb_id": "KB973d4aea07f14c60ae1974404a636ad4", //知识库id
      "kb_name": "dataset_s_1" //知识库名称
    }
  ] //知识库列表
}
```

## <h2><p id="获取文件列表post">获取文件列表（POST）</p></h2>

### <h3><p id="url-httpyour_host8777apilocal_doc_qalist_files">URL: <http://{your_host}:8777/api/local_doc_qa/list_files></p></h3>

### <h3><p id="获取文件列表请求参数body">获取文件列表请求参数（Body）</p></h3>

| 参数名  | 示例参数值                           | 是否必填 | 参数类型 | 描述说明  |
| ------- | ------------------------------------ | -------- | -------- | --------- |
| user_id | "zzp"                                | 是       | String   | 用户 id   |
| kb_id   | "KBb1dd58e8485443ce81166d24f6febda7" | 是       | String   | 知识库 id |

### <h3><p id="获取文件列表请求示例">获取文件列表请求示例</p></h3>

```python
import requests
import json

url = "http://{your_host}:8777/api/local_doc_qa/list_files"
headers = {
    "Content-Type": "application/json"
}
data = {
	"user_id": "zzp",  //用户id
	"kb_id": "KBb1dd58e8485443ce81166d24f6febda7" //知识库id
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.status_code)
print(response.text)
```

### <h3><p id="获取文件列表响应示例">获取文件列表响应示例</p></h3>

```json
{
	"code": 200, //状态码
	"msg": "success", //提示信息
	"data": {
		"total": {  // 知识库所有文件状态
			"green": 100,
			"red": 1,
			"gray": 1,
			"yellow": 1,
		},
		"details": {  // 每个文件的具体状态
			{
				"file_id": "21a9f13832594b0f936b62a54254543b", //文件id
				"file_name": "有道知识库问答产品介绍.pptx", //文件名
				"status": "green", //文件状态（red：入库失败-切分失败，green，成功入库，yellow：入库失败-milvus失败，gray：正在入库）
				"bytes": 177925,
				"content_length": 3059,  // 文件解析后字符长度，用len()计算
				"timestamp": "202401261708",
				"msg": "上传成功"
			},
			{
				"file_id": "333e69374a8d4b9bac54f274291f313e", //文件id
				"file_name": "网易有道智云平台产品介绍2023.6.ppt", //文件名
				"status": "green", //文件状态（red：入库失败-切分失败，green，成功入库，yellow：入库失败-milvus失败，gray：正在入库）
				"bytes": 12379,
				"content_length": 3239,  // 文件解析后字符长度，用len()计算
				"timestamp": "202401261708",
				"msg": "上传成功"
			}
		}
		// ...
	}
}
```

## <h2><p id="问答post">问答（POST）</p></h2>

### <h3><p id="-urlhttpyour_host8777apilocal_doc_qalocal_doc_chat"> URL：<http://{your_host}:8777/api/local_doc_qa/local_doc_chat></p></h3>

### <h3><p id="-问答请求参数body"> 问答请求参数（Body）</p></h3>

| 参数名    | 示例参数值                                                                   | 是否必填 | 参数类型     | 描述说明                                 |
| --------- | ---------------------------------------------------------------------------- | -------- | ------------ | ---------------------------------------- |
| user_id   | "zzp"                                                                        | 是       | String       | 用户 id                                  |
| kb_ids    | ["KBb1dd58e8485443ce81166d24f6febda7", "KB633c69d07a2446b39b3a38e3628b8ada"] | 是       | Array        | 知识库 id 的列表，支持多个知识库联合问答 |
| question  | "保险单号是多少？"                                                           | 是       | String       | 知识库 id 的列表，支持多个知识库联合问答 |
| history   | [["question1","answer1"],["question2","answer2"]]                            | 否       | Array[Array] | 历史对话                                 |
| rerank    | True                                                                         | 否       | Bool         | 是否开启 rerank，默认为 True             |
| streaming | False                                                                        | 否       | Bool         | 是否开启流式输出，默认为 False           |

### <h3><p id="-问答非流式请求示例"> 问答非流式请求示例</p></h3>

```python
import sys
import requests
import time

def send_request():
    url = 'http://{your_host}:8777/api/local_doc_qa/local_doc_chat'
    headers = {
        'content-type': 'application/json'
    }
    data = {
        "user_id": "zzp",
        "kb_ids": ["KBf652e9e379c546f1894597dcabdc8e47"],
        "question": "一嗨案件中保险单号是多少？",
    }
    try:
        start_time = time.time()
        response = requests.post(url=url, headers=headers, json=data, timeout=60)
        end_time = time.time()
        res = response.json()
        print(res['response'])
        print(f"响应状态码: {response.status_code}, 响应时间: {end_time - start_time}秒")
    except Exception as e:
        print(f"请求发送失败: {e}")


if __name__ == '__main__':
    send_request()
```

### <h3><p id="-问答非流式响应示例"> 问答非流式响应示例</p></h3>

```json
{
  "code": 200, //状态码
  "msg": "success", //提示信息
  "question": "一嗨案件中保险单号是多少？", //用户问题
  "response": "保险单号是601J312512022000536。", //模型回答
  "history": [["一嗨案件中保险单号是多少？", "保险单号是601J312512022000536。"]], //历史对话：List[str]，至少会显示当前轮对话
  "source_documents": [
    {
      "file_id": "f9b794233c304dd5b5a010f2ead67f51", //文本内容对应的文件id
      "file_name": "一嗨案件支付三者车损、人伤保险赔款及权益转让授权书.docx", //文本内容对应的文件名
      "content": "未支付第三者车损、人伤赔款及同意直赔第三者确认书 华泰财产保险有限公司  北京   分公司： 本人租用一嗨在贵司承保车辆（车牌号：京KML920）商业险保单号： 601J312512022000536、交强险保单号:  601J310022022000570， 对 2023 年 03 月 25日所发生的保险事故（事故号：  9010020230325004124）中所涉及的交强险和商业险的保险赔款总金额 (依：三者京AFT5538定损金额)， 同意支付给本次事故中第三者方。 在此本人确认：本人从未支付给第三者方任何赔偿，且承诺不就本次事故再向贵司及一嗨进行索赔。 同时本人保证如上述内容不属实、违反承诺，造成保险人损失的，由本人承担赔偿责任。 确认人（驾驶员）签字:              第三者方签字: 联系电话：                        联系电话： 确认日期：    年    月    日", //文本内容
      "retrieval_query": "一嗨案件中保险单号是多少？", //文本内容对应的问题
      "score": "3.5585756", //相关性得分，分数越高越相关
      "embed_version": "local_v0.0.1_20230525_6d4019f1559aef84abc2ab8257e1ad4c" //embedding模型版本号
    },
    {
      "file_id": "4867d3fcf395414083c896993fae291e",
      "file_name": "有道领世辅导日常问题处理手册.pdf",
      "content": "html? t  高二数学体系规划学习卡·上（2期）  ype=busy&onlyForm=true  高二语文  高二数学  高二英语  高二物理  半年卡王伟  ￥4700. 00  12:05  李秋颖  高二语文全体系规划学习卡·上（2期）  哈咯盛哥！   为保障你的合法权益与服务质量，本会话内容将可能被对  半年卡李秋颖  ￥4700. 00  支付失败  语文李秋颖  方所在的企业存档，若不同意可拒绝存档。 查看服务须知  怎么办?   应付款￥9400. 00  报名链接：  https://aike-sale. ydshengxue. com/p  有未支付的订单  查看详情  迷续支付  roductList/34  请在“我的订单“中继续支付  取消  查看订单  我  是  哈哈  向  没事  哈哈！   β  123  空格  发送  合计￥0. 00  去结算  选购单  02023/4/20 17:25  有道领世辅导⽇常问题处理⼿册  https://shimo. youdao. com/docs/ky6NGnVY6zMOIXKJ  4/30  8、ios ⽤户反馈领世 app ⾥充值问题  ios的领世app最新版⽀持微信和⽀付宝⽀付， 引导⽤户通过续报H5链接或者最新版领世app⾥  使⽤微信或⽀付宝⽀付即可。   9、⽤户需要使⽤花呗⽀付，但是提示不可⽤是怎么回事  ⽀付宝有单独的⻛控策略，⽤户在使⽤花呗时需要遵守⽀付宝的相关规定和限制，否则  花呗⽀付可能会被关闭或者受限。 如果⽤户在使⽤花呗⽀付时出现问题，建议⽤户拨打  ⽀付宝客服热线 95188 进⾏咨询和核实个⼈花呗状态，以便更好地解决问题。   10、我的微信提示交易有⻛险该怎么办呢？   如果微信提示交易有⻛险，可能是因为您的账户存在异常或者安全⻛险。",
      "retrieval_query": "一嗨案件中保险单号是多少？",
      "score": "-4.617383",
      "embed_version": "local_v0.0.1_20230525_6d4019f1559aef84abc2ab8257e1ad4c"
    },
    {
      "file_id": "4867d3fcf395414083c896993fae291e",
      "file_name": "有道领世辅导日常问题处理手册.pdf",
      "content": "立即订阅  我已阅读并同意有道领世《服务条款》  《",
      "retrieval_query": "一嗨案件中保险单号是多少？",
      "score": "-8.9622345",
      "embed_version": "local_v0.0.1_20230525_6d4019f1559aef84abc2ab8257e1ad4c"
    }
  ] //知识库相关文本内容
}
```

### <h3><p id="-问答流式请求示例"> 问答流式请求示例</p></h3>

```python
import os
import json
import requests
import time
import random
import string
import argparse

def stream_requests(data_raw):
    url = 'http://{your_host}:8777/api/local_doc_qa/local_doc_chat'
    response = requests.post(
        url,
        json=data_raw,
        timeout=60,
        stream=True
    )
    for line in response.iter_lines(decode_unicode=False, delimiter=b"\n\n"):
        if line:
            yield line

def test():
    data_raw = {
      "kb_ids": ["KB633c69d07a2446b39b3a38e3628b8ada"],
      "question": "你好",
      "user_id": "zzp",
      "streaming": True,
      "history": []
    }
    for i, chunk in enumerate(stream_requests(data_raw)):
        if chunk:
            chunkstr = chunk.decode("utf-8")[6:]
            chunkjs = json.loads(chunkstr)
            print(chunkjs)

if __name__ == '__main__':
    test()
```

### <h3><p id="-问答流式响应示例"> 问答流式响应示例</p></h3>

```Text
{'code': 200, 'msg': 'success', 'question': '', 'response': '', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '你', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '好', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '！', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '我', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '是', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '有', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '道', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '开', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '发', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '的', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '大', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '模', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '型', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '。', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '有', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '什', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '么', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '问题', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '我', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '可以', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '帮', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '助', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '你', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '解', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '答', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '吗', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '？', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '', 'response': '', 'history': [], 'source_documents': []}
{'code': 200, 'msg': 'success', 'question': '你好', 'response': '你好！我是有道开发的大模型。有什么问题我可以帮助你解答吗？', 'history': [['你好', '你好！我是有道开发的大模型。有什么问题我可以帮助你解答吗？']], 'source_documents':[]}
```

## <h2><p id="-删除文件post"> 删除文件（POST）</p></h2>

### <h3><p id="-urlhttpyour_host8777apilocal_doc_qadelete_files"> URL：<http://{your_host}:8777/api/local_doc_qa/delete_files></p></h3>

### <h3><p id="-删除文件请求参数body"> 删除文件请求参数（Body）</p></h3>

| 参数名   | 示例参数值                           | 是否必填 | 参数类型 | 描述说明                      |
| -------- | ------------------------------------ | -------- | -------- | ----------------------------- |
| user_id  | "zzp"                                | 是       | String   | 用户 id                       |
| kb_id    | "KB1271e71c36ec4028a6542586946a3906" | 是       | String   | 知识库 id                     |
| file_ids | ["73ff7cf76ff34c8aa3a5a0b4ba3cf534"] | 是       | Array    | 要删除文件的 id，支持批量删除 |

### <h3><p id="-删除文件请求示例"> 删除文件请求示例</p></h3>

```python
import requests
import json

url = "http://{your_host}:8777/api/local_doc_qa/delete_files"
headers = {
    "Content-Type": "application/json"
}
data = {
	"user_id": "zzp", //用户id
	"kb_id": "KB1271e71c36ec4028a6542586946a3906", //知识库id
	"file_ids": [
		"73ff7cf76ff34c8aa3a5a0b4ba3cf534"
	] //文件id列表
}
response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.status_code)
print(response.text)
```

### <h3><p id="-删除文件响应示例"> 删除文件响应示例</p></h3>

```json
{
  "code": 200, //状态码
  "msg": "documents ['73ff7cf76ff34c8aa3a5a0b4ba3cf534'] delete success" //提示信息
}
```

## <h3><p id="-删除知识库post"> 删除知识库（POST）</p></h3>

URL：<http://{your_host}:8777/api/local_doc_qa/delete_knowledge_base>

### <h3><p id="-删除知识库请求参数body"> 删除知识库请求参数（Body）</p></h3>

| 参数名  | 示例参数值                             | 是否必填 | 参数类型 | 描述说明                        |
| ------- | -------------------------------------- | -------- | -------- | ------------------------------- |
| user_id | "zzp"                                  | 是       | String   | 用户 id                         |
| kb_ids  | ["KB1cd81f2bc515437294bda1934a20b235"] | 是       | Array    | 要删除的知识库 id，支持批量删除 |

### <h3><p id="-删除知识库请求示例"> 删除知识库请求示例</p></h3>

```python
import requests
import json

url = "http://{your_host}:8777/api/local_doc_qa/delete_knowledge_base"
headers = {
    "Content-Type": "application/json"
}
data = {
	"user_id": "zzp", //用户id
	"kb_ids": [
		"KB1cd81f2bc515437294bda1934a20b235"
	] //知识库id列表
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.status_code)
print(response.text)
```

### <h3><p id="-删除知识库响应示例"> 删除知识库响应示例</p></h3>

```json
{
  "code": 200, //状态码
  "msg": "Knowledge Base [('KB1cd81f2bc515437294bda1934a20b235',)] delete success" //提示信息
}
```

## <h2><p id="-获取所有知识库状态post"> 获取所有知识库状态（POST）</p></h2>

### <h3><p id="-urlhttpyour_host8777apilocal_doc_qaget_total_status"> URL：<http://{your_host}:8777/api/local_doc_qa/get_total_status></p></h3>

### <h3><p id="获取所有知识库状态请求参数body">获取所有知识库状态请求参数（Body）</p></h3>

| 参数名  | 示例参数值 | 是否必填 | 参数类型 | 描述说明 |
| ------- | ---------- | -------- | -------- | -------- |
| user_id | "zzp"      | 是       | String   | 用户 id  |

### <h3><p id="获取所有知识库状态请求示例">获取所有知识库状态请求示例</p></h3>

```python
import requests
import json

url = "http://{your_host}:8777/api/local_doc_qa/get_total_status"
headers = {
    "Content-Type": "application/json"
}
data = {
	"user_id": "zzp"
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.status_code)
print(response.text)
```

### <h3><p id="获取所有知识库状态响应示例">获取所有知识库状态响应示例</p></h3>

```json
{
  "code": 200,
  "status": {
    "zzp": {
      "默认知识库KB0015df77a8eb46f6951de513392dc250": {
        // kb_name + kb_id
        "green": 10,
        "yellow": 0,
        "red": 0,
        "gray": 0
      },
      "知识问答KB6a4534f753b54c4198b62770e26b1a92": {
        "green": 16,
        "yellow": 0,
        "red": 0,
        "gray": 0
      }
    }
  }
}
```

## <h3><p id="清理知识库post">清理知识库（POST）</p></h3>

### <h3><p id="urlhttpyour_host8777apilocal_doc_qaclean_files_by_status">URL：<http://{your_host}:8777/api/local_doc_qa/clean_files_by_status></p></h3>

### <h3><p id="清理知识库请求参数body">清理知识库请求参数（Body）</p></h3>

| 参数名  | 示例参数值                                                                   | 是否必填 | 参数类型 | 描述说明                                                                                                    |
| ------- | ---------------------------------------------------------------------------- | -------- | -------- | ----------------------------------------------------------------------------------------------------------- |
| user_id | "zzp"                                                                        | 是       | String   | 用户 id                                                                                                     |
| status  | "gray"                                                                       | 是       | String   | 用清理何种状态的文件，red：入库失败-切分失败，green，成功入库，yellow：入库失败-milvus 失败，gray：正在入库 |
| kd_ids  | ["KB0015df77a8eb46f6951de513392dc250", "KB6a4534f753b54c4198b62770e26b1a92"] | 否       | String   | 需要清理的知识库 id 列表，默认为[]，此时默认清理 user_id 下全部的知识库                                     |

### <h3><p id="清理知识库请求示例">清理知识库请求示例</p></h3>

```python
import requests
import json

url = "http://{your_host}:8777/api/local_doc_qa/clean_files_by_status"
headers = {
    "Content-Type": "application/json"
}
data = {
	"user_id": "zzp",
	"status": "gray"
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.status_code)
print(response.text)
```

### <h3><p id="清理知识库响应示例">清理知识库响应示例</p></h3>

```json
{
  "code": 200,
  "msg": "delete gray files success",
  "data": [
    "升学百科benchmark_20231120.xlsx_20231121235103.txt",
    "网页翻译-API文档.md",
    "api_inter_virtualPerson_execution_3451_executionrersult_4699_10_stats.csv",
    "中文成语-如火如荼.png",
    "有道知识库问答产品介绍.pptx",
    "韦小宝身份证.jpg",
    "fake-email.eml",
    "有道领世辅导日常问题处理手册.pdf",
    "xx案件支付三者车损人伤保险赔款及权益转让授权书.docx"
  ]
}
```

## <h2><p id="重命名知识库post">重命名知识库（POST）</p></h2>

### <h3><p id="urlhttpyour_host8777apilocal_doc_qarename_knowledge_base">URL：<http://{your_host}:8777/api/local_doc_qa/rename_knowledge_base></p></h3>

### <h3><p id="重命名知识库请求参数body">重命名知识库请求参数（Body）</p></h3>

| 参数名      | 示例参数值                           | 是否必填 | 参数类型 | 描述说明              |
| ----------- | ------------------------------------ | -------- | -------- | --------------------- |
| user_id     | "zzp"                                | 是       | String   | 用户 id               |
| kb_id       | "KB0015df77a8eb46f6951de513392dc250" | 是       | String   | 需要重命名的知识库 id |
| new_kb_name | "新知识库"                           | 是       | String   | 重命名后的知识库名字  |

### <h3><p id="重命名知识库请求示例">重命名知识库请求示例</p></h3>

```python
import requests
import json

url = "http://{your_host}:8777/api/local_doc_qa/rename_knowledge_base"
headers = {
    "Content-Type": "application/json"
}
data = {
	"user_id": "zzp",
	"kb_id": "KB0015df77a8eb46f6951de513392dc250",
	"new_kb_name": "新知识库"
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.status_code)
print(response.text)
```

### <h3><p id="重命名知识库响应示例">重命名知识库响应示例</p></h3>

```json
{
  "code": 200,
  "msg": "Knowledge Base 'KB0015df77a8eb46f6951de513392dc250' rename success"
}
```

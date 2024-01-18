
QAnything接口文档
==================

<details open="open">
<summary>目录</summary>

<!-- TOC -->

- [全局参数](#全局参数)
- [新建知识库（POST）](#新建知识库post)
    - [URL](#url)
    - [请求参数（Body）](#请求参数body)
    - [请求示例](#请求示例)
    - [响应示例](#响应示例)
- [上传文件（POST）](#上传文件post)
    - [URL](#url-1)
    - [请求参数（Body）](#请求参数body-1)
    - [请求示例](#请求示例-1)
    - [响应示例](#响应示例-1)
- [查看知识库（POST）](#查看知识库post)
    - [URL](#url-2)
    - [请求参数（Body)](#请求参数body)
    - [请求示例](#请求示例-2)
    - [响应示例](#响应示例-2)
- [获取文件列表（POST）](#获取文件列表post)
    - [URL](#url-3)
    - [请求参数（Body）](#请求参数body-2)
    - [请求示例](#请求示例-3)
    - [响应示例](#响应示例-3)
- [问答（POST）](#问答post)
    - [URL](#url-4)
    - [请求参数（Body）](#请求参数body-3)
    - [请求示例](#请求示例-4)
- [删除文件（POST）](#删除文件post)
    - [URL](#url-5)
    - [请求参数（Body）](#请求参数body-4)
    - [请求示例](#请求示例-5)
    - [响应示例](#响应示例-4)
- [删除知识库（POST）](#删除知识库post)
    - [URL](#url-6)
    - [请求参数（Body）](#请求参数body-5)
    - [请求示例](#请求示例-6)
    - [响应示例](#响应示例-5)

<!-- /TOC -->

</details>

## 全局参数
我们提供用户区分的功能，每个接口中有user_id的参数，如果需要请传入user_id的值。

user_id需要满足： 以字母开头，只允许包含字母，数字或下划线。

如果不需要用户区分，传入user_id="zzp"即可

### 注意前端默认只显示user_id="zzp"的知识库


## 新建知识库（POST）
### URL
<http://{your_host}:8777/api/local_doc_qa/new_knowledge_base>

### 请求参数（Body）
| 参数名              | 示例参数值                                | 是否必填 | 参数类型    | 描述说明                  |
| ---------------- | ---------------------------------- | ---- | ------- |-----------------------|
| user_id          | "zzp"                          | 是    | String  | 用户id （如需使用前端填zzp不要更换） |
| kb_name          | "kb_test"                         | 是    | String  | 知识库名称 （可以随意指定）        |


### 请求示例
```json
{
    "user_id": "zzp",  //用户id
    "kb_name": "kb_test"  //知识库名称
}
```

### 响应示例

```json
{
	"code": 200, //状态码
	"msg": "success", //提示信息
	"data": {
		"kb_id": "KB4c50de98d6b548af9aa0bc5e10b2e3a7", //知识库id
		"kb_name": "kb_test" //知识库名称
	}
}
```
## 上传文件（POST）

### URL
<http://{your_host}:8777/api/local_doc_qa/upload_files>


Content-Type: multipart/form-data

### 请求参数（Body）

| 参数名              | 示例参数值                                | 是否必填 | 参数类型    | 描述说明                                       |
| ---------------- | ---------------------------------- | ---- | ------- | ------------------------------------------ |
| files            | 文件二进制                           | 是    | \[文件类型] | 需要上传的文件，可多选（当use\_lcoal\_file为true时，选择无效）  |
| user\_id         | "zzp"                          | 是    | String  | 用户id                                       |
| kb\_id           | "KBb1dd58e8485443ce81166d24f6febda7" | 是    | String  | 知识库id                                      |
| mode             | "strong"                             | 是    | String  | 上传模式，soft：文件名重复的文件不再上传，strong：文件名重复的文件强制上传 |
| use\_local\_file | false                              | 是    | Boolean | 是否使用本地目录上传文件：「source/data」，可选：【true，false】 |

### 请求示例
同步请求示例：

```python
import os
import requests

url = "http://{your_host}:8777/api/local_doc_qa/upload_files"
folder_path = "./docx_data"
data = {
    "user_id": "zzp",
    "kb_id": "KB6dae785cdd5d47a997e890521acbe1c9"
}

files = []
for root, dirs, file_names in os.walk(folder_path):
    for file_name in file_names:
        if file_name.endswith(".md"):  # 只上传md文件
            file_path = os.path.join(root, file_name)
            files.append(("files", open(file_path, "rb")))

response = requests.post(url, files=files, data=data)
print(response.text)
```

异步请求示例：

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
import hashlib
import statistics

files = []
for root, dirs, file_names in os.walk("./docx_data"):
    for file_name in file_names:
        if file_name.endswith(".docx"):  # 只上传docx文件
            file_path = os.path.join(root, file_name)
            # if len(file_path) < 50:
            # print(file_path)
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
        # print(file_path)
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
              #print(await response.json())
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

### 响应示例

```json
{
	"code": 200,  //状态码
	"msg": "success，后台正在飞速上传文件，请耐心等待",  //提示信息
	"data": [
		{
			"file_id": "1b6c0781fb9245b2973504cb031cc2f3",  //文件id
			"file_name": "网易有道智云平台产品介绍2023.6.ppt",  //文件名
			"status": "gray"  //文件状态（red：入库失败-切分失败，green，成功入库，yellow：入库失败-milvus失败，gray：正在入库）
		},
		{
			"file_id": "aeaec708c7a34952b7de484fb3374f5d",
			"file_name": "有道知识库问答产品介绍.pptx",
			"status": "gray"
		},
		{
			"file_id": "8ee98a88457c414a986a09c536fedde9",
			"file_name": "韦小宝身份证.jpg",
			"status": "gray"
		},
		{
			"file_id": "67af479f907b497cadb30c6e4b2d3fbc",
			"file_name": "成长中心-辅导老师日常问题文档.pdf",
			"status": "gray"
		}
	]  //文件列表
}
```

## 查看知识库（POST）

### URL
<http://{your_host}:8777/api/local_doc_qa/list_knowledge_base>

### 请求参数（Body)
| 参数名              | 示例参数值                                | 是否必填 | 参数类型    | 描述说明                                       |
| ---------------- | ---------------------------------- | ---- | ------- | ------------------------------------------ |
| user\_id         | "zzp"                             | 是    | String  | 用户id                                       |

### 请求示例
```json
{
	"user_id": "zzp" //用户id
}
```

### 响应示例

```json
{
	"code": 200,  //状态码
	"msg": "success",  //提示信息
	"data": [
		{
			"kb_id": "KB973d4aea07f14c60ae1974404a636ad4",  //知识库id
			"kb_name": "kb_test"  //知识库名称
		}
	]   //知识库列表
}
```

## 获取文件列表（POST）

### URL
<http://{your_host}:8777/api/local_doc_qa/list_files>


### 请求参数（Body）
| 参数名              | 示例参数值                                | 是否必填 | 参数类型    | 描述说明                                       |
| ---------------- | ---------------------------------- | ---- | ------- | ------------------------------------------ |
| user_id         | "zzp"                              | 是    | String  | 用户id                                       |
| kb_id           | "KBb1dd58e8485443ce81166d24f6febda7"  | 是    | String  | 知识库id                                      |

### 请求示例

```json
{
	"user_id": "zzp", //用户id  注意需要满足 只含有字母 数字 和下划线且字母开头 的要求
	"kb_id": "KBb1dd58e8485443ce81166d24f6febda7" //知识库id
}
```

### 响应示例

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
		        "status": "green" //文件状态（red：入库失败-切分失败，green，成功入库，yellow：入库失败-milvus失败，gray：正在入库）
	            },
	            {
		        "file_id": "333e69374a8d4b9bac54f274291f313e", //文件id
		        "file_name": "网易有道智云平台产品介绍2023.6.ppt", //文件名
		        "status": "green" //文件状态（red：入库失败-切分失败，green，成功入库，yellow：入库失败-milvus失败，gray：正在入库）
                    },
	            {
		         "file_id": "709d6c3e071947038645f1f26ad99c6f", //文件id
		         "file_name": "韦小宝身份证.jpg", //文件名
                         "status": "green" //文件状态（red：入库失败-切分失败，green，成功入库，yellow：入库失败-milvus失败，gray：正在入库）
	            },
	            {
		        "file_id": "85297c0b56104028913e89b6834c1a39", //文件id
		        "file_name": "成长中心-辅导老师日常问题文档.pdf", //文件名
		        "status": "green" //文件状态（red：入库失败-切分失败，green，成功入库，yellow：入库失败-milvus失败，gray：正在入库）
	            },
	        }
	}
}		
```

## 问答（POST）

### URL
<http://{your_host}:8777/api/local_doc_qa/local_doc_chat>


### 请求参数（Body）
| 参数名              | 示例参数值                                | 是否必填 | 参数类型    | 描述说明                |
| ---------------- | ---------------------------------- | ---- | ------- | -----------------------------------|
| user_id         | "zzp"                              | 是    | String  | 用户id                              |
| kb_ids          | ["KBb1dd58e8485443ce81166d24f6febda7"]  | 是    | Array  | 知识库id的列表，支持多个知识库联合问答|
| question          | "保险单号是多少？" | 是    | String  | 知识库id的列表，支持多个知识库联合问答|
| history          | [["question1","answer1"],["question2","answer2"]]         | 是    | Array | 历史对话                     |

### 请求示例

```json
{
	"user_id": "zzp", //用户id
	"kb_ids": ["KBb1dd58e8485443ce81166d24f6febda7"], //知识库id，支持多个知识库联合问答
	"question": "保险单号是多少？", //用户问题
	"history": [] //历史对话：List[str]
}
```

响应示例

```json
{
	"code": 200, //状态码
	"msg": "success", //提示信息
	"question": "保险单号是多少？", //用户问题
	"response": "保险单号是601J389343982022000536",  //模型回答
	"related_questions": [],  //相关问题
	"history": [
		[
			"保险单号是多少？",
			"保险单号是601J389343982022000536。"
		]
	], //历史对话：List[List[str]]
	"source_documents": [
		{
			"file_id": "f9b794233c304dd5b5a010f2ead67f51", //文本内容对应的文件id
			"file_name": "一嗨案件支付三者车损、人伤保险赔款及权益转让授权书.docx", //文本内容对应的文件名
			"content": "未支付第三者车损、人伤赔款及同意直赔第三者确认书 华泰财产保险有限公司  北京   分公司： 本人租用一嗨在贵司承保车辆（车牌号：京KML920）商业险保单号： 601J389343982022000536、交强险保单号:  601J310028493882022000570， 对 2023 年 03 月 25日所发生的保险事故（事故号：  9010020230325004124）中所涉及的交强险和商业险的保险赔款总金额 (依：三者京AFT5538定损金额)， 同意支付给本次事故中第三者方。 在此本人确认：本人从未支付给第三者方任何赔偿，且承诺不就本次事故再向贵司及一嗨进行索赔。 同时本人保证如上述内容不属实、违反承诺，造成保险人损失的，由本人承担赔偿责任。 确认人（驾驶员）签字:              第三者方签字: 联系电话：                        联系电话： 确认日期：    年    月    日", //文本内容
			"retrieval_query": "保险单号是多少？", //文本内容对应的问题
			"score": "3.5585756", //相关性得分，分数越高越相关
			"embed_version": "local_v0.0.1_20230525_6d4019f1559aef84abc2ab8257e1ad4c" //embedding模型版本号
		}
	], //知识库相关文本内容
	"rematched_source_documents": [] //重新匹配的文本内容
}
```


## 删除文件（POST）

### URL
<http://{your_host}:8777/api/local_doc_qa/delete_files>

### 请求参数（Body）
| 参数名              | 示例参数值                                | 是否必填 | 参数类型    | 描述说明                |
| ---------------- | ----------------------------------  | ---- | ------- | -----------------------------------|
| user_id          | "zzp"                             | 是    | String  | 用户id                              |
| kb_id            | "KB1271e71c36ec4028a6542586946a3906"  | 是    | String  | 知识库id|
| file_ids         | ["73ff7cf76ff34c8aa3a5a0b4ba3cf534"] | 是    | Array  | 要删除文件的id，支持批量删除|

### 请求示例
```json
{
	"user_id": "zzp", //用户id
	"kb_id": "KB1271e71c36ec4028a6542586946a3906", //知识库id
	"file_ids": [
		"73ff7cf76ff34c8aa3a5a0b4ba3cf534"
	] //文件id列表
}
```

### 响应示例

```json
{
	"code": 200, //状态码
	"msg": "documents ['73ff7cf76ff34c8aa3a5a0b4ba3cf534'] delete success" //提示信息
}
```

## 删除知识库（POST）

### URL
<http://{your_host}:8777/api/local_doc_qa/delete_knowledge_base>

### 请求参数（Body）

| 参数名              | 示例参数值                                | 是否必填 | 参数类型    | 描述说明                |
| ---------------- | ----------------------------------  | ---- | ------- | -----------------------------------|
| user_id          | "zzp"                             | 是    | String  | 用户id                              |
| kb_ids            | ["KB1cd81f2bc515437294bda1934a20b235"]  | 是    | Array  | 要删除的知识库id，支持批量删除|

### 请求示例
```json
{
	"user_id": "zzp", //用户id
	"kb_ids": [
		"KB1cd81f2bc515437294bda1934a20b235"
	] //知识库id列表
}
```

### 响应示例

```json
{
	"code": 200, //状态码
	"msg": "Knowledge Base [('KB1cd81f2bc515437294bda1934a20b235',)] delete success" //提示信息
}
```

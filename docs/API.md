
# QAnything接口文档

## 全局参数
注意所有请求参数中的user\_id需要满足： 只含有字母 数字 和下划线且字母开头 的要求


## 新建知识库

URL：<http://0.0.0.0:8777/api/local_doc_qa/new_knowledge_base>

请求参数

```json
{
    "user_id": "liujx",  //用户id
    "kb_name": "dataset_s_4"  //知识库名称
}
```

响应示例

```json
{
	"code": 200, //状态码
	"msg": "success", //提示信息
	"data": {
		"kb_id": "KB4c50de98d6b548af9aa0bc5e10b2e3a7", //知识库id
		"kb_name": "dataset_s_3" //知识库名称
	} //文件列表
}
```
## 上传文件

URL：<http://0.0.0.0:8777/api/local_doc_qa/upload_files>

Content-Type: multipart/form-data

请求参数：

| 参数名              | 参数值                                | 是否必填 | 参数类型    | 描述说明                                       |
| ---------------- | ---------------------------------- | ---- | ------- | ------------------------------------------ |
| files            | \[文件路径]                            | 是    | \[文件类型] | 需要上传的文件，可多选（当use\_lcoal\_file为true时，选择无效）  |
| user\_id         | liujx                              | 是    | String  | 用户id                                       |
| kb\_id           | KBb1dd58e8485443ce81166d24f6febda7 | 是    | String  | 知识库id                                      |
| mode             | strong                             | 是    | String  | 上传模式，soft：文件名重复的文件不再上传，strong：文件名重复的文件强制上传 |
| use\_local\_file | false                              | 是    | Boolean | 是否使用本地目录上传文件：「source/data」，可选：【true，false】 |

同步请求示例：

```python
import os
import requests

url = "http://0.0.0.0:8777/api/local_doc_qa/upload_files"
folder_path = "./docx_data"
data = {
    "user_id": "liujx",
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
    url = 'http://0.0.0.0:8777/api/local_doc_qa/upload_files'
    data = aiohttp.FormData()
    data.add_field('user_id', 'liujx')
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

响应示例：

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
			"file_id": "df4106df7c674cb495b172ece3ea3f6e",
			"file_name": "智能文档问答提升生产力.pptx",
			"status": "gray"
		},
		{
			"file_id": "67af479f907b497cadb30c6e4b2d3fbc",
			"file_name": "成长中心-辅导老师日常问题文档.pdf",
			"status": "gray"
		},
		{
			"file_id": "81269c324d9b46bdb7c01d11a5b51395",
			"file_name": "有道领世辅导日常问题处理手册.pdf",
			"status": "gray"
		},
		{
			"file_id": "829cca2a8cb445d9a7317a3979ec3523",
			"file_name": "一嗨案件支付三者车损、人伤保险赔款及权益转让授权书.docx",
			"status": "gray"
		}
	]  //文件列表
}
```
## 查看知识库

URL：<http://0.0.0.0:8777/api/local_doc_qa/list_knowledge_base>

请求参数

```json
{
	"user_id": "liujx" //用户id
}
```

响应示例

```json
{
	"code": 200,  //状态码
	"msg": "success",  //提示信息
	"data": [
		{
			"kb_id": "KB973d4aea07f14c60ae1974404a636ad4",  //知识库id
			"kb_name": "dataset_s_1"  //知识库名称
		}
	]   //知识库列表
}
```

## 获取文件列表

URL: <http://0.0.0.0:8777/api/local_doc_qa/list_files>

请求参数

```json
{
	"user_id": "liujx", //用户id  注意需要满足 只含有字母 数字 和下划线且字母开头 的要求
	"kb_id": "KBb1dd58e8485443ce81166d24f6febda7" //知识库id
}
```

响应示例

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
		"file_id": "362a03f82ac44ed7afca9d873d03b901", //文件id
		"file_name": "智能文档问答提升生产力.pptx", //文件名
		"status": "green" //文件状态（red：入库失败-切分失败，green，成功入库，yellow：入库失败-milvus失败，gray：正在入库）
	},
	{
		"file_id": "4867d3fcf395414083c896993fae291e", //文件id
		"file_name": "有道领世辅导日常问题处理手册.pdf", //文件名
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
	{
		"file_id": "f9b794233c304dd5b5a010f2ead67f51", //文件id
		"file_name": "一嗨案件支付三者车损、人伤保险赔款及权益转让授权书.docx", //文件名
		"status": "green" //文件状态（red：入库失败-切分失败，green，成功入库，yellow：入库失败-milvus失败，gray：正在入库）
	}
	}
	}
}		
```

## 对话

URL：<http://0.0.0.0:8777/api/local_doc_qa/local_doc_chat>

请求参数

```json
{
	"user_id": "liujx", //用户id
	"kb_ids": ["KBb1dd58e8485443ce81166d24f6febda7"], //知识库id，支持多个知识库联合问答
	"question": "一嗨案件中保险单号是多少？", //用户问题
	"history": [] //历史对话：List[str]
}
```

响应示例

```json
{
	"code": 200, //状态码
	"msg": "success", //提示信息
	"question": "一嗨案件中保险单号是多少？", //用户问题
	"response": "保险单号是601J312512022000536。",  //模型回答
	"related_questions": [],  //相关问题
	"history": [
		[
			"一嗨案件中保险单号是多少？",
			"保险单号是601J312512022000536。"
		]
	], //历史对话：List[str]
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
	], //知识库相关文本内容
	"rematched_source_documents": [] //重新匹配的文本内容
}
```


## 删除文件

URL：<http://0.0.0.0:8777/api/local_doc_qa/delete_files>

请求参数：

```json
{
	"user_id": "liujx", //用户id
	"kb_id": "KB1271e71c36ec4028a6542586946a3906", //知识库id
	"file_ids": [
		"73ff7cf76ff34c8aa3a5a0b4ba3cf534"
	] //文件id列表
}
```

响应示例：

```json
{
	"code": 200, //状态码
	"msg": "documents ['73ff7cf76ff34c8aa3a5a0b4ba3cf534'] delete success" //提示信息
}
```

## 删除知识库

URL：<http://0.0.0.0:8777/api/local_doc_qa/delete_knowledge_base>

请求参数：

```json
{
	"user_id": "liujx", //用户id
	"kb_ids": [
		"KB1cd81f2bc515437294bda1934a20b235"
	] //知识库id列表
}
```

响应示例：

```json
{
	"code": 200, //状态码
	"msg": "Knowledge Base [('KB1cd81f2bc515437294bda1934a20b235',)] delete success" //提示信息
}
```

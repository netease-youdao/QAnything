# QAnything RAG服务api说明


## 接口
接口列表
| 接口地址 | 请求类型 | 说明 |
| ---- | ---- | ---- |
| [/api/qanything/delete_knowledge_base](#删除知识库) | POST | 删除知识库和文件 |
| [/api/qanything/document_parser](#解析文件) | POST | 解析文件 |
| [/api/qanything/document_parser_embedding](#解析文件并存储) | POST | 解析文件并保存 |
| [/api/qanything/chunk_embedding](#上传切片后的文本数据) | POST | 上传切片后的文本数据 |
| [/api/qanything/question_rag_search](#问答检索) | POST | 问答接口 |
| [/api/qanything/get_files_statu](#获取指定文件状态) | POST | 获取指定文件状态 |
| [/api/qanything/upload_faqs](#上传faq) | POST | 上传FAQ(docker版本暂时不可用) |




### 删除知识库

* 请求参数和示例

| 参数名 | 类型 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| user_id | string | 是 | 用户id |
| kb_id   | string | 是 | 需要删除的知识库id |
| file_ids| []string | 否 | 需要删除的知识库中文件id，不输入则删除该知识库下所有文件 |

```
    curl -X POST -H "Content-Type: application/json" http://ip:port/api/qanything/delete_knowledge_base \
    -d '{"user_id": "123456", "kb_id": "KB123456789"}'
```

返回结果

| 参数名 | 类型 | 说明 |
| ----- | --- | --- |
| code | int    | 200表示成功，其他表示失败 |
| msg  | string | 返回结果说明 |


```
{
    "code":200,
    "msg":"Knowledge Base KB6dae785cdd5d47a997e890521acbe1c5 delete success"}
```

code其他情况说明
| code参数名值 | 说明 |
| ---- | --- |
| 200  | 成功 |
| 2002 | 未输入user_id |
| 2005 | 输入user_id不可用 |
| 2003 | kb_id未找到 |





### 获取指定文件状态

* 请求参数和示例

| 参数名 | 类型 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| user_id | string | 是 | 用户id |
| kb_id   | string | 是 | 需要获取知识库文件列表的知识库id |
| file_ids   | []string | 是 | 文件列表的文件id |

```
    curl -X POST -H "Content-Type: application/json" http://ip:port/api/qanything/list_files \
    -d '{"user_id": "123456", "kb_id": "KB123456789", file_ids: ["123", "124"]}'
```

* 返回结果

| 参数名 | 类型 | 说明 |
| ----- | --- | --- |
| code | int    | 200表示成功，其他表示失败 |
| msg  | string | 返回结果说明 |
| data | List | 知识库文件列表 |


```
{
    "code":200,
    "msg":"success",
    "data":{
        "total":{"green":2},
        "details":[
            {
                "file_id":"123",
                "file_name":"\u6d4b\u8bd5langchain.docx",
                "status":"green",
                "bytes":4174317,
                "content_length":1556,
                "timestamp":"202406121611",
                "msg":"\u4e0a\u4f20\u6210\u529f"
            },{
                "file_id":"124",
                "file_name":"repiBench.pdf",
                "status":"green",
                "bytes":2614844,
                "content_length":72019,
                "timestamp":"202406121655",
                "msg":"\u4e0a\u4f20\u6210\u529f"
            }
        ]
    }
}
```



### 解析文件

* 请求参数和示例

| 参数名 | 类型 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| user_id | string | 是 | 用户id |
| files | 文件 | 是 | 待解析文档，只能上传一个，[('file', open('/home/darren/文档/repiBench.pdf','rb'))] |

```
curl -X POST "http://<your_host>:<your_port>/api/qanything/document_parser"  -d '{"user_id": "zzp"}' -F "file=@/home/darren/文档/repiBench.pdf"
```

* 返回结果

| 参数名 | 类型 | 说明 |
| ----- | --- | --- |
| code | int    | 200表示成功，其他表示失败 |
| msg  | string | 返回结果说明 |
| parser_documents | List | 知识库文件列表 |


```
{
    "code":200,
    "msg":"document parser success",
    "parser_documents":[
        {
            "metadata":{
                "user_id":"zzp",
                "kb_id":"kb_id",
                "file_id":"file_id",
                "file_name":"repiBench.pdf",
                "chunk_id":0,
                "file_path":"/home/darren/code/QAnything/QANY_DB/content/zzp/file_id/repiBench.pdf",
                "faq_dict":{}
            },
            "page_content":"RepoBench: Benchmarking Repository-Level Code\nAuto-Completion ........ McAuley\nUniversity of California."
        }
    ]
}
```






### 解析文件并存储

* 请求参数和示例

| 参数名 | 类型 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| user_id | string | 是 | 用户id |
| kb_id   | string | 是 | 知识库id |
| file_ids| string | 是 | 文件id列表，与files中文件顺序相同,多个文件id中间用逗号隔开 |
| mode    | string | 否 | "soft"代表不上传同名文件，"strong"表示强制上传同名文件，默认"soft" |
| files | 文件 | 是 | 待解析文档，支持上传多个，[('files', open('/home/darren/文档/repiBench.pdf','rb')),] |

```
curl -X POST "http://<your_host>:<your_port>/api/qanything/document_parser_embedding"  -d '{"user_id": "zzp", "kb_id":"kb_id", "file_ids":"file_id1,file_id2", "mode":"soft"}' -F "files=@/home/darren/文档/repiBench.pdf" -F "files=@/home/darren/文档/repiBench2.pdf"
```

* 返回结果

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| code | int | 200表示成功，其他表示失败 |
| msg | string | 返回结果说明 |
| data | List | 每个文件的保存详细情况，其中，status表示文件的处理状态，gray表示后台处理中，还未处理完成（可能出现处理失败的情况，需要后台自行查询） |


```
{
    "code":200,
    "msg":"success\uff0c\u540e\u53f0\u6b63\u5728\u98de\u901f\u4e0a\u4f20\u6587\u4ef6\uff0c\u8bf7\u8010\u5fc3\u7b49\u5f85",
    "data":[
        {
            "file_id":"124",
            "file_name":"repiBench.pdf",
            "status":"gray",
            "bytes":2614844,
            "timestamp":"202406121655"
        }
    ]
}
```




### 上传切片后的文本数据
* 请求参数和示例

| 参数名 | 类型 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| user_id | string | 是 | 用户id |
| kb_id   | string | 是 | 知识库id |
| file_id | string | 否 | 文件id，默认uuid.uuid4().hex |
| file_name | string | 否 | 文件名称，默认为file_id+".txt" |
| chunk_datas | []string | 是 | 切片后的文本数据 |


```
curl -X POST "http://<your_host>:<your_port>/api/qanything/document_parser_embedding"  -d '{"user_id": "zzp", "kb_id":"kb_id", "file_id":"file_id", "file_name":"xxx.txt", "chunk_datas": ["切片文本1","切片文本2"] }'
```

* 返回结果

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| code | int | 200表示成功，其他表示失败 |
| msg | string | 返回结果说明 |
| data | dict | 文本的保存详细情况，其中，status表示文件的处理状态，gray表示后台处理中，还未处理完成（可能出现处理失败的情况，需要后台自行查询） |


```
{
    "code":200,
    "msg":"success\uff0c\u540e\u53f0\u6b63\u5728\u98de\u901f\u4e0a\u4f20\u6587\u4ef6\uff0c\u8bf7\u8010\u5fc3\u7b49\u5f85",
    "data":{
            "file_id":"124",
            "file_name":"repiBench.pdf",
            "status":"gray",
            "bytes":2614844,
            "timestamp":"202406121655"
    }
}
```



### 问答检索

* 请求参数和示例

| 参数名 | 类型 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| user_id | string | 是 | 用户id |
| kb_ids  | []string | 是 | 检索的知识库id列表 |
| question| string | 是 | 待检索问题 |

```
    curl -X POST -H "Content-Type: application/json" http://ip:port/api/qanything/question_rag_search \
    -d '{"user_id": "zzp", "kb_id": "KB6dae785cdd5d47a997e890521acbe1c5", "question": "python document"}'
```

返回结果

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| code | int | 200表示成功，其他表示失败 |
| msg | string | 返回结果说明 |
| question | string | 检索问题 |
| retrieval_documents | List | 检索结果 |


```
{
    "code": 0,
    "msg": "success",
    "question": "xxxx",
    "retrieval_documents": [
        {
            "metadata": {
                "user_id":"zzp",
                "kb_id":"KB6dae785cdd5d47a997e890521acbe1c5",
                "file_id":"123",
                "file_name":"\u6d4b\u8bd5langchain.docx",
                "chunk_id":5,
                "file_path":"/home/darren/code/QAnything/QANY_DB/content/zzp/123/\u6d4b\u8bd5langchain.docx",
                "faq_dict":{},
                "score":0.480712890625,
                "retrieval_query":"python document",
                "embed_version":"local_v0.0.1_20230525_6d4019f1559aef84abc2ab8257e1ad4c"
            },
            "page_content": "sadfasdf",
        },
        {
            "metadata": {

            },
            "page_content": "xxxx2",
        }
    ]
}
```
    




### 上传FAQ

* 请求参数和示例

| 参数名 | 类型 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| user_id | string | 是 | 用户id |
| kb_id   | string | 是 | 知识库id |
| faqs | list | 是 | 格式：[{"question": "xxxx", "answer": "xxx"}, ...,{"question": "xxx", "answer": "xxx"}],单次最大支持1000条上传 |
| --question | string | 是 | 问题 |
| --answer | string | 是 | 回答 |


*  返回结果

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| code | int | 200表示成功，其他表示失败 |
| msg | string | 返回结果说明 |
| data | List | 每个上传QA对的状态 |


```
{
    "code":200,
    "msg":"success\uff0c\u540e\u53f0\u6b63\u5728\u98de\u901f\u4e0a\u4f20\u6587\u4ef6\uff0c\u8bf7\u8010\u5fc3\u7b49\u5f85",
    "file_status":{},
    "data":[
        {
            "file_id":"319e2082f5e345eb825e4387003321bf",
            "file_name":"FAQ_\u5982\u4f55\u4f7f\u7528python.faq",
            "status":"gray",
            "length":94,
            "timestamp":"202406121756"
        },{
            "file_id":"66a71bc57eee49b2a4b61b8448bc4f18",
            "file_name":"FAQ_\u5982\u4f55\u4f7f\u7528docker.faq",
            "status":"gray",
            "length":85,
            "timestamp":"202406121756"
        }
    ]
}

```


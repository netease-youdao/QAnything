from qanything_kernel.core.local_file import LocalFile
from qanything_kernel.core.local_doc_qa import LocalDocQA
from qanything_kernel.utils.general_utils import *
from qanything_kernel.utils.custom_log import debug_logger, qa_logger
from sanic.response import ResponseStream
from sanic.response import json as sanic_json
from sanic.response import text as sanic_text
from sanic import request
import uuid
import json
import asyncio
import urllib.parse
import re
from datetime import datetime
import os

__all__ = ["new_knowledge_base", "upload_files", "list_kbs", "list_docs", "delete_knowledge_base", "delete_docs",
           "rename_knowledge_base", "get_total_status", "clean_files_by_status", "upload_weblink", "local_doc_chat",
           "document"]

INVALID_USER_ID = f"fail, Invalid user_id: . user_id 必须只含有字母，数字和下划线且字母开头"


async def new_knowledge_base(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("new_knowledge_base %s", user_id)
    kb_name = safe_get(req, 'kb_name')
    kb_id = 'KB' + uuid.uuid4().hex
    local_doc_qa.create_milvus_collection(user_id, kb_id, kb_name)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")
    return sanic_json({"code": 200, "msg": "success create knowledge base {}".format(kb_id),
                       "data": {"kb_id": kb_id, "kb_name": kb_name, "timestamp": timestamp}})


async def upload_weblink(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("upload_weblink %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    url = safe_get(req, 'url')
    mode = safe_get(req, 'mode', default='soft')  # soft代表不上传同名文件，strong表示强制上传同名文件
    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
        return sanic_json({"code": 2001, "msg": msg, "data": [{}]})
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")
    exist_files = []
    if mode == 'soft':
        exist_files = local_doc_qa.milvus_summary.check_file_exist_by_name(user_id, kb_id, [url])
    if exist_files:
        file_id, file_name, file_size, status = exist_files[0]
        msg = f'warning，当前的mode是soft，无法上传同名文件，如果想强制上传同名文件，请设置mode：strong'
        data = [{"file_id": file_id, "file_name": url, "status": status, "bytes": file_size, "timestamp": timestamp}]
    else:
        file_id, msg = local_doc_qa.milvus_summary.add_file(user_id, kb_id, url, timestamp)
        local_file = LocalFile(user_id, kb_id, url, file_id, url, local_doc_qa.embeddings, is_url=True)
        data = [{"file_id": file_id, "file_name": url, "status": "gray", "bytes": 0, "timestamp": timestamp}]
        asyncio.create_task(local_doc_qa.insert_files_to_milvus(user_id, kb_id, [local_file]))
        msg = "success，后台正在飞速上传文件，请耐心等待"
    return sanic_json({"code": 200, "msg": msg, "data": data})


async def upload_files(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.form: {req.form}，request.files: {req.files}请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("upload_files %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    mode = safe_get(req, 'mode', default='soft')  # soft代表不上传同名文件，strong表示强制上传同名文件
    debug_logger.info("mode: %s", mode)
    use_local_file = safe_get(req, 'use_local_file', 'false')
    if use_local_file == 'true':
        files = read_files_with_extensions()
    else:
        files = req.files.getlist('files')

    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
        return sanic_json({"code": 2001, "msg": msg, "data": [{}]})

    data = []
    local_files = []
    file_names = []
    for file in files:
        if isinstance(file, str):
            file_name = os.path.basename(file)
        else:
            debug_logger.info('ori name: %s', file.name)
            file_name = urllib.parse.unquote(file.name, encoding='UTF-8')
            debug_logger.info('decode name: %s', file_name)
        # 删除掉全角字符
        file_name = re.sub(r'[\uFF01-\uFF5E\u3000-\u303F]', '', file_name)
        file_name = file_name.replace("/", "_")
        debug_logger.info('cleaned name: %s', file_name)
        file_name = truncate_filename(file_name)
        file_names.append(file_name)

    exist_file_names = []
    if mode == 'soft':
        exist_files = local_doc_qa.milvus_summary.check_file_exist_by_name(user_id, kb_id, file_names)
        exist_file_names = [f[1] for f in exist_files]

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")

    for file, file_name in zip(files, file_names):
        if file_name in exist_file_names:
            continue
        file_id, msg = local_doc_qa.milvus_summary.add_file(user_id, kb_id, file_name, timestamp)
        debug_logger.info(f"{file_name}, {file_id}, {msg}")
        local_file = LocalFile(user_id, kb_id, file, file_id, file_name, local_doc_qa.embeddings)
        local_files.append(local_file)
        local_doc_qa.milvus_summary.update_file_size(file_id, len(local_file.file_content))
        data.append(
            {"file_id": file_id, "file_name": file_name, "status": "gray", "bytes": len(local_file.file_content),
             "timestamp": timestamp})

    asyncio.create_task(local_doc_qa.insert_files_to_milvus(user_id, kb_id, local_files))
    if exist_file_names:
        msg = f'warning，当前的mode是soft，无法上传同名文件{exist_file_names}，如果想强制上传同名文件，请设置mode：strong'
    else:
        msg = "success，后台正在飞速上传文件，请耐心等待"
    return sanic_json({"code": 200, "msg": msg, "data": data})


async def list_kbs(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("list_kbs %s", user_id)
    kb_infos = local_doc_qa.milvus_summary.get_knowledge_bases(user_id)
    data = []
    for kb in kb_infos:
        data.append({"kb_id": kb[0], "kb_name": kb[1]})
    debug_logger.info("all kb infos: {}".format(data))
    return sanic_json({"code": 200, "data": data})


async def list_docs(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("list_docs %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    debug_logger.info("kb_id: {}".format(kb_id))
    data = []
    file_infos = local_doc_qa.milvus_summary.get_files(user_id, kb_id)
    status_count = {}
    msg_map = {'gray': "正在上传中，请耐心等待",
               'red': "split或embedding失败，请检查文件类型，仅支持[md,txt,pdf,jpg,png,jpeg,docx,xlsx,pptx,eml,csv]",
               'yellow': "milvus插入失败，请稍后再试", 'green': "上传成功"}
    for file_info in file_infos:
        status = file_info[2]
        if status not in status_count:
            status_count[status] = 1
        else:
            status_count[status] += 1
        data.append({"file_id": file_info[0], "file_name": file_info[1], "status": file_info[2], "bytes": file_info[3],
                     "content_length": file_info[4], "timestamp": file_info[5], "msg": msg_map[file_info[2]]})

    return sanic_json({"code": 200, "msg": "success", "data": {'total': status_count, 'details': data}})


async def delete_knowledge_base(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("delete_knowledge_base %s", user_id)
    kb_ids = safe_get(req, 'kb_ids')
    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, kb_ids)
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids)})

    milvus = local_doc_qa.match_milvus_kb(user_id, kb_ids)
    for kb_id in kb_ids:
        milvus.delete_partition(kb_id)
    local_doc_qa.milvus_summary.delete_knowledge_base(user_id, kb_ids)
    return sanic_json({"code": 200, "msg": "Knowledge Base {} delete success".format(kb_ids)})


async def rename_knowledge_base(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("rename_knowledge_base %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    new_kb_name = safe_get(req, 'new_kb_name')
    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids[0])})
    local_doc_qa.milvus_summary.rename_knowledge_base(user_id, kb_id, new_kb_name)
    return sanic_json({"code": 200, "msg": "Knowledge Base {} rename success".format(kb_id)})


async def delete_docs(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("delete_docs %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    file_ids = safe_get(req, "file_ids")
    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids[0])})
    valid_file_infos = local_doc_qa.milvus_summary.check_file_exist(user_id, kb_id, file_ids)
    if len(valid_file_infos) == 0:
        return sanic_json({"code": 2004, "msg": "fail, files {} not found".format(file_ids)})
    milvus_kb = local_doc_qa.match_milvus_kb(user_id, [kb_id])
    milvus_kb.delete_files(file_ids)
    # 删除数据库中的记录
    local_doc_qa.milvus_summary.delete_files(kb_id, file_ids)
    return sanic_json({"code": 200, "msg": "documents {} delete success".format(file_ids)})


async def get_total_status(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info('get_total_status %s', user_id)
    if not user_id:
        users = local_doc_qa.milvus_summary.get_users()
        users = [user[0] for user in users]
    else:
        users = [user_id]
    res = {}
    for user in users:
        res[user] = {}
        kbs = local_doc_qa.milvus_summary.get_knowledge_bases(user)
        for kb_id, kb_name in kbs:
            gray_file_infos = local_doc_qa.milvus_summary.get_file_by_status([kb_id], 'gray')
            red_file_infos = local_doc_qa.milvus_summary.get_file_by_status([kb_id], 'red')
            yellow_file_infos = local_doc_qa.milvus_summary.get_file_by_status([kb_id], 'yellow')
            green_file_infos = local_doc_qa.milvus_summary.get_file_by_status([kb_id], 'green')
            res[user][kb_name + kb_id] = {'green': len(green_file_infos), 'yellow': len(yellow_file_infos),
                                          'red': len(red_file_infos),
                                          'gray': len(gray_file_infos)}

    return sanic_json({"code": 200, "status": res})


async def clean_files_by_status(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info('clean_files_by_status %s', user_id)
    status = safe_get(req, 'status', default='gray')
    kb_ids = safe_get(req, 'kb_ids')
    if not kb_ids:
        kbs = local_doc_qa.milvus_summary.get_knowledge_bases(user_id)
        kb_ids = [kb[0] for kb in kbs]
    else:
        not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, kb_ids)
        if not_exist_kb_ids:
            return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids)})

    gray_file_infos = local_doc_qa.milvus_summary.get_file_by_status(kb_ids, status)
    gray_file_ids = [f[0] for f in gray_file_infos]
    gray_file_names = [f[1] for f in gray_file_infos]
    debug_logger.info(f'{status} files number: {len(gray_file_names)}')
    # 删除milvus中的file
    if gray_file_ids:
        milvus_kb = local_doc_qa.match_milvus_kb(user_id, kb_ids)
        milvus_kb.delete_files(gray_file_ids)
        for kb_id in kb_ids:
            local_doc_qa.milvus_summary.delete_files(kb_id, gray_file_ids)
    return sanic_json({"code": 200, "msg": f"delete {status} files success", "data": gray_file_names})


async def local_doc_chat(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info('local_doc_chat %s', user_id)
    kb_ids = safe_get(req, 'kb_ids')
    question = safe_get(req, 'question')
    rerank = safe_get(req, 'rerank', default=True)
    debug_logger.info('rerank %s', rerank)
    streaming = safe_get(req, 'streaming', False)
    history = safe_get(req, 'history', [])
    debug_logger.info("history: %s ", history)
    debug_logger.info("question: %s", question)
    debug_logger.info("kb_ids: %s", kb_ids)
    debug_logger.info("user_id: %s", user_id)

    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, kb_ids)
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids)})

    file_infos = []
    milvus_kb = local_doc_qa.match_milvus_kb(user_id, kb_ids)
    for kb_id in kb_ids:
        file_infos.extend(local_doc_qa.milvus_summary.get_files(user_id, kb_id))
    valid_files = [fi for fi in file_infos if fi[2] == 'green']
    if len(valid_files) == 0:
        return sanic_json({"code": 200, "msg": "当前知识库为空，请上传文件或等待文件解析完毕", "question": question,
                           "response": "All knowledge bases {} are empty or haven't green file, please upload files".format(
                               kb_ids), "history": history, "source_documents": [{}]})
    else:
        debug_logger.info("streaming: %s", streaming)
        if streaming:
            debug_logger.info("start generate answer")

            async def generate_answer(response):
                debug_logger.info("start generate...")
                for resp, next_history in local_doc_qa.get_knowledge_based_answer(
                        query=question, milvus_kb=milvus_kb, chat_history=history, streaming=True, rerank=rerank
                ):
                    chunk_data = resp["result"]
                    if not chunk_data:
                        continue
                    chunk_str = chunk_data[6:]
                    if chunk_str.startswith("[DONE]"):
                        source_documents = []
                        for inum, doc in enumerate(resp["source_documents"]):
                            source_info = {'file_id': doc.metadata['file_id'],
                                           'file_name': doc.metadata['file_name'],
                                           'content': doc.page_content,
                                           'retrieval_query': doc.metadata['retrieval_query'],
                                           'score': str(doc.metadata['score'])}
                            source_documents.append(source_info)

                        retrieval_documents = format_source_documents(resp["retrieval_documents"])
                        source_documents = format_source_documents(resp["source_documents"])
                        chat_data = {'user_info': user_id, 'kb_ids': kb_ids, 'query': question, 'history': history,
                                     'prompt': resp['prompt'], 'result': next_history[-1][1],
                                     'retrieval_documents': retrieval_documents, 'source_documents': source_documents}
                        qa_logger.info("chat_data: %s", chat_data)
                        debug_logger.info("response: %s", chat_data['result'])
                        stream_res = {
                            "code": 200,
                            "msg": "success",
                            "question": question,
                            # "response":next_history[-1][1],
                            "response": "",
                            "history": next_history,
                            "source_documents": source_documents,
                        }
                    else:
                        chunk_js = json.loads(chunk_str)
                        delta_answer = chunk_js["answer"]
                        stream_res = {
                            "code": 200,
                            "msg": "success",
                            "question": "",
                            "response": delta_answer,
                            "history": [],
                            "source_documents": [],
                        }
                    await response.write(f"data: {json.dumps(stream_res, ensure_ascii=False)}\n\n")
                    if chunk_str.startswith("[DONE]"):
                        await response.eof()
                    await asyncio.sleep(0.001)

            response_stream = ResponseStream(generate_answer, content_type='text/event-stream')
            return response_stream

        else:
            for resp, history in local_doc_qa.get_knowledge_based_answer(
                    query=question, milvus_kb=milvus_kb, chat_history=history, streaming=False, rerank=rerank
            ):
                pass
            retrieval_documents = format_source_documents(resp["retrieval_documents"])
            source_documents = format_source_documents(resp["source_documents"])
            chat_data = {'user_id': user_id, 'kb_ids': kb_ids, 'query': question, 'history': history,
                         'retrieval_documents': retrieval_documents, 'prompt': resp['prompt'], 'result': resp['result'],
                         'source_documents': source_documents}
            qa_logger.info("chat_data: %s", chat_data)
            debug_logger.info("response: %s", chat_data['result'])
            return sanic_json({"code": 200, "msg": "success chat", "question": question, "response": resp["result"],
                               "history": history, "source_documents": source_documents})


async def document(req: request):
    description = """
# QAnything 介绍
[戳我看视频>>>>>【有道QAnything介绍视频.mp4】](https://docs.popo.netease.com/docs/7e512e48fcb645adadddcf3107c97e7c)

**QAnything** (**Q**uestion and **A**nswer based on **Anything**) 是支持任意格式的本地知识库问答系统。

您的任何格式的本地文件都可以往里扔，即可获得准确、快速、靠谱的问答体验。

**目前已支持格式:**
* PDF
* Word(doc/docx)
* PPT
* TXT
* 图片
* 网页链接
* ...更多格式，敬请期待

# API 调用指南

## API Base URL

https://qanything.youdao.com

## 鉴权
目前使用微信鉴权,步骤如下:
1. 客户端通过扫码微信二维码(首次登录需要关注公众号)
2. 获取token
3. 调用下面所有API都需要通过authorization参数传入这个token

注意：authorization参数使用Bearer auth认证方式

生成微信二维码以及获取token的示例代码下载地址：[微信鉴权示例代码](https://docs.popo.netease.com/docs/66652d1a967e4f779594aef3306f6097)

## API 接口说明
    {
        "api": "/api/local_doc_qa/upload_files"
        "name": "上传文件",
        "description": "上传文件接口，支持多个文件同时上传，需要指定知识库名称",
    },
    {
        "api": "/api/local_doc_qa/upload_weblink"
        "name": "上传网页链接",
        "description": "上传网页链接，自动爬取网页内容，需要指定知识库名称",
    },
    {
        "api": "/api/local_doc_qa/local_doc_chat" 
        "name": "问答接口",
        "description": "知识库问答接口，指定知识库名称，上传用户问题，通过传入history支持多轮对话",
    },
    {
        "api": "/api/local_doc_qa/list_files" 
        "name": "文件列表",
        "description": "列出指定知识库下的所有文件名，需要指定知识库名称",
    },
    {
        "api": "/api/local_doc_qa/delete_files" 
        "name": "删除文件",
        "description": "删除指定知识库下的指定文件，需要指定知识库名称",
    },

"""
    return sanic_text(description)

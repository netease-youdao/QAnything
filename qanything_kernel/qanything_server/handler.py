from qanything_kernel.core.local_file import LocalFile
from qanything_kernel.core.local_doc_qa import LocalDocQA
from qanything_kernel.utils.general_utils import *
from qanything_kernel.utils.custom_log import debug_logger, qa_logger
from qanything_kernel.configs.model_config import BOT_DESC, BOT_IMAGE, BOT_PROMPT, BOT_WELCOME
from sanic.response import ResponseStream
from sanic.response import json as sanic_json
from sanic.response import text as sanic_text
from sanic import request, response
import math
import uuid
import json
import asyncio
import urllib.parse
import re
from datetime import datetime
from tqdm import tqdm
import os
import base64

__all__ = ["new_knowledge_base", "upload_files", "list_kbs", "list_docs", "delete_knowledge_base", "delete_docs",
           "rename_knowledge_base", "get_total_status", "clean_files_by_status", "upload_weblink", "local_doc_chat",
           "document", "new_bot", "delete_bot", "update_bot", "get_bot_info", "upload_faqs", "get_file_base64",
           "get_qa_info"]

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
    default_kb_id = 'KB' + uuid.uuid4().hex
    kb_id = safe_get(req, 'kb_id', default_kb_id)
    if kb_id[:2] != 'KB':
        return sanic_json({"code": 2001, "msg": "fail, kb_id must start with 'KB'"})
    not_exist_kb_ids = local_doc_qa.mysql_client.check_kb_exist(user_id, [kb_id])
    if not not_exist_kb_ids:
        return sanic_json({"code": 2001, "msg": "fail, knowledge Base {} already exist".format(kb_id)})

    local_doc_qa.mysql_client.new_knowledge_base(kb_id, user_id, kb_name)
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
    not_exist_kb_ids = local_doc_qa.mysql_client.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
        return sanic_json({"code": 2001, "msg": msg, "data": [{}]})
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")
    exist_files = []
    if mode == 'soft':
        exist_files = local_doc_qa.mysql_client.check_file_exist_by_name(user_id, kb_id, [url])
    if exist_files:
        file_id, file_name, file_size, status = exist_files[0]
        msg = f'warning，当前的mode是soft，无法上传同名文件，如果想强制上传同名文件，请设置mode：strong'
        data = [{"file_id": file_id, "file_name": url, "status": status, "bytes": file_size, "timestamp": timestamp}]
    else:
        file_id, msg = local_doc_qa.mysql_client.add_file(user_id, kb_id, url, timestamp)
        local_file = LocalFile(user_id, kb_id, url, file_id, url, local_doc_qa.embeddings, is_url=True)
        local_doc_qa.mysql_client.update_file_path(file_id, local_file.file_path)
        data = [{"file_id": file_id, "file_name": url, "status": "gray", "bytes": 0, "timestamp": timestamp}]
        asyncio.create_task(local_doc_qa.insert_files_to_faiss(user_id, kb_id, [local_file]))
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

    not_exist_kb_ids = local_doc_qa.mysql_client.check_kb_exist(user_id, [kb_id])
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
        exist_files = local_doc_qa.mysql_client.check_file_exist_by_name(user_id, kb_id, file_names)
        exist_file_names = [f[1] for f in exist_files]

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")

    for file, file_name in zip(files, file_names):
        if file_name in exist_file_names:
            continue
        file_id, msg = local_doc_qa.mysql_client.add_file(user_id, kb_id, file_name, timestamp)
        debug_logger.info(f"{file_name}, {file_id}, {msg}")
        local_file = LocalFile(user_id, kb_id, file, file_id, file_name, local_doc_qa.embeddings)
        local_doc_qa.mysql_client.update_file_path(file_id, local_file.file_path)
        local_files.append(local_file)
        local_doc_qa.mysql_client.update_file_size(file_id, len(local_file.file_content))
        data.append(
            {"file_id": file_id, "file_name": file_name, "status": "gray", "bytes": len(local_file.file_content),
             "timestamp": timestamp})

    asyncio.create_task(local_doc_qa.insert_files_to_faiss(user_id, kb_id, local_files))
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
    kb_infos = local_doc_qa.mysql_client.get_knowledge_bases(user_id)
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
    file_infos = local_doc_qa.mysql_client.get_files(user_id, kb_id)
    status_count = {}
    msg_map = {'gray': "正在上传中，请耐心等待",
               'red': "split或embedding失败，请检查文件类型，仅支持[md,txt,pdf,jpg,png,jpeg,docx,xlsx,pptx,eml,csv]",
               'yellow': "faiss插入失败，请稍后再试", 'green': "上传成功"}
    for file_info in file_infos:
        status = file_info[2]
        if status not in status_count:
            status_count[status] = 1
        else:
            status_count[status] += 1
        data.append({"file_id": file_info[0], "file_name": file_info[1], "status": file_info[2], "bytes": file_info[3],
                     "content_length": file_info[4], "timestamp": file_info[5], "msg": file_info[6]})
        file_name = file_info[1]
        file_id = file_info[0]
        if file_name.endswith('.faq'):
            faq_info = local_doc_qa.mysql_client.get_faq(file_id)
            if faq_info:
                data[-1]['question'] = faq_info[2]
                data[-1]['answer'] = faq_info[3]

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
    not_exist_kb_ids = local_doc_qa.mysql_client.check_kb_exist(user_id, kb_ids)
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids)})

    for kb_id in kb_ids:
        local_doc_qa.faiss_client.delete_documents(kb_id=kb_id)
    local_doc_qa.mysql_client.delete_knowledge_base(user_id, kb_ids)
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
    not_exist_kb_ids = local_doc_qa.mysql_client.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids[0])})
    local_doc_qa.mysql_client.rename_knowledge_base(user_id, kb_id, new_kb_name)
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
    debug_logger.info("kb_id %s", kb_id)
    file_ids = safe_get(req, "file_ids")
    not_exist_kb_ids = local_doc_qa.mysql_client.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids[0])})
    valid_file_infos = local_doc_qa.mysql_client.check_file_exist(user_id, kb_id, file_ids)
    if len(valid_file_infos) == 0:
        return sanic_json({"code": 2004, "msg": "fail, files {} not found".format(file_ids)})
    local_doc_qa.faiss_client.delete_documents(kb_id=kb_id, file_ids=file_ids)
    # 删除数据库中的记录
    local_doc_qa.mysql_client.delete_files(kb_id, file_ids)
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
        users = local_doc_qa.mysql_client.get_users()
        users = [user[0] for user in users]
    else:
        users = [user_id]
    res = {}
    for user in users:
        res[user] = {}
        kbs = local_doc_qa.mysql_client.get_knowledge_bases(user)
        for kb_id, kb_name in kbs:
            gray_file_infos = local_doc_qa.mysql_client.get_file_by_status([kb_id], 'gray')
            red_file_infos = local_doc_qa.mysql_client.get_file_by_status([kb_id], 'red')
            yellow_file_infos = local_doc_qa.mysql_client.get_file_by_status([kb_id], 'yellow')
            green_file_infos = local_doc_qa.mysql_client.get_file_by_status([kb_id], 'green')
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
        kbs = local_doc_qa.mysql_client.get_knowledge_bases(user_id)
        kb_ids = [kb[0] for kb in kbs]
    else:
        not_exist_kb_ids = local_doc_qa.mysql_client.check_kb_exist(user_id, kb_ids)
        if not_exist_kb_ids:
            return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids)})

    gray_file_infos = local_doc_qa.mysql_client.get_file_by_status(kb_ids, status)
    gray_file_ids = [f[0] for f in gray_file_infos]
    gray_file_names = [f[1] for f in gray_file_infos]
    debug_logger.info(f'{status} files number: {len(gray_file_names)}')
    # 删除milvus中的file
    if gray_file_ids:
        for kb_id in kb_ids:
            local_doc_qa.faiss_client.delete_documents(kb_id=kb_id, file_ids=gray_file_ids)
            local_doc_qa.mysql_client.delete_files(kb_id, gray_file_ids)
    return sanic_json({"code": 200, "msg": f"delete {status} files success", "data": gray_file_names})


async def local_doc_chat(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    chat_user_id = user_id
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info('local_doc_chat %s', user_id)
    bot_id = safe_get(req, 'bot_id')
    if bot_id:
        if not local_doc_qa.mysql_client.check_bot_is_exist(bot_id):
            return sanic_json({"code": 2003, "msg": "fail, Bot {} not found".format(bot_id)})
        bot_info = local_doc_qa.mysql_client.get_bot(None, bot_id)[0]
        bot_id, bot_name, desc, image, prompt, welcome, model, kb_ids_str, upload_time, user_id = bot_info
        kb_ids = kb_ids_str.split(',')
        if not kb_ids:
            return sanic_json({"code": 2003, "msg": "fail, Bot {} unbound knowledge base.".format(bot_id)})
        custom_prompt = prompt
    else:
        kb_ids = safe_get(req, 'kb_ids')
        custom_prompt = safe_get(req, 'custom_prompt', None)
    question = safe_get(req, 'question')
    rerank = safe_get(req, 'rerank', default=True)
    debug_logger.info('rerank %s', rerank)
    streaming = safe_get(req, 'streaming', False)
    history = safe_get(req, 'history', [])
    product_source = safe_get(req, 'product_source', 'paas')
    need_web_search = safe_get(req, 'networking', False)
    model = local_doc_qa.model
    debug_logger.info("model: %s", model)
    debug_logger.info("product_source: %s", product_source)
    debug_logger.info("history: %s ", history)
    debug_logger.info("question: %s", question)
    debug_logger.info("kb_ids: %s", kb_ids)
    debug_logger.info("user_id: %s", user_id)
    debug_logger.info("custom_prompt: %s", custom_prompt)

    not_exist_kb_ids = local_doc_qa.mysql_client.check_kb_exist(user_id, kb_ids)
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids)})
    faq_kb_ids = [kb + '_FAQ' for kb in kb_ids]
    not_exist_faq_kb_ids = local_doc_qa.mysql_client.check_kb_exist(user_id, faq_kb_ids)
    exist_faq_kb_ids = [kb for kb in faq_kb_ids if kb not in not_exist_faq_kb_ids]
    debug_logger.info("exist_faq_kb_ids: %s", exist_faq_kb_ids)
    kb_ids += exist_faq_kb_ids

    file_infos = []
    for kb_id in kb_ids:
        file_infos.extend(local_doc_qa.mysql_client.get_files(user_id, kb_id))
    valid_files = [fi for fi in file_infos if fi[2] == 'green']
    if len(valid_files) == 0:
        return sanic_json({"code": 200, "msg": "当前知识库为空，请上传文件或等待文件解析完毕", "question": question,
                           "response": "All knowledge bases {} are empty or haven't green file, please upload files".format(
                               kb_ids), "history": history, "source_documents": [{}]})
    else:
        time_record = {}
        debug_logger.info("streaming: %s", streaming)
        if streaming:
            debug_logger.info("start generate answer")

            async def generate_answer(response):
                debug_logger.info("start generate...")
                async for resp, next_history in local_doc_qa.get_knowledge_based_answer(custom_prompt=custom_prompt,
                                                                                        query=question, 
                                                                                        kb_ids=kb_ids, 
                                                                                        chat_history=history, 
                                                                                        streaming=True, 
                                                                                        rerank=rerank,
                                                                                        need_web_search=need_web_search):
                    chunk_data = resp["result"]
                    if not chunk_data:
                        continue
                    chunk_str = chunk_data[6:]
                    if chunk_str.startswith("[DONE]"):
                        source_documents = []
                        for inum, doc in enumerate(resp["source_documents"]):
                            source_info = {'file_id': doc.metadata.get('source', doc.metadata.get('file_id','')),
                                           'file_name': doc.metadata.get('title', doc.metadata.get('file_name','')),
                                           'content': doc.page_content,
                                           'retrieval_query': doc.metadata['retrieval_query'],
                                           'score': str(doc.metadata['score'])}
                            source_documents.append(source_info)

                        retrieval_documents = format_source_documents(resp["retrieval_documents"])
                        source_documents = format_source_documents(resp["source_documents"])
                        chat_data = {'user_id': chat_user_id, 'bot_id': bot_id, 'kb_ids': kb_ids, 'query': question,
                                     'history': history, 'model': model, 'product_source': product_source,
                                     'time_record': time_record, 'prompt': resp['prompt'],
                                     'result': next_history[-1][1], 'condense_question': question,
                                     'retrieval_documents': retrieval_documents, 'source_documents': source_documents}
                        local_doc_qa.mysql_client.add_qalog(**chat_data)
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
            async for resp, history in local_doc_qa.get_knowledge_based_answer(custom_prompt=custom_prompt,
                                                                               query=question, 
                                                                               kb_ids=kb_ids, 
                                                                               chat_history=history, 
                                                                               streaming=False, 
                                                                               rerank=rerank,
                                                                               need_web_search=need_web_search):
                pass
            retrieval_documents = format_source_documents(resp["retrieval_documents"])
            source_documents = format_source_documents(resp["source_documents"])
            # chat_data = {'user_id': user_id, 'kb_ids': kb_ids, 'query': question, 'history': history,
            #              'retrieval_documents': retrieval_documents, 'prompt': resp['prompt'], 'result': resp['result'],
            #              'source_documents': source_documents}
            chat_data = {'user_id': chat_user_id, 'bot_id': bot_id, 'kb_ids': kb_ids, 'query': question,
                         'history': history, 'model': model, 'product_source': product_source,
                         'time_record': time_record, 'prompt': resp['prompt'], 'result': resp['result'],
                         'condense_question': question, 'retrieval_documents': retrieval_documents,
                         'source_documents': source_documents}
            local_doc_qa.mysql_client.add_qalog(**chat_data)
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


async def new_bot(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    bot_name = safe_get(req, "bot_name")
    desc = safe_get(req, "description", BOT_DESC)
    head_image = safe_get(req, "head_image", BOT_IMAGE)
    prompt_setting = safe_get(req, "prompt_setting", BOT_PROMPT)
    welcome_message = safe_get(req, "welcome_message", BOT_WELCOME)
    model = safe_get(req, "model", 'MiniChat-2-3B')
    kb_ids = safe_get(req, "kb_ids", [])
    kb_ids_str = ",".join(kb_ids)

    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    not_exist_kb_ids = local_doc_qa.mysql_client.check_kb_exist(user_id, kb_ids)
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
        return sanic_json({"code": 2001, "msg": msg, "data": [{}]})
    debug_logger.info("new_bot %s", user_id)
    bot_id = 'BOT' + uuid.uuid4().hex
    local_doc_qa.mysql_client.new_qanything_bot(bot_id, user_id, bot_name, desc, head_image, prompt_setting,
                                                welcome_message, model, kb_ids_str)
    create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return sanic_json({"code": 200, "msg": "success create qanything bot {}".format(bot_id),
                       "data": {"bot_id": bot_id, "bot_name": bot_name, "create_time": create_time}})


async def delete_bot(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("delete_bot %s", user_id)
    bot_id = safe_get(req, 'bot_id')
    if not local_doc_qa.mysql_client.check_bot_is_exist(bot_id):
        return sanic_json({"code": 2003, "msg": "fail, Bot {} not found".format(bot_id)})
    local_doc_qa.mysql_client.delete_bot(user_id, bot_id)
    return sanic_json({"code": 200, "msg": "Bot {} delete success".format(bot_id)})


async def update_bot(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("update_bot %s", user_id)
    bot_id = safe_get(req, 'bot_id')
    if not local_doc_qa.mysql_client.check_bot_is_exist(bot_id):
        return sanic_json({"code": 2003, "msg": "fail, Bot {} not found".format(bot_id)})
    bot_info = local_doc_qa.mysql_client.get_bot(user_id, bot_id)[0]
    bot_name = safe_get(req, "bot_name", bot_info[1])
    description = safe_get(req, "description", bot_info[2])
    head_image = safe_get(req, "head_image", bot_info[3])
    prompt_setting = safe_get(req, "prompt_setting", bot_info[4])
    welcome_message = safe_get(req, "welcome_message", bot_info[5])
    model = safe_get(req, "model", bot_info[6])
    kb_ids = safe_get(req, "kb_ids")
    if kb_ids is not None:
        not_exist_kb_ids = local_doc_qa.mysql_client.check_kb_exist(user_id, kb_ids)
        if not_exist_kb_ids:
            msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
            return sanic_json({"code": 2001, "msg": msg, "data": [{}]})
        kb_ids_str = ",".join(kb_ids)
    else:
        kb_ids_str = bot_info[7]
    # 判断哪些项修改了
    if bot_name != bot_info[1]:
        debug_logger.info(f"update bot name from {bot_info[1]} to {bot_name}")
    if description != bot_info[2]:
        debug_logger.info(f"update bot description from {bot_info[2]} to {description}")
    if head_image != bot_info[3]:
        debug_logger.info(f"update bot head_image from {bot_info[3]} to {head_image}")
    if prompt_setting != bot_info[4]:
        debug_logger.info(f"update bot prompt_setting from {bot_info[4]} to {prompt_setting}")
    if welcome_message != bot_info[5]:
        debug_logger.info(f"update bot welcome_message from {bot_info[5]} to {welcome_message}")
    if model != bot_info[6]:
        debug_logger.info(f"update bot model from {bot_info[6]} to {model}")
    if kb_ids_str != bot_info[7]:
        debug_logger.info(f"update bot kb_ids from {bot_info[7]} to {kb_ids_str}")
    #  update_time     TIMESTAMP DEFAULT CURRENT_TIMESTAMP 根据这个mysql的格式获取现在的时间
    update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    debug_logger.info(f"update_time: {update_time}")
    local_doc_qa.mysql_client.update_bot(user_id, bot_id, bot_name, description, head_image, prompt_setting,
                                         welcome_message, model, kb_ids_str, update_time)
    return sanic_json({"code": 200, "msg": "Bot {} update success".format(bot_id)})


async def get_bot_info(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    bot_id = safe_get(req, 'bot_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    if bot_id:
        if not local_doc_qa.mysql_client.check_bot_is_exist(bot_id):
            return sanic_json({"code": 2003, "msg": "fail, Bot {} not found".format(bot_id)})
    debug_logger.info("get_bot_info %s", user_id)
    bot_infos = local_doc_qa.mysql_client.get_bot(user_id, bot_id)
    data = []
    for bot_info in bot_infos:
        if bot_info[7] != "":
            kb_ids = bot_info[7].split(',')
            kb_infos = local_doc_qa.mysql_client.get_knowledge_base_name(kb_ids)
            kb_names = []
            for kb_id in kb_ids:
                for kb_info in kb_infos:
                    if kb_id == kb_info[1]:
                        kb_names.append(kb_info[2])
                        break
        else:
            kb_ids = []
            kb_names = []
        info = {"bot_id": bot_info[0], "user_id": user_id, "bot_name": bot_info[1], "description": bot_info[2],
                "head_image": bot_info[3], "prompt_setting": bot_info[4], "welcome_message": bot_info[5],
                "model": bot_info[6], "kb_ids": kb_ids, "kb_names": kb_names, "update_time": bot_info[8]}
        data.append(info)
    return sanic_json({"code": 200, "msg": "success", "data": data})


async def upload_faqs(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("upload_faqs %s", user_id)
    kb_id = safe_get(req, 'kb_id')
    debug_logger.info("kb_id %s", kb_id)
    faqs = safe_get(req, 'faqs')
    file_status = {}
    if faqs is None:
        files = req.files.getlist('files')
        faqs = []
        for file in files:
            debug_logger.info('ori name: %s', file.name)
            file_name = urllib.parse.unquote(file.name, encoding='UTF-8')
            debug_logger.info('decode name: %s', file_name)
            # 删除掉全角字符
            file_name = re.sub(r'[\uFF01-\uFF5E\u3000-\u303F]', '', file_name)
            file_name = file_name.replace("/", "_")
            debug_logger.info('cleaned name: %s', file_name)
            file_name = truncate_filename(file_name)
            file_faqs = check_and_transform_excel(file.body)
            if isinstance(file_faqs, str):
                file_status[file_name] = file_faqs
            else:
                faqs.extend(file_faqs)
                file_status[file_name] = "success"

    if len(faqs) > 1000:
        return sanic_json({"code": 2002, "msg": f"fail, faqs too many, The maximum length of each request is 1000."})

    not_exist_kb_ids = local_doc_qa.mysql_client.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
        return sanic_json({"code": 2001, "msg": msg})

    data = []
    now = datetime.now()
    local_files = []
    timestamp = now.strftime("%Y%m%d%H%M")
    debug_logger.info(f"start insert {len(faqs)} faqs to mysql, user_id: {user_id}, kb_id: {kb_id}")
    exist_questions = []
    for faq in tqdm(faqs):
        ques = faq['question']
        if ques not in exist_questions:
            exist_questions.append(ques)
        else:
            debug_logger.info(f"question {ques} already exists, skip it")
            continue
        if len(ques) > 512 or len(faq['answer']) > 2048:
            return sanic_json({"code": 2003, "msg": f"fail, faq too long, max length of question is 512, answer is 2048."})
        content_length = len(ques) + len(faq['answer'])
        file_name = f"FAQ_{ques}.faq"
        file_name = file_name.replace("/", "_").replace(":", "_")  # 文件名中的/和：会导致写入时出错
        file_name = simplify_filename(file_name)
        file_id, msg = local_doc_qa.mysql_client.add_file(user_id, kb_id, file_name, timestamp, status='green')
        local_file = LocalFile(user_id, kb_id, faq, file_id, file_name, local_doc_qa.embeddings)
        local_doc_qa.mysql_client.update_file_path(file_id, local_file.file_path)
        local_files.append(local_file)
        local_doc_qa.mysql_client.add_faq(file_id, user_id, kb_id, ques, faq['answer'], faq.get("nos_keys", ""))
        # debug_logger.info(f"{file_name}, {file_id}, {msg}, {faq}")
        data.append(
            {"file_id": file_id, "file_name": file_name, "status": "gray", "length": content_length,
             "timestamp": timestamp})
    debug_logger.info(f"end insert {len(faqs)} faqs to mysql, user_id: {user_id}, kb_id: {kb_id}")

    asyncio.create_task(local_doc_qa.insert_files_to_faiss(user_id, kb_id, local_files))

    msg = "success，后台正在飞速上传文件，请耐心等待"
    return sanic_json({"code": 200, "msg": msg, "file_status": file_status, "data": data})


async def get_file_base64(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("get_file_base64 %s", user_id)
    file_id = safe_get(req, 'file_id')
    debug_logger.info("file_id: {}".format(file_id))
    file_path = local_doc_qa.mysql_client.get_file_path(file_id)
    if file_path in ['URL', 'FAQ', 'UNK']:
        return sanic_json({"code": 2003, "msg": f"fail, {file_path} file has no base64"})
    with open(file_path, 'rb') as f:
        content = f.read()
    base64_content = base64.b64encode(content).decode()
    return sanic_json({"code": 200, "msg": "success", "base64_content": base64_content})


async def get_qa_info(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'输入非法！request.json：{req.json}，请检查！'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info("get_qa_info %s", user_id)
    kb_ids = safe_get(req, 'kb_ids')
    query = safe_get(req, 'query')
    bot_id = safe_get(req, 'bot_id')
    time_start = safe_get(req, 'time_start')
    time_end = safe_get(req, 'time_end')
    # 检查time_end和time_start是否满足2024-10-05的格式
    if time_start:
        if not re.match(r'\d{4}-\d{2}-\d{2}', time_start):
            return sanic_json({"code": 2002, "msg": f'输入非法！time_start格式错误，time_start: {time_start}，示例：2024-10-05，请检查！'})
    if time_end:
        if not re.match(r'\d{4}-\d{2}-\d{2}', time_end):
            return sanic_json({"code": 2002, "msg": f'输入非法！time_end格式错误，time_end: {time_end}，示例：2024-10-05，请检查！'})
    time_range = None
    if time_end and time_start:
        time_range = (time_start, time_end)
    elif time_start:  # 2024-10-05
        time_range = (time_start + " 00:00:00", time_start + " 23:59:59")

    page_id = safe_get(req, 'page_id')
    default_need_info = ["qa_id", "user_id", "bot_id", "kb_ids", "query", "model", "product_source", "time_record",
                         "history", "condense_question", "prompt", "result", "retrieval_documents", "source_documents",
                         "timestamp"]
    need_info = safe_get(req, 'need_info', default_need_info)
    save_to_excel = safe_get(req, 'save_to_excel', False)
    qa_infos = local_doc_qa.mysql_client.get_qalog_by_filter(need_info=need_info, user_id=user_id, kb_ids=kb_ids,
                                                             query=query, bot_id=bot_id, time_range=time_range)
    if save_to_excel:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        file_name = f"QAnything_QA_{timestamp}.xlsx"
        file_path = export_qalogs_to_excel(qa_infos, need_info, file_name)
        return await response.file(file_path, filename=file_name,
                                   mime_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                   headers={'Content-Disposition': f'attachment; filename="{file_name}"'})

    if len(qa_infos) > 100:
        pages = math.ceil(len(qa_infos) // 100)
        if page_id is None:
            msg = f"检索到的Log数超过100，需要分页返回，总数为{len(qa_infos)}, 请使用page_id参数获取某一页数据，参数范围：[0, {pages - 1}], 本次返回page_id为0的数据"
            qa_infos = qa_infos[:100]
            page_id = 0
        elif page_id >= pages:
            return sanic_json({"code": 2002, "msg": f'输入非法！page_id超过最大值，page_id: {page_id}，最大值：{pages - 1}，请检查！'})
        else:
            msg = f"检索到的Log数超过100，需要分页返回，总数为{len(qa_infos)}, page范围：[0, {pages - 1}], 本次返回page_id为{page_id}的数据"
            qa_infos = qa_infos[page_id * 100:(page_id + 1) * 100]
    else:
        msg = f"检索到的Log数为{len(qa_infos)}，一次返回所有数据"
        page_id = 0
    return sanic_json({"code": 200, "msg": msg, "page_id": page_id, "qa_infos": qa_infos})

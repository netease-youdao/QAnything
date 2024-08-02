from qanything_kernel.configs.model_config import VECTOR_SEARCH_TOP_K, VECTOR_SEARCH_SCORE_THRESHOLD, \
    PROMPT_TEMPLATE, STREAMING, SYSTEM, INSTRUCTIONS, LOCAL_RERANK_PATH
from typing import List, Tuple, Union
import time
from qanything_kernel.connector.embedding.embedding_for_online_client import YouDaoEmbeddings
from qanything_kernel.connector.rerank.rerank_for_online_client import YouDaoRerank
from qanything_kernel.connector.llm import OpenAILLM
from langchain.schema import Document
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from qanything_kernel.core.retriever.vectorstore import VectorStoreMilvusClient
from qanything_kernel.core.retriever.elasticsearchstore import StoreElasticSearchClient
from qanything_kernel.core.retriever.parent_retriever import ParentRetriever
from qanything_kernel.utils.general_utils import clear_string_is_equal
from qanything_kernel.utils.general_utils import get_time, clear_string, get_time_async, num_tokens, cosine_similarity, \
    num_tokens_local, deduplicate_documents
from qanything_kernel.utils.custom_log import debug_logger, qa_logger, rerank_logger
from qanything_kernel.core.chains.condense_q_chain import RewriteQuestionChain
from qanything_kernel.core.tools.web_search_tool import duckduckgo_search
from transformers import AutoTokenizer
import requests
import json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import traceback
import re


class LocalDocQA:
    def __init__(self, port):
        self.port = port
        self.milvus_cache = None
        self.embeddings: YouDaoEmbeddings = None
        self.rerank: YouDaoRerank = None
        self.top_k: int = VECTOR_SEARCH_TOP_K
        self.chunk_conent: bool = True
        self.score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD
        self.milvus_kb: VectorStoreMilvusClient = None
        self.retriever: ParentRetriever = None
        self.milvus_summary: KnowledgeBaseManager = None
        self.es_client: StoreElasticSearchClient = None
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(LOCAL_RERANK_PATH, local_files_only=True)
        self.session = self.create_retry_session(retries=3, backoff_factor=1)

    @staticmethod
    def create_retry_session(retries, backoff_factor):
        session = requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def init_cfg(self, args=None):
        self.embeddings = YouDaoEmbeddings()
        self.rerank = YouDaoRerank()
        self.milvus_summary = KnowledgeBaseManager()
        self.milvus_kb = VectorStoreMilvusClient()
        self.es_client = StoreElasticSearchClient()
        self.retriever = ParentRetriever(self.milvus_kb, self.milvus_summary, self.es_client)

    @get_time
    def get_web_search(self, queries, top_k=None):
        if not top_k:
            top_k = self.top_k
        query = queries[0]
        web_content, web_documents = duckduckgo_search(query, top_k)
        source_documents = []
        for idx, doc in enumerate(web_documents):
            doc.metadata['retrieval_query'] = query  # 添加查询到文档的元数据中
            file_name = re.sub(r'[\uFF01-\uFF5E\u3000-\u303F]', '', doc.metadata['title'])
            doc.metadata['file_name'] = file_name + '.web'
            doc.metadata['file_url'] = doc.metadata['source']
            doc.metadata['embed_version'] = self.embeddings.embed_version
            doc.metadata['score'] = 1 - (idx / len(web_documents))
            source_documents.append(doc)
            if 'description' in doc.metadata:
                desc_doc = Document(page_content=doc.metadata['description'], metadata=doc.metadata)
                source_documents.append(desc_doc)
        # source_documents = self.web_splitter.split_documents(source_documents)
        return web_content, source_documents

    def web_page_search(self, query, top_k=None):
        # 防止get_web_search调用失败，需要try catch
        try:
            web_content, source_documents = self.get_web_search([query], top_k)
        except Exception as e:
            debug_logger.error(f"web search error: {traceback.format_exc()}")
            return []

        return source_documents

    @get_time_async
    async def get_source_documents(self, query, retriever: ParentRetriever, kb_ids, time_record, hybrid_search):
        source_documents = []
        start_time = time.perf_counter()
        query_docs = await retriever.get_retrieved_documents(query, partition_keys=kb_ids, time_record=time_record,
                                                             hybrid_search=hybrid_search)
        end_time = time.perf_counter()
        time_record['retriever_search'] = round(end_time - start_time, 2)
        debug_logger.info(f"retriever_search time: {time_record['retriever_search']}s")
        # debug_logger.info(f"query_docs num: {len(query_docs)}, query_docs: {query_docs}")
        for idx, doc in enumerate(query_docs):
            if retriever.mysql_client.is_deleted_file(doc.metadata['file_id']):
                debug_logger.warning(f"file_id: {doc.metadata['file_id']} is deleted")
                continue
            doc.metadata['retrieval_query'] = query  # 添加查询到文档的元数据中
            doc.metadata['embed_version'] = self.embeddings.embed_version
            if 'score' not in doc.metadata:
                doc.metadata['score'] = 1 - (idx / len(query_docs))  # TODO 这个score怎么获取呢
            source_documents.append(doc)
        # if cosine_thresh:
        #     source_documents = [item for item in source_documents if float(item.metadata['score']) > cosine_thresh]

        return source_documents

    def reprocess_source_documents(self, custom_llm: OpenAILLM, query: str,
                                   source_docs: List[Document],
                                   history: List[str],
                                   prompt_template: str) -> Tuple[List[Document], int]:
        # 组装prompt,根据max_token
        query_token_num = custom_llm.num_tokens_from_messages([query]) * 4
        history_token_num = custom_llm.num_tokens_from_messages([x for sublist in history for x in sublist])
        template_token_num = custom_llm.num_tokens_from_messages([prompt_template])
        limited_token_nums = custom_llm.token_window - custom_llm.max_token - custom_llm.offcut_token - query_token_num - history_token_num - template_token_num

        debug_logger.info(f"limited token nums: {limited_token_nums}")
        debug_logger.info(f"template token nums: {template_token_num}")
        debug_logger.info(f"query token nums: {query_token_num}")
        debug_logger.info(f"history token nums: {history_token_num}")

        # if limited_token_nums < 200:
        #     return []
        # 从最后一个往前删除，直到长度合适,这样是最优的，因为超长度的情况比较少见
        # 已知箱子容量，装满这个箱子
        new_source_docs = []
        total_token_num = 0
        for doc in source_docs:
            doc_token_num = custom_llm.num_tokens_from_docs([doc])
            # 计算metadata_infos的token数量
            if 'kb_id' in doc.metadata:
                kb_name = self.milvus_summary.get_knowledge_base_name([doc.metadata['kb_id']])
                kb_name = kb_name[0][2]
                metadata_infos = f"知识库名: {kb_name}\n"
                if 'file_name' in doc.metadata:
                    metadata_infos += f"文件名: {doc.metadata['file_name']}\n"
                metadata_infos += '文件内容如下: \n'
                metadata_infos_token_num = custom_llm.num_tokens_from_messages([metadata_infos])
            else:
                metadata_infos_token_num = 0
            doc_token_num += metadata_infos_token_num
            if total_token_num + doc_token_num <= limited_token_nums:
                new_source_docs.append(doc)
                total_token_num += doc_token_num
            else:
                break

        debug_logger.info(f"new_source_docs token nums: {custom_llm.num_tokens_from_docs(new_source_docs)}")
        return new_source_docs, limited_token_nums

    def generate_prompt(self, query, source_docs, custom_prompt):
        # 获取今日日期
        today = time.strftime("%Y-%m-%d", time.localtime())
        # 获取当前时间
        now = time.strftime("%H:%M:%S", time.localtime())
        if source_docs:
            context = "\n".join([doc.page_content for doc in source_docs])
            context = context.replace("{", "{{").replace("}", "}}")  # 防止content中包含{}

            prompt_template = PROMPT_TEMPLATE.replace("{context}", context)
            prompt = prompt_template.format(system=SYSTEM.format(today_date=today, current_time=now),
                                            user_instructions=custom_prompt, instructions=INSTRUCTIONS)
            prompt = prompt.replace("{question}", query)
        else:
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = f"""
                - You are a helpful assistant. You can help me by answering my questions. You can also ask me questions.
                - Today's date is {today}. The current time is {now}.
                """
            prompt += f"""
            - Now, answer the following question:
            {query}
            - Return your answer in Markdown formatting, and in the same language as the question "{query}". 
            """
        return prompt

    async def get_rerank_results(self, query, doc_ids=None, doc_strs=None):
        docs = []
        if doc_strs:
            docs = [Document(page_content=doc_str) for doc_str in doc_strs]
        else:
            for doc_id in doc_ids:
                doc_json = self.milvus_summary.get_document_by_doc_id(doc_id)
                if doc_json is None:
                    docs.append(None)
                    continue
                user_id, file_id, file_name, kb_id = doc_json['kwargs']['metadata']['user_id'], \
                    doc_json['kwargs']['metadata']['file_id'], doc_json['kwargs']['metadata']['file_name'], \
                    doc_json['kwargs']['metadata']['kb_id']
                doc = Document(page_content=doc_json['kwargs']['page_content'], metadata=doc_json['kwargs']['metadata'])
                doc.metadata['doc_id'] = doc_id
                doc.metadata['retrieval_query'] = query
                doc.metadata['embed_version'] = self.embeddings.embed_version
                if file_name.endswith('.faq'):
                    faq_dict = doc.metadata['faq_dict']
                    page_content = f"{faq_dict['question']}：{faq_dict['answer']}"
                    nos_keys = faq_dict.get('nos_keys')
                    doc.page_content = page_content
                    doc.metadata['nos_keys'] = nos_keys
                docs.append(doc)

        if len(docs) > 1 and num_tokens_local(query, self.rerank_tokenizer) <= 300:
            try:
                debug_logger.info(f"use rerank, rerank docs num: {len(docs)}")
                docs = await self.rerank.arerank_documents(query, docs)
                if len(docs) > 1:
                    docs = [doc for doc in docs if float(doc.metadata['score']) >= 0.28]
                return docs
            except Exception as e:
                debug_logger.error(f"query tokens: {num_tokens(query)}, rerank error: {e}")
                embed1 = await self.embeddings.aembed_query(query)
                for doc in docs:
                    embed2 = self.embeddings.aembed_query(doc.page_content)
                    doc.metadata['score'] = cosine_similarity(embed1, embed2)
                return docs
        else:
            embed1 = await self.embeddings.aembed_query(query)
            for doc in docs:
                embed2 = await self.embeddings.aembed_query(doc.page_content)
                doc.metadata['score'] = cosine_similarity(embed1, embed2)
            return docs

    async def get_knowledge_based_answer(self, model, max_token, kb_ids, query, retriever, custom_prompt, time_record,
                                         temperature, api_base, api_key, api_context_length, top_p, parent_chunk_size,
                                         chat_history=None, streaming: bool = STREAMING, rerank: bool = False,
                                         only_need_search_results: bool = False, need_web_search=False,
                                         hybrid_search=False):
        custom_llm = OpenAILLM(model, max_token, api_base, api_key, api_context_length, top_p, temperature)
        if chat_history is None:
            chat_history = []
        retrieval_query = query
        condense_question = query
        if chat_history:
            formatted_chat_history = []
            for msg in chat_history:
                formatted_chat_history += [
                    HumanMessage(content=msg[0]),
                    AIMessage(content=msg[1]),
                ]
            debug_logger.info(f"formatted_chat_history: {formatted_chat_history}")

            rewrite_q_chain = RewriteQuestionChain(model_name=model, openai_api_base=api_base, openai_api_key=api_key)
            full_prompt = rewrite_q_chain.condense_q_prompt.format(
                chat_history=formatted_chat_history,
                question=query
            )
            while custom_llm.num_tokens_from_messages([full_prompt]) >= 4096 - 256:
                formatted_chat_history = formatted_chat_history[2:]
                full_prompt = rewrite_q_chain.condense_q_prompt.format(
                    chat_history=formatted_chat_history,
                    question=query
                )
            debug_logger.info(
                f"Subtract formatted_chat_history: {len(chat_history) * 2} -> {len(formatted_chat_history)}")
            try:
                t1 = time.perf_counter()
                condense_question = await rewrite_q_chain.condense_q_chain.ainvoke(
                    {
                        "chat_history": formatted_chat_history,
                        "question": query,
                    },
                )
                t2 = time.perf_counter()
                # 时间保留两位小数
                time_record['condense_q_chain'] = round(t2 - t1, 2)
                time_record['rewrite_completion_tokens'] = custom_llm.num_tokens_from_messages([condense_question])
                debug_logger.info(f"condense_q_chain time: {time_record['condense_q_chain']}s")
            except Exception as e:
                debug_logger.error(f"condense_q_chain error: {e}")
                condense_question = query
            # 生成prompt
            # full_prompt = condense_q_prompt.format_messages(
            #     chat_history=formatted_chat_history,
            #     question=query
            # )
            # qa_logger.info(f"condense_q_chain full_prompt: {full_prompt}, condense_question: {condense_question}")
            debug_logger.info(f"condense_question: {condense_question}")
            time_record['rewrite_prompt_tokens'] = custom_llm.num_tokens_from_messages([full_prompt, condense_question])
            # 判断两个字符串是否相似：只保留中文，英文和数字
            if clear_string(condense_question) != clear_string(query):
                retrieval_query = condense_question

        if kb_ids:
            source_documents = await self.get_source_documents(retrieval_query, retriever, kb_ids, time_record,
                                                               hybrid_search)
        else:
            source_documents = []

        if need_web_search:
            t1 = time.perf_counter()
            web_search_results = self.web_page_search(query, top_k=3)
            web_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " ", ""],
                chunk_size=parent_chunk_size,
                chunk_overlap=int(parent_chunk_size / 4),
                length_function=num_tokens,
            )
            web_search_results = web_splitter.split_documents(web_search_results)
            t2 = time.perf_counter()
            time_record['web_search'] = round(t2 - t1, 2)
            source_documents += web_search_results

        source_documents = deduplicate_documents(source_documents)
        if rerank and len(source_documents) > 1 and num_tokens_local(query, self.rerank_tokenizer) <= 300:
            try:
                t1 = time.perf_counter()
                debug_logger.info(f"use rerank, rerank docs num: {len(source_documents)}")
                source_documents = await self.rerank.arerank_documents(condense_question, source_documents)
                t2 = time.perf_counter()
                time_record['rerank'] = round(t2 - t1, 2)
                # 过滤掉低分的文档
                if len(source_documents) > 1:
                    source_documents = [doc for doc in source_documents if float(doc.metadata['score']) >= 0.28]
            except Exception as e:
                time_record['rerank'] = 0.0
                debug_logger.error(f"query {query}: kb_ids: {kb_ids}, rerank error: {traceback.format_exc()}")

        high_score_faq_documents = [doc for doc in source_documents if
                                    doc.metadata['file_name'].endswith('.faq') and float(doc.metadata['score'] >= 0.9)]
        if high_score_faq_documents:
            source_documents = high_score_faq_documents
        # FAQ完全匹配处理逻辑
        for doc in source_documents:
            if doc.metadata['file_name'].endswith('.faq') and clear_string_is_equal(
                    doc.metadata['faq_dict']['question'], query):
                debug_logger.info(f"match faq question: {query}")
                if only_need_search_results:
                    yield source_documents, None
                    return
                res = doc.metadata['faq_dict']['answer']
                history = chat_history + [[query, res]]
                if streaming:
                    res = 'data: ' + json.dumps({'answer': res}, ensure_ascii=False)
                response = {"query": query,
                            "prompt": 'MATCH_FAQ',
                            "result": res,
                            "condense_question": condense_question,
                            "retrieval_documents": source_documents,
                            "source_documents": source_documents}
                time_record['llm_completed'] = 0.0
                time_record['total_tokens'] = 0
                time_record['prompt_tokens'] = 0
                time_record['completion_tokens'] = 0
                yield response, history
                if streaming:
                    response['result'] = "data: [DONE]\n\n"
                    yield response, history
                # 退出函数
                return

        # 获取今日日期
        today = time.strftime("%Y-%m-%d", time.localtime())
        # 获取当前时间
        now = time.strftime("%H:%M:%S", time.localtime())

        system_prompt = SYSTEM.format(today_date=today, current_time=now)
        t1 = time.perf_counter()
        if source_documents:
            if custom_prompt:
                prompt_template = PROMPT_TEMPLATE.replace("{system}", system_prompt).replace("{instructions}",
                                                                                             INSTRUCTIONS).replace(
                    "{user_instructions}", custom_prompt)
            else:
                prompt_template = PROMPT_TEMPLATE.replace("{system}", system_prompt).replace("{instructions}",
                                                                                             INSTRUCTIONS)
        else:
            if custom_prompt:
                prompt_template = custom_prompt
            else:
                prompt_template = f"""
                - You are a helpful assistant. You can help me by answering my questions. You can also ask me questions.
                - Today's date is {today}. The current time is {now}.
                - Now, answer the following question:
                {{question}}
                Return your answer in Markdown formatting, and in the same language as the question "{{question}}". 
                """

        retrieval_documents, limited_token_nums = self.reprocess_source_documents(custom_llm=custom_llm, query=query,
                                                                                  source_docs=source_documents,
                                                                                  history=chat_history,
                                                                                  prompt_template=prompt_template)

        if not need_web_search:
            try:
                new_docs = self.aggregate_documents(retrieval_documents, limited_token_nums, custom_llm)
                if new_docs:
                    source_documents = new_docs
                else:
                    # 合并所有候选文档，从前往后，所有file_id相同的文档合并，按照doc_id排序
                    merged_documents_file_ids = []
                    for doc in retrieval_documents:
                        if doc.metadata['file_id'] not in merged_documents_file_ids:
                            merged_documents_file_ids.append(doc.metadata['file_id'])
                    source_documents = []
                    for file_id in merged_documents_file_ids:
                        docs = [doc for doc in retrieval_documents if doc.metadata['file_id'] == file_id]
                        docs = sorted(docs, key=lambda x: int(x.metadata['doc_id'].split('_')[-1]))
                        source_documents.extend(docs)

                # 补充metadata信息
                for idx, doc in enumerate(source_documents):
                    # $print(kb_names, idx, flush=True)
                    kb_name = self.milvus_summary.get_knowledge_base_name([doc.metadata['kb_id']])
                    kb_name = kb_name[0][2]
                    metadata_infos = f"知识库名: {kb_name}\n"
                    if 'file_name' in doc.metadata:
                        metadata_infos += f"文件名: {doc.metadata['file_name']}\n"
                    doc.page_content = metadata_infos + '文件内容如下: \n' + doc.page_content
            except Exception as e:
                debug_logger.error(f"aggregate_documents error: {traceback.format_exc()}")
                source_documents = retrieval_documents
        else:
            source_documents = retrieval_documents

        t2 = time.perf_counter()
        time_record['reprocess'] = round(t2 - t1, 2)
        if only_need_search_results:
            yield source_documents, None
            return
        prompt = self.generate_prompt(query=query,
                                      source_docs=source_documents,
                                      custom_prompt=custom_prompt)

        t1 = time.perf_counter()
        has_first_return = False

        acc_resp = ''
        est_prompt_tokens = num_tokens(prompt) + num_tokens(str(chat_history))
        async for answer_result in custom_llm.generatorAnswer(prompt=prompt, history=chat_history, streaming=streaming):
            resp = answer_result.llm_output["answer"]

            prompt = answer_result.prompt
            history = answer_result.history
            total_tokens = answer_result.total_tokens
            prompt_tokens = answer_result.prompt_tokens
            completion_tokens = answer_result.completion_tokens
            history[-1][0] = query
            response = {"query": query,
                        "prompt": prompt,
                        "result": resp,
                        "condense_question": condense_question,
                        "retrieval_documents": retrieval_documents,
                        "source_documents": source_documents}
            time_record['prompt_tokens'] = prompt_tokens if prompt_tokens != 0 else est_prompt_tokens
            time_record['completion_tokens'] = completion_tokens if completion_tokens != 0 else num_tokens(acc_resp)
            time_record['total_tokens'] = total_tokens if total_tokens != 0 else time_record['prompt_tokens'] + \
                                                                                 time_record['completion_tokens']
            if has_first_return is False:
                first_return_time = time.perf_counter()
                has_first_return = True
                time_record['llm_first_return'] = round(first_return_time - t1, 2)
            if resp[6:].startswith("[DONE]"):
                last_return_time = time.perf_counter()
                time_record['llm_completed'] = round(last_return_time - t1, 2) - time_record['llm_first_return']
            yield response, history

    def get_completed_document(self, file_id, limit=None):
        sorted_json_datas = self.milvus_summary.get_document_by_file_id(file_id)
        if limit:
            sorted_json_datas = sorted_json_datas[limit[0]: limit[1] + 1]
        completed_content = ''
        existing_titles = []
        for doc_json in sorted_json_datas:
            doc = Document(page_content=doc_json['kwargs']['page_content'], metadata=doc_json['kwargs']['metadata'])
            if 'title_lst' in doc.metadata:
                title_lst = doc.metadata['title_lst']
                title_lst = [t for t in title_lst if t.replace('#', '') != '']
                first_appearance_titles = []
                for title in title_lst:
                    if title in existing_titles:
                        continue
                    first_appearance_titles.append(title)
                existing_titles += first_appearance_titles
                # 删除所有仅有多个#的title
                if doc.page_content == "":  # page_content为空时把first_appearance_titles当做正文
                    cleaned_list = [re.sub(r'^#+\s*', '', item) for item in first_appearance_titles]
                    doc.page_content = '\n'.join(cleaned_list)
                else:
                    doc.page_content = '\n'.join(first_appearance_titles) + '\n' + doc.page_content
            completed_content += doc.page_content + '\n'
        completed_doc = Document(page_content=completed_content, metadata=sorted_json_datas[0]['kwargs']['metadata'])
        return completed_doc

    def aggregate_documents(self, source_documents, limited_token_nums, custom_llm):
        # 聚合文档，具体逻辑是帮我判断所有候选是否集中在一个或两个文件中，是的话直接返回这一个或两个完整文档，如果tokens不够则截取文档中的完整上下文
        first_file_dict = {}
        second_file_dict = {}
        for doc in source_documents:
            file_id = doc.metadata['file_id']
            if not first_file_dict:
                first_file_dict['file_id'] = file_id
                first_file_dict['doc_ids'] = [int(doc.metadata['doc_id'].split('_')[-1])]
                first_file_dict['score'] = doc.metadata['score']
            elif first_file_dict['file_id'] == file_id:
                first_file_dict['doc_ids'].append(int(doc.metadata['doc_id'].split('_')[-1]))
            elif not second_file_dict:
                second_file_dict['file_id'] = file_id
                second_file_dict['doc_ids'] = [int(doc.metadata['doc_id'].split('_')[-1])]
                second_file_dict['score'] = doc.metadata['score']
            elif second_file_dict['file_id'] == file_id:
                second_file_dict['doc_ids'].append(int(doc.metadata['doc_id'].split('_')[-1]))
            else:  # 如果有第三个文件，直接返回
                return source_documents

        new_docs = []
        first_completed_doc = self.get_completed_document(first_file_dict['file_id'])
        first_completed_doc.metadata['score'] = first_file_dict['score']
        first_doc_tokens = custom_llm.num_tokens_from_docs([first_completed_doc])
        if first_doc_tokens > limited_token_nums:
            # 获取first_file_dict['doc_ids']的最小值和最大值
            doc_limit = [min(first_file_dict['doc_ids']), max(first_file_dict['doc_ids'])]
            first_completed_doc_limit = self.get_completed_document(first_file_dict['file_id'], doc_limit)
            first_completed_doc_limit.metadata['score'] = first_file_dict['score']
            first_doc_tokens = custom_llm.num_tokens_from_docs([first_completed_doc_limit])
            if first_doc_tokens > limited_token_nums:
                debug_logger.info(
                    f"first_limit_doc_tokens {doc_limit}: {first_doc_tokens} > limited_token_nums: {limited_token_nums}")
                return new_docs
            else:
                debug_logger.info(
                    f"first_limit_doc_tokens {doc_limit}: {first_doc_tokens} <= limited_token_nums: {limited_token_nums}")
                new_docs.append(first_completed_doc_limit)
        else:
            debug_logger.info(f"first_doc_tokens: {first_doc_tokens} <= limited_token_nums: {limited_token_nums}")
            new_docs.append(first_completed_doc)
        if second_file_dict:
            second_completed_doc = self.get_completed_document(second_file_dict['file_id'])
            second_completed_doc.metadata['score'] = second_file_dict['score']
            second_doc_tokens = custom_llm.num_tokens_from_docs([second_completed_doc])
            if first_doc_tokens + second_doc_tokens > limited_token_nums:
                doc_limit = [min(second_file_dict['doc_ids']), max(second_file_dict['doc_ids'])]
                second_completed_doc_limit = self.get_completed_document(second_file_dict['file_id'], doc_limit)
                second_completed_doc_limit.metadata['score'] = second_file_dict['score']
                second_doc_tokens = custom_llm.num_tokens_from_docs([second_completed_doc_limit])
                if first_doc_tokens + second_doc_tokens > limited_token_nums:
                    debug_logger.info(
                        f"first_doc_tokens: {first_doc_tokens} + second_limit_doc_tokens {doc_limit}: {second_doc_tokens} > limited_token_nums: {limited_token_nums}")
                    return new_docs
                else:
                    debug_logger.info(
                        f"first_doc_tokens: {first_doc_tokens} + second_limit_doc_tokens {doc_limit}: {second_doc_tokens} <= limited_token_nums: {limited_token_nums}")
                    new_docs.append(second_completed_doc_limit)
            else:
                debug_logger.info(
                    f"first_doc_tokens: {first_doc_tokens} + second_doc_tokens: {second_doc_tokens} <= limited_token_nums: {limited_token_nums}")
                new_docs.append(second_completed_doc)
        return new_docs

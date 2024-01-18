from qanything_kernel.configs.model_config import VECTOR_SEARCH_TOP_K, CHUNK_SIZE, VECTOR_SEARCH_SCORE_THRESHOLD, \
    PROMPT_TEMPLATE, STREAMING
from typing import List
from qanything_kernel.connector.embedding.embedding_for_online import YouDaoEmbeddings
from qanything_kernel.connector.embedding.embedding_for_local import YouDaoLocalEmbeddings
import time
from qanything_kernel.connector.llm import OpenAILLM, ZiyueLLM
from langchain.schema import Document
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from qanything_kernel.connector.database.milvus.milvus_client import MilvusClient
from .local_file import LocalFile
from qanything_kernel.utils.general_utils import get_time
import aiohttp
import requests
import traceback


def _embeddings_hash(self):
    return hash(self.model_name)


YouDaoLocalEmbeddings.__hash__ = _embeddings_hash
YouDaoEmbeddings.__hash__ = _embeddings_hash


class LocalDocQA:
    def __init__(self):
        self.llm: object = None
        self.embeddings: object = None
        self.top_k: int = VECTOR_SEARCH_TOP_K
        self.chunk_size: int = CHUNK_SIZE
        self.chunk_conent: bool = True
        self.score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD
        self.milvus_kbs: List[MilvusClient] = []
        self.milvus_summary: KnowledgeBaseManager = None
        self.mode: str = None
        self.local_rerank_service_url = "http://0.0.0.0:8776"
        self.ocr_url = 'http://0.0.0.0:8010/ocr'
        self.logger = None

    def print(self, *args):
        if self.logger:
            self.logger.info(*args)
        else:
            print(*args, flush=True)

    def error(self, *args):
        if self.logger:
            self.logger.error(*args)
        else:
            print(*args, flush=True)

    def warning(self, *args):
        if self.logger:
            self.logger.warning(*args)
        else:
            print(*args, flush=True)

    def get_ocr_result(self, image_data: dict):
        response = requests.post(self.ocr_url, json=image_data)
        response.raise_for_status()  # 如果请求返回了错误状态码，将会抛出异常
        return response.json()['results']

    def init_cfg(self, mode='local', logger=None):
        self.logger = logger
        self.mode = mode
        if self.mode == 'local':
            self.llm: ZiyueLLM = ZiyueLLM()
            self.embeddings = YouDaoLocalEmbeddings()
        else:
            self.llm: OpenAILLM = OpenAILLM()
            self.embeddings = YouDaoEmbeddings()
        self.milvus_summary = KnowledgeBaseManager(self.mode, self.logger)

    def create_milvus_collection(self, user_id, kb_id, kb_name):
        milvus_kb = MilvusClient(self.mode, user_id, [kb_id], self.logger)
        self.milvus_kbs.append(milvus_kb)
        self.milvus_summary.new_milvus_base(kb_id, user_id, kb_name)

    def match_milvus_kb(self, user_id, kb_ids):
        for kb in self.milvus_kbs:
            if user_id == kb.user_id and kb_ids == kb.kb_ids:
                self.print(f'match milvus_client: {kb}')
                return kb
        milvus_kb = MilvusClient(self.mode, user_id, kb_ids, self.logger)
        self.milvus_kbs.append(milvus_kb)
        return milvus_kb

    async def insert_files_to_milvus(self, user_id, kb_id, local_files: List[LocalFile]):
        self.print(f'insert_files_to_milvus: {kb_id}')
        milvus_kv = self.match_milvus_kb(user_id, [kb_id])
        assert milvus_kv is not None
        success_list = []
        failed_list = []

        for local_file in local_files:
            start = time.time()
            try:
                local_file.split_file_to_docs(self.get_ocr_result)
                content_length = sum([len(doc.page_content) for doc in local_file.docs])
            except Exception as e:
                error_info = f'split error: {traceback.format_exc()}'
                self.error(error_info)
                self.milvus_summary.update_file_status(local_file.file_id, status='red')
                failed_list.append(local_file)
                continue
            end = time.time()
            self.milvus_summary.update_content_length(local_file.file_id, content_length)
            self.print(f'split time: {end - start} {len(local_file.docs)}')
            start = time.time()
            try:
                local_file.create_embedding()
            except Exception as e:
                error_info = f'embedding error: {traceback.format_exc()}'
                self.error(error_info)
                self.milvus_summary.update_file_status(local_file.file_id, status='red')
                failed_list.append(local_file)
                continue
            end = time.time()
            self.print(f'embedding time: {end - start} {len(local_file.embs)}')

            self.milvus_summary.update_chunk_size(local_file.file_id, len(local_file.docs))
            ret = await milvus_kv.insert_files(local_file.file_id, local_file.file_name, local_file.file_path,
                                               local_file.docs, local_file.embs)
            insert_time = time.time()
            self.print(f'insert time: {insert_time - end}')
            if ret:
                self.milvus_summary.update_file_status(local_file.file_id, status='green')
                success_list.append(local_file)
            else:
                self.milvus_summary.update_file_status(local_file.file_id, status='yellow')
                failed_list.append(local_file)
        self.print(
            f"insert_to_milvus: success num: {len(success_list)}, failed num: {len(failed_list)}")

    def deduplicate_documents(self, source_docs):
        unique_docs = set()
        deduplicated_docs = []
        for doc in source_docs:
            if doc.page_content not in unique_docs:
                unique_docs.add(doc.page_content)
                deduplicated_docs.append(doc)
        return deduplicated_docs

    def get_source_documents(self, queries, milvus_kb, cosine_thresh=None, top_k=None):
        milvus_kb: MilvusClient
        if not top_k:
            top_k = self.top_k
        source_documents = []
        embs = self.embeddings._get_len_safe_embeddings(queries)
        t1 = time.time()
        batch_result = milvus_kb.search_emb_async(embs=embs, top_k=top_k)
        t2 = time.time()
        self.print(f"milvus search time: {t2 - t1}")
        for query, query_docs in zip(queries, batch_result):
            for doc in query_docs:
                doc.metadata['retrieval_query'] = query  # 添加查询到文档的元数据中
                doc.metadata['embed_version'] = self.embeddings.embed_version
                source_documents.append(doc)
        if cosine_thresh:
            source_documents = [item for item in source_documents if float(item.metadata['score']) > cosine_thresh]

        return source_documents

    def reprocess_source_documents(self, query: str,
                                   source_docs: List[Document],
                                   history: List[str],
                                   prompt_template: str) -> List[Document]:
        # 组装prompt,根据max_token
        query_token_num = self.llm.num_tokens_from_messages([query])
        history_token_num = self.llm.num_tokens_from_messages([x for sublist in history for x in sublist])
        template_token_num = self.llm.num_tokens_from_messages([prompt_template])
        limited_token_nums = self.llm.token_window - self.llm.max_token - self.llm.offcut_token - query_token_num - history_token_num - template_token_num
        new_source_docs = []
        total_token_num = 0
        for doc in source_docs:
            doc_token_num = self.llm.num_tokens_from_docs([doc])
            if total_token_num + doc_token_num <= limited_token_nums:
                new_source_docs.append(doc)
                total_token_num += doc_token_num
            else:
                remaining_token_num = limited_token_nums - total_token_num
                doc_content = doc.page_content
                doc_content_token_num = self.llm.num_tokens_from_messages([doc_content])
                while doc_content_token_num > remaining_token_num:
                    # Truncate the doc content to fit the remaining tokens
                    if len(doc_content) > 2 * self.llm.truncate_len:
                        doc_content = doc_content[self.llm.truncate_len: -self.llm.truncate_len]
                    else:  # 如果最后不够truncate_len长度的2倍，说明不够切了，直接赋值为空
                        doc_content = ""
                        break
                    doc_content_token_num = self.llm.num_tokens_from_messages([doc_content])
                doc.page_content = doc_content
                new_source_docs.append(doc)
                break

        self.print(f"limited token nums: {limited_token_nums}")
        self.print(f"template token nums: {template_token_num}")
        self.print(f"query token nums: {query_token_num}")
        self.print(f"history token nums: {history_token_num}")
        self.print(f"new_source_docs token nums: {self.llm.num_tokens_from_docs(new_source_docs)}")
        return new_source_docs

    def generate_prompt(self, query, source_docs, prompt_template):
        context = "\n".join([doc.page_content for doc in source_docs])
        prompt = prompt_template.replace("{question}", query).replace("{context}", context)
        return prompt

    def rerank_documents(self, query, source_documents):
        return self.rerank_documents_for_local(query, source_documents)

    def rerank_documents_for_local(self, query, source_documents):
        if len(query) > 300:  # tokens数量超过300时不使用local rerank
            return source_documents
        try:
            response = requests.post(f"{self.local_rerank_service_url}/rerank",
                                     json={"passages": [doc.page_content for doc in source_documents], "query": query})
            scores = response.json()
            for idx, score in enumerate(scores):
                source_documents[idx].metadata['score'] = score

            source_documents = sorted(source_documents, key=lambda x: x.metadata['score'], reverse=True)
        except Exception as e:
            self.error("rerank error: %s", traceback.format_exc())
            self.warning("rerank error, use origin retrieval docs")

        return source_documents

    @get_time
    def get_knowledge_based_answer(self, query, milvus_kb, chat_history=None, streaming: bool = STREAMING,
                                   rerank: bool = False):
        if chat_history is None:
            chat_history = []
        retrieval_queries = [query]

        source_documents = self.get_source_documents(retrieval_queries, milvus_kb)

        deduplicated_docs = self.deduplicate_documents(source_documents)
        retrieval_documents = sorted(deduplicated_docs, key=lambda x: x.metadata['score'], reverse=True)
        if rerank and len(retrieval_documents) > 1:
            self.print(f"use rerank, rerank docs num: {len(retrieval_documents)}")
            retrieval_documents = self.rerank_documents(query, retrieval_documents)

        source_documents = self.reprocess_source_documents(query=query,
                                                           source_docs=retrieval_documents,
                                                           history=chat_history,
                                                           prompt_template=PROMPT_TEMPLATE)
        prompt = self.generate_prompt(query=query,
                                      source_docs=source_documents,
                                      prompt_template=PROMPT_TEMPLATE)
        t1 = time.time()
        for answer_result in self.llm.generatorAnswer(prompt=prompt,
                                                      history=chat_history,
                                                      streaming=streaming):
            resp = answer_result.llm_output["answer"]
            prompt = answer_result.prompt
            history = answer_result.history
            history[-1][0] = query
            response = {"query": query,
                        "prompt": prompt,
                        "result": resp,
                        "retrieval_documents": retrieval_documents,
                        "source_documents": source_documents}
            yield response, history
        t2 = time.time()
        self.print(f"LLM time: {t2 - t1}")

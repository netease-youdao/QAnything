from qanything_kernel.configs.model_config import VECTOR_SEARCH_TOP_K, CHUNK_SIZE, VECTOR_SEARCH_SCORE_THRESHOLD, \
    PROMPT_TEMPLATE, STREAMING
from typing import List
from qanything_kernel.connector.embedding.embedding_for_online import YouDaoEmbeddings
from qanything_kernel.connector.embedding.embedding_for_local import YouDaoLocalEmbeddings
from langchain.schema import Document
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from qanything_kernel.connector.database.milvus.milvus_client import MilvusClient
from qanything_kernel.utils.custom_log import debug_logger, qa_logger
from .local_file import LocalFile
from qanything_kernel.utils.general_utils import get_time
import time
import requests
import traceback
import logging

logging.basicConfig(level=logging.INFO)

def _embeddings_hash(self):
    return hash(self.model_name)


YouDaoLocalEmbeddings.__hash__ = _embeddings_hash
YouDaoEmbeddings.__hash__ = _embeddings_hash


class LocalDocSearch:
    def __init__(self):
        # self.llm: object = None
        self.embeddings: object = None
        self.local_rerank_backend: object = None
        self.top_k: int = VECTOR_SEARCH_TOP_K
        self.chunk_size: int = CHUNK_SIZE
        self.chunk_conent: bool = True
        self.score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD
        self.milvus_kbs: List[MilvusClient] = []
        self.milvus_summary: KnowledgeBaseManager = None
        self.mode: str = None
        self.use_cpu: bool = True
        self.local_rerank_service_url = "http://0.0.0.0:8776"
        self.ocr_url = 'http://0.0.0.0:8010/ocr'

    def get_ocr_result(self, image_data: dict):
        response = requests.post(self.ocr_url, json=image_data, timeout=60)
        response.raise_for_status()  # 如果请求返回了错误状态码，将会抛出异常
        return response.json()['results']

    def init_cfg(self, args):
        self.mode = args.mode
        self.use_cpu = args.use_cpu
        if self.use_cpu:
            from qanything_kernel.connector.rerank.rerank_onnx_backend import RerankOnnxBackend
            from qanything_kernel.connector.embedding.embedding_onnx_backend import EmbeddingOnnxBackend
            self.local_rerank_backend: RerankOnnxBackend = RerankOnnxBackend(self.use_cpu)
            self.embeddings: EmbeddingOnnxBackend = EmbeddingOnnxBackend(self.use_cpu)
        else:
            self.embeddings = YouDaoLocalEmbeddings()
        self.milvus_summary = KnowledgeBaseManager(self.mode)

    def create_milvus_collection(self, user_id, kb_id, kb_name):
        milvus_kb = MilvusClient(self.mode, user_id, [kb_id])
        self.milvus_kbs.append(milvus_kb)
        self.milvus_summary.new_milvus_base(kb_id, user_id, kb_name)

    def match_milvus_kb(self, user_id, kb_ids):
        for kb in self.milvus_kbs:
            if user_id == kb.user_id and kb_ids == kb.kb_ids:
                debug_logger.info(f'match milvus_client: {kb}')
                return kb
        milvus_kb = MilvusClient(self.mode, user_id, kb_ids)
        self.milvus_kbs.append(milvus_kb)
        return milvus_kb

    async def insert_files_to_milvus(self, user_id, kb_id, local_files: List[LocalFile]):
        debug_logger.info(f'insert_files_to_milvus: {kb_id}')
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
                debug_logger.error(error_info)
                self.milvus_summary.update_file_status(local_file.file_id, status='red')
                failed_list.append(local_file)
                continue
            end = time.time()
            self.milvus_summary.update_content_length(local_file.file_id, content_length)
            debug_logger.info(f'split time: {end - start} {len(local_file.docs)}')
            start = time.time()
            try:
                local_file.create_embedding()
            except Exception as e:
                error_info = f'embedding error: {traceback.format_exc()}'
                debug_logger.error(error_info)
                self.milvus_summary.update_file_status(local_file.file_id, status='red')
                failed_list.append(local_file)
                continue
            end = time.time()
            debug_logger.info(f'embedding time: {end - start} {len(local_file.embs)}')

            self.milvus_summary.update_chunk_size(local_file.file_id, len(local_file.docs))
            ret = await milvus_kv.insert_files(local_file.file_id, local_file.file_name, local_file.file_path,
                                               local_file.docs, local_file.embs)
            insert_time = time.time()
            debug_logger.info(f'insert time: {insert_time - end}')
            if ret:
                self.milvus_summary.update_file_status(local_file.file_id, status='green')
                success_list.append(local_file)
            else:
                self.milvus_summary.update_file_status(local_file.file_id, status='yellow')
                failed_list.append(local_file)
        debug_logger.info(
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
        # embs = self.embeddings._get_len_safe_embeddings(queries)
        if self.use_cpu:
            embs = self.embeddings.aembed_documents(queries)
        else:
            embs = self.embeddings._get_len_safe_embeddings(queries)
            
        

        t1 = time.time()
        batch_result = milvus_kb.search_emb_async(embs=embs, top_k=top_k, queries=queries)
        t2 = time.time()
        debug_logger.info(f"milvus search time: {t2 - t1}")
        for query, query_docs in zip(queries, batch_result):
            for doc in query_docs:
                doc.metadata['retrieval_query'] = query  # 添加查询到文档的元数据中                
                if self.use_cpu:
                    doc.metadata['embed_version'] = self.embeddings.getModelVersion
                else:
                    doc.metadata['embed_version'] = self.embeddings.embed_version
                source_documents.append(doc)
        if cosine_thresh:
            source_documents = [item for item in source_documents if float(item.metadata['score']) > cosine_thresh]

        return source_documents

    
    def generate_prompt(self, query, source_docs, prompt_template):
        context = "\n".join([doc.page_content for doc in source_docs])
        prompt = prompt_template.replace("{question}", query).replace("{context}", context)
        return prompt

    def rerank_documents(self, query, source_documents):
        return self.rerank_documents_for_local(query, source_documents)

    def rerank_documents_for_local(self, query, source_documents):
        if len(query) > 300:  # tokens数量超过300时不使用local rerank
            return source_documents

        source_documents_reranked = []
        try:
            if self.use_cpu:
                response = self.local_rerank_backend.predict(query, [doc.page_content for doc in source_documents])
                debug_logger.info(f"rerank scores: {response}")
            else:
                response = requests.post(f"{self.local_rerank_service_url}/rerank",
                                         json={"passages": [doc.page_content for doc in source_documents], "query": query}, timeout=60)
                debug_logger.info(f"rerank scores: {response}")
            scores = response.json()
            for idx, score in enumerate(scores):
                source_documents[idx].metadata['score'] = score
                if score < 0.35 and len(source_documents_reranked) > 0:
                    continue
                source_documents_reranked.append(source_documents[idx])

            source_documents_reranked = sorted(source_documents_reranked, key=lambda x: x.metadata['score'], reverse=True)
        except Exception as e:
            debug_logger.error("rerank error: %s", traceback.format_exc())
            debug_logger.warning("rerank error, use origin retrieval docs")
            source_documents_reranked = sorted(source_documents, key=lambda x: x.metadata['score'], reverse=True)

        return source_documents_reranked

    
    def get_knowledge_based(self, query, milvus_kb, rerank: bool = False):
        
        retrieval_queries = [query]

        source_documents = self.get_source_documents(retrieval_queries, milvus_kb)

        deduplicated_docs = self.deduplicate_documents(source_documents)
        retrieval_documents = sorted(deduplicated_docs, key=lambda x: x.metadata['score'], reverse=True)
        if rerank and len(retrieval_documents) > 1:
            debug_logger.info(f"use rerank, rerank docs num: {len(retrieval_documents)}")
            retrieval_documents = self.rerank_documents(query, retrieval_documents)

        # source_documents = self.reprocess_source_documents(query=query,
        #                                                    source_docs=retrieval_documents,
        #                                                    history=chat_history,
        #                                                    prompt_template=PROMPT_TEMPLATE)
        
        # prompt = self.generate_prompt(query=query,
        #                               source_docs=source_documents,
        #                               prompt_template=PROMPT_TEMPLATE)
        # t1 = time.time()
        # for answer_result in self.llm.generatorAnswer(prompt=prompt,
        #                                               history=chat_history,
        #                                               streaming=streaming):
        #     resp = answer_result.llm_output["answer"]
        #     prompt = answer_result.prompt
        #     history = answer_result.history

        #     # logging.info(f"[debug] get_knowledge_based_answer history = {history}")
        #     history[-1][0] = query
        #     response = {"query": query,
        #                 "prompt": prompt,
        #                 "result": resp,
        #                 "retrieval_documents": retrieval_documents,
        #                 "source_documents": source_documents}
        #     yield response, history
        # t2 = time.time()
        # debug_logger.info(f"LLM time: {t2 - t1}")

        return retrieval_documents


    
        
        
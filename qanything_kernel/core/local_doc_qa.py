import platform

from qanything_kernel.configs.model_config import VECTOR_SEARCH_TOP_K, CHUNK_SIZE, VECTOR_SEARCH_SCORE_THRESHOLD, \
    PROMPT_TEMPLATE, STREAMING
from typing import List
from qanything_kernel.connector.embedding.embedding_for_local import YouDaoLocalEmbeddings
import time
from qanything_kernel.connector.llm import *
# from qanything_kernel.connector.llm import OpenAILLM, LlamaCPPCustomLLM, BigDLCustomLLM
from langchain.schema import Document
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from qanything_kernel.connector.database.faiss.faiss_cilent import FaissClient
import easyocr
from easyocr import Reader
from qanything_kernel.utils.custom_log import debug_logger, qa_logger
from .local_file import LocalFile
from qanything_kernel.utils.general_utils import get_time
import requests
import traceback
import logging
import base64
import numpy as np
import qanything_kernel.connector.gpuinfo.global_vars as global_vars

gpu_type = global_vars.get_gpu_type()
if gpu_type == 'nvidia':
    from qanything_kernel.connector.rerank.rerank_client_onnx import RerankONNXBackend
elif gpu_type == 'intel':
    from qanything_kernel.connector.rerank.rerank_client_torch_xpu import RerankTorchXPUBackend
elif gpu_type == 'metal':
    from qanything_kernel.connector.rerank.rerank_client_torch_mps import RerankTorchMPSBackend
else:
    raise NotImplementedError('Unsupported GPU type: {}'.format(gpu_type))

logging.basicConfig(level=logging.INFO)

def _embeddings_hash(self):
    return hash(self.model_name)


YouDaoLocalEmbeddings.__hash__ = _embeddings_hash


class LocalDocQA:
    def __init__(self):
        self.llm: object = None
        self.embeddings: object = None
        self.top_k: int = VECTOR_SEARCH_TOP_K
        self.chunk_size: int = CHUNK_SIZE
        self.chunk_conent: bool = True
        self.score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD
        self.faiss_client: FaissClient = None
        self.mysql_client: KnowledgeBaseManager = None
        self.local_rerank_backend: object = None
        self.ocr_reader: Reader = None
        self.mode: str = None

    def get_ocr_result(self, input: dict):
        img_file = input['img64']
        height = input['height']
        width = input['width']
        channels = input['channels']
        binary_data = base64.b64decode(img_file)
        img_array = np.frombuffer(binary_data, dtype=np.uint8).reshape((height, width, channels))
        ocr_res = self.ocr_reader.readtext(img_array, detail=0)
        res = [line.replace(' ', '') for line in ocr_res if line]
        return res

    def init_cfg(self, mode='local', args=None):
        self.mode = mode
        self.embeddings = YouDaoLocalEmbeddings()
        if self.mode == 'local':
            if gpu_type == 'nvidia':
                self.llm: OpenAICustomLLM = OpenAICustomLLM()
                self.local_rerank_backend = RerankONNXBackend()
            elif gpu_type == 'intel':
                self.llm: BigDLCustomLLM = BigDLCustomLLM()
                self.local_rerank_backend = RerankTorchXPUBackend()
            elif gpu_type == 'metal':
                self.llm: LlamaCPPCustomLLM = LlamaCPPCustomLLM()
                self.local_rerank_backend = RerankTorchMPSBackend()
            else:
                raise NotImplementedError('Unsupported platform')
        else:
            self.llm: OpenAILLM = OpenAILLM()
        self.mysql_client = KnowledgeBaseManager()
        self.ocr_reader = easyocr.Reader(['ch_sim', 'en'])
        self.faiss_client = FaissClient(self.mysql_client, self.embeddings.embedding_client)

    async def insert_files_to_faiss(self, user_id, kb_id, local_files: List[LocalFile]):
        debug_logger.info(f'insert_files_to_faiss: {kb_id}')
        success_list = []
        failed_list = []

        for local_file in local_files:
            start = time.time()
            try:
                local_file.split_file_to_docs(self.get_ocr_result)
                if local_file.file_name.endswith('.faq'):
                    content_length = len(local_file.docs[0].metadata['faq_dict']['question']) + len(
                        local_file.docs[0].metadata['faq_dict']['answer'])
                else:
                    content_length = sum([len(doc.page_content) for doc in local_file.docs])
            except Exception as e:
                error_info = f'split error: {traceback.format_exc()}'
                debug_logger.error(error_info)
                self.mysql_client.update_file_status(local_file.file_id, status='red')
                failed_list.append(local_file)
                continue
            end = time.time()
            self.mysql_client.update_content_length(local_file.file_id, content_length)
            debug_logger.info(f'split time: {end - start} {len(local_file.docs)}')
            self.mysql_client.update_chunk_size(local_file.file_id, len(local_file.docs))
            add_ids = await self.faiss_client.add_document(local_file.docs)
            insert_time = time.time()
            debug_logger.info(f'insert time: {insert_time - end}')
            self.mysql_client.update_file_status(local_file.file_id, status='green')
            success_list.append(local_file)
        debug_logger.info(
            f"insert_to_faiss: success num: {len(success_list)}, failed num: {len(failed_list)}")

    def deduplicate_documents(self, source_docs):
        unique_docs = set()
        deduplicated_docs = []
        for doc in source_docs:
            if doc.page_content not in unique_docs:
                unique_docs.add(doc.page_content)
                deduplicated_docs.append(doc)
        return deduplicated_docs

    async def get_source_documents(self, query, kb_ids, cosine_thresh=None, top_k=None):
        if not top_k:
            top_k = self.top_k
        source_documents = []
        t1 = time.time()
        filter = lambda metadata: metadata['kb_id'] in kb_ids
        # filter = None
        debug_logger.info(f"query: {query}")
        docs = await self.faiss_client.search(kb_ids, query, filter=filter, top_k=top_k)
        debug_logger.info(f"query_docs: {len(docs)}")
        t2 = time.time()
        debug_logger.info(f"faiss search time: {t2 - t1}")
        for idx, doc in enumerate(docs):
            if doc.metadata['file_name'].endswith('.faq'):
                faq_dict = doc.metadata['faq_dict']
                doc.page_content = f"{faq_dict['question']}：{faq_dict['answer']}"
                nos_keys = faq_dict.get('nos_keys')
                doc.metadata['nos_keys'] = nos_keys
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

        debug_logger.info(f"limited token nums: {limited_token_nums}")
        debug_logger.info(f"template token nums: {template_token_num}")
        debug_logger.info(f"query token nums: {query_token_num}")
        debug_logger.info(f"history token nums: {history_token_num}")
        debug_logger.info(f"new_source_docs token nums: {self.llm.num_tokens_from_docs(new_source_docs)}")
        return new_source_docs

    def generate_prompt(self, query, source_docs, prompt_template):
        context = "\n".join([doc.page_content for doc in source_docs])
        prompt = prompt_template.replace("{question}", query).replace("{context}", context)
        return prompt

    def rerank_documents(self, query, source_documents):
        if len(query) > 300:  # tokens数量超过300时不使用local rerank
            return source_documents

        scores = self.local_rerank_backend.predict(query, [doc.page_content for doc in source_documents])
        debug_logger.info(f"rerank scores: {scores}")
        for idx, score in enumerate(scores):
                source_documents[idx].metadata['score'] = score
        source_documents = sorted(source_documents, key=lambda x: x.metadata['score'], reverse=True)
        return source_documents

    async def get_knowledge_based_answer(self, custom_prompt, query, kb_ids, chat_history=None,
                                         streaming: bool = STREAMING,
                                         rerank: bool = False):
        if chat_history is None:
            chat_history = []

        source_documents = await self.get_source_documents(query, kb_ids)

        deduplicated_docs = self.deduplicate_documents(source_documents)
        retrieval_documents = sorted(deduplicated_docs, key=lambda x: x.metadata['score'], reverse=True)
        if rerank and len(retrieval_documents) > 1:
            debug_logger.info(f"use rerank, rerank docs num: {len(retrieval_documents)}")
            retrieval_documents = self.rerank_documents(query, retrieval_documents)[: self.rerank_top_k]

        if custom_prompt is None:
            prompt_template = PROMPT_TEMPLATE
        else:
            prompt_template = custom_prompt + '\n' + PROMPT_TEMPLATE

        source_documents = self.reprocess_source_documents(query=query,
                                                           source_docs=retrieval_documents,
                                                           history=chat_history,
                                                           prompt_template=prompt_template)
        prompt = self.generate_prompt(query=query,
                                      source_docs=source_documents,
                                      prompt_template=prompt_template)
        t1 = time.time()
        async for answer_result in self.llm.generatorAnswer(prompt=prompt,
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
        debug_logger.info(f"LLM time: {t2 - t1}")
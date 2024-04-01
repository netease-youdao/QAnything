from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from qanything_kernel.configs.model_config import VECTOR_SEARCH_TOP_K, FAISS_LOCATION, FAISS_CACHE_SIZE
from typing import Optional, Union, Callable, Dict, Any, List, Tuple
from langchain_community.vectorstores.faiss import dependable_faiss_import
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from functools import lru_cache
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 可能是由于是MacOS系统的原因


@lru_cache(FAISS_CACHE_SIZE)
def load_vector_store(faiss_index_path, embeddings):
    return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)


class FaissClient:
    def __init__(self, mysql_client: KnowledgeBaseManager, embeddings):
        self.mysql_client: KnowledgeBaseManager = mysql_client
        self.embeddings = embeddings
        self.faiss_client: FAISS = None

    def _init(self, kb_id):
        faiss_index_path = os.path.join(FAISS_LOCATION, kb_id, 'faiss_index')
        if os.path.exists(faiss_index_path):
            self.faiss_client: FAISS = load_vector_store(faiss_index_path, self.embeddings)
        else:
            faiss = dependable_faiss_import()
            index = faiss.IndexFlatL2(768)
            docstore = InMemoryDocstore()
            self.faiss_client: FAISS = FAISS(self.embeddings, index, docstore, index_to_docstore_id={})

    async def search(self, kb_ids, query, filter: Optional[Union[Callable, Dict[str, Any]]] = None,
                     top_k=VECTOR_SEARCH_TOP_K):
        if self.faiss_client is None:
            self._init(kb_ids[0])  # TODO 多知识库暂未支持
        # filter = {'page': 1}
        if filter is None:
            filter = {}
        docs_with_score = await self.faiss_client.asimilarity_search_with_score(query, k=top_k, filter=filter,
                                                                                fetch_k=200)
        for doc, score in docs_with_score:
            doc.metadata['score'] = score
        docs = [doc for doc, score in docs_with_score]
        docs_with_score = self.merge_docs(docs)
        return docs_with_score

    def merge_docs(self, docs):
        # 把docs按照file_id进行合并，但是需要对所有file_id相同的doc根据chunk_id先排序，chunk_id相邻的doc合并
        merged_docs = []
        docs = sorted(docs, key=lambda x: (x.metadata['file_id'], x.metadata['chunk_id']))
        for doc in docs:
            if not merged_docs or merged_docs[-1].metadata['file_id'] != doc.metadata['file_id']:
                merged_docs.append(doc)
            else:
                if merged_docs[-1].metadata['chunk_id'] == doc.metadata['chunk_id'] - 1:
                    print('MERGE:', merged_docs[-1].metadata['chunk_id'], doc.metadata['chunk_id'])
                    merged_docs[-1].page_content += doc.page_content
                    merged_docs[-1].metadata['chunk_id'] = doc.metadata['chunk_id']
                else:
                    print('NOT MERGE:', merged_docs[-1].metadata['chunk_id'], doc.metadata['chunk_id'])
                    merged_docs.append(doc)
        return merged_docs

    async def add_document(self, docs):
        kb_id = docs[0].metadata['kb_id']
        if self.faiss_client is None:
            self._init(kb_id)
        add_ids = await self.faiss_client.aadd_documents(docs)
        # doc带上id存入Document表中
        chunk_id = 0
        for doc, add_id in zip(docs, add_ids):
            self.mysql_client.add_document(add_id, chunk_id, doc.metadata['file_id'], doc.metadata['file_name'],
                                           doc.metadata['kb_id'])
            chunk_id += 1
        debug_logger.info(f'add documents number: {len(add_ids)}')
        faiss_index_path = os.path.join(FAISS_LOCATION, kb_id, 'faiss_index')
        self.faiss_client.save_local(faiss_index_path)
        return add_ids

    def delete_documents(self, kb_id, file_ids=None):
        doc_ids = []
        if file_ids is None:
            doc_ids = self.mysql_client.get_documents_by_kb_id(kb_id)
        elif file_ids:
            doc_ids = self.mysql_client.get_documents_by_file_ids(file_ids)
        doc_ids = [doc_id[0] for doc_id in doc_ids]
        if not doc_ids:
            debug_logger.info(f'no documents to delete')
            return
        if self.faiss_client is None:
            self._init(kb_id)
        res = self.faiss_client.delete(doc_ids)
        debug_logger.info(f'delete documents: {res}')

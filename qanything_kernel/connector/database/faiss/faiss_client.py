from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from qanything_kernel.configs.model_config import VECTOR_SEARCH_TOP_K, FAISS_INDEX_FILE_PATH
from typing import Optional, Union, Callable, Dict, Any, List, Tuple
from langchain_community.vectorstores.faiss import dependable_faiss_import
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 可能是由于是MacOS系统的原因


class FaissClient:
    def __init__(self, mysql_client: KnowledgeBaseManager, embeddings):
        self.mysql_client: KnowledgeBaseManager = mysql_client
        if os.path.exists(FAISS_INDEX_FILE_PATH):
            self.faiss_client: FAISS = FAISS.load_local(FAISS_INDEX_FILE_PATH, embeddings,
                                                        allow_dangerous_deserialization=True)
        else:
            faiss = dependable_faiss_import()
            index = faiss.IndexFlatL2(768)
            docstore = InMemoryDocstore()
            self.faiss_client: FAISS = FAISS(embeddings, index, docstore, index_to_docstore_id={})
            # self.faiss_client.save_local(FAISS_INDEX_FILE_PATH)

    async def search(self, query, filter: Optional[Union[Callable, Dict[str, Any]]] = None, top_k=VECTOR_SEARCH_TOP_K):
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
        add_ids = await self.faiss_client.aadd_documents(docs)
        # doc带上id存入Document表中
        chunk_id = 0
        for doc, add_id in zip(docs, add_ids):
            self.mysql_client.add_document(add_id, chunk_id, doc.metadata['file_id'], doc.metadata['file_name'], doc.metadata['kb_id'])
            chunk_id += 1
        debug_logger.info(f'add documents number: {len(add_ids)}')
        self.faiss_client.save_local(FAISS_INDEX_FILE_PATH)
        return add_ids

    def delete_documents(self, kb_id=None, file_ids=None):
        doc_ids = []
        if kb_id:
            doc_ids = self.mysql_client.get_documents_by_kb_id(kb_id)
        elif file_ids:
            doc_ids = self.mysql_client.get_documents_by_file_ids(file_ids)
        doc_ids = [doc_id[0] for doc_id in doc_ids]
        if not doc_ids:
            debug_logger.info(f'no documents to delete')
            return
        res = self.faiss_client.delete(doc_ids)
        debug_logger.info(f'delete documents: {res}')

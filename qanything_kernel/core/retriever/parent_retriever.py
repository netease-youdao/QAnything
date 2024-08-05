from langchain.retrievers import ParentDocumentRetriever
from qanything_kernel.core.retriever.vectorstore import VectorStoreMilvusClient
from qanything_kernel.core.retriever.elasticsearchstore import StoreElasticSearchClient
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from qanything_kernel.core.retriever.docstrore import MysqlStore
from qanything_kernel.configs.model_config import VECTOR_SEARCH_TOP_K, ES_TOP_K, DEFAULT_CHILD_CHUNK_SIZE, DEFAULT_PARENT_CHUNK_SIZE
from qanything_kernel.utils.custom_log import debug_logger, insert_logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qanything_kernel.utils.general_utils import num_tokens, get_time_async
import copy
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
)
from langchain_community.vectorstores.milvus import Milvus
from langchain_elasticsearch import ElasticsearchStore
import time
import traceback


class SelfParentRetriever(ParentDocumentRetriever):
    def set_search_kwargs(self, search_type, **kwargs):
        self.search_type = search_type
        self.search_kwargs = kwargs
        debug_logger.info(f"Set search kwargs: {self.search_kwargs}")

    async def _aget_relevant_documents(
            self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        debug_logger.info(f"Search: query: {query}, {self.search_type} with {self.search_kwargs}")
        if self.search_type == "mmr":
            sub_docs = await self.vectorstore.amax_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            sub_docs = await self.vectorstore.asimilarity_search(
                query, **self.search_kwargs
            )

        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = await self.docstore.amget(ids)
        res = [d for d in docs if d is not None]
        sub_docs_lengths = [len(d.page_content) for d in sub_docs]
        res_lengths = [len(d.page_content) for d in res]
        debug_logger.info(
            f"Got child docs: {len(sub_docs)}, {sub_docs_lengths} and Parent docs: {len(res)}, {res_lengths}")
        return res

    async def aadd_documents(
            self,
            documents: List[Document],
            ids: Optional[List[str]] = None,
            add_to_docstore: bool = True,
            backup_vectorstore: Optional[Milvus] = None,
            es_store: Optional[ElasticsearchStore] = None,
            single_parent: bool = False,
    ) -> int:
        insert_logger.info(f"Inserting {len(documents)} complete documents, single_parent: {single_parent}")
        if self.parent_splitter is not None and not single_parent:
            documents = self.parent_splitter.split_documents(documents)
        insert_logger.info(f"Inserting {len(documents)} parent documents")
        if ids is None:
            file_id = documents[0].metadata['file_id']
            doc_ids = [file_id + '_' + str(i) for i, _ in enumerate(documents)]
            if not add_to_docstore:
                raise ValueError(
                    "If ids are not passed in, `add_to_docstore` MUST be True"
                )
        else:
            if len(documents) != len(ids):
                raise ValueError(
                    "Got uneven list of documents and ids. "
                    "If `ids` is provided, should be same length as `documents`."
                )
            doc_ids = ids

        docs = []
        full_docs = []
        for i, doc in enumerate(documents):
            _id = doc_ids[i]
            sub_docs = self.child_splitter.split_documents([doc])
            if self.child_metadata_fields is not None:
                for _doc in sub_docs:
                    _doc.metadata = {
                        k: _doc.metadata[k] for k in self.child_metadata_fields
                    }
            for _doc in sub_docs:
                _doc.metadata[self.id_key] = _id
            docs.extend(sub_docs)
            full_docs.append((_id, doc))
        insert_logger.info(f"Inserting {len(docs)} child documents, metadata: {docs[0].metadata}")
        # if backup_vectorstore is not None:
        #     res = await self.vectorstore.aadd_documents(docs, ids=doc_ids)
        # else:
        #     res = await self.vectorstore.aadd_documents(docs)
        res = await self.vectorstore.aadd_documents(docs)
        insert_logger.info(f'vectorstore insert number: {len(res)}, {res[0]}')
        if backup_vectorstore is not None:
            backup_res = await backup_vectorstore.aadd_documents(docs)
            insert_logger.info(
                f'backup vectorstore insert number: {len(backup_res)}, {backup_res[0]}')
        if es_store is not None:
            try:
                # docs的doc_id是file_id + '_' + i
                docs_ids = [doc.metadata['file_id'] + '_' + str(i) for i, doc in enumerate(docs)]
                es_res = await es_store.aadd_documents(docs, ids=docs_ids)
                insert_logger.info(f'es_store insert number: {len(es_res)}, {es_res[0]}')
            except Exception as e:
                insert_logger.error(f"Error in aadd_documents on es_store: {traceback.format_exc()}")

        if add_to_docstore:
            await self.docstore.amset(full_docs)
        return len(res)


class ParentRetriever:
    def __init__(self, vectorstore_client: VectorStoreMilvusClient, mysql_client: KnowledgeBaseManager, es_client: StoreElasticSearchClient):
        self.mysql_client = mysql_client
        # This text splitter is used to create the parent documents
        # parent_splitter = RecursiveCharacterTextSplitter(
        #     separators=["\n\n", "\n", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " ", ""],
        #     chunk_size=DEFAULT_PARENT_CHUNK_SIZE,
        #     chunk_overlap=0,
        #     length_function=num_tokens)
        # # This text splitter is used to create the child documents
        # # It should create documents smaller than the parent
        # child_splitter = RecursiveCharacterTextSplitter(
        #     separators=["\n\n", "\n", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " ", ""],
        #     chunk_size=DEFAULT_CHILD_CHUNK_SIZE,
        #     chunk_overlap=int(DEFAULT_CHILD_CHUNK_SIZE / 4),
        #     length_function=num_tokens)
        self.retriever = SelfParentRetriever(
            vectorstore=vectorstore_client.local_vectorstore,
            docstore=MysqlStore(mysql_client),
            child_splitter=None,
            parent_splitter=None,
        )
        self.backup_vectorstore: Optional[Milvus] = None
        self.es_store = es_client.es_store

    @get_time_async
    async def insert_documents(self, docs, parent_chunk_size, single_parent=False):
        self.retriever.parent_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " ", ""],
            chunk_size=parent_chunk_size,
            chunk_overlap=0,
            length_function=num_tokens)
        child_chunk_size = min(DEFAULT_CHILD_CHUNK_SIZE, int(parent_chunk_size / 2))
        self.retriever.child_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " ", ""],
            chunk_size=child_chunk_size,
            chunk_overlap=int(child_chunk_size / 4),
            length_function=num_tokens)

        # insert_logger.info(f'insert documents: {len(docs)}')
        embed_docs = copy.deepcopy(docs)
        # 补充metadata信息
        for idx, doc in enumerate(embed_docs):
            if 'kb_id' not in doc.metadata:
                kb_name = self.mysql_client.get_knowledge_base_name([doc.metadata['kb_id']])
                kb_name = kb_name[0][2]
                metadata_infos = f"知识库名: {kb_name}\n"
                if 'file_name' in doc.metadata:
                    metadata_infos += f"文件名: {doc.metadata['file_name']}\n"
                doc.page_content = metadata_infos + '文件内容如下: \n' + doc.page_content
            if 'title_lst' in doc.metadata:
                del doc.metadata['title_lst']
            if 'has_table' in doc.metadata:
                del doc.metadata['has_table']
            if 'images_number' in doc.metadata:
                del doc.metadata['images_number']
        ids = None if not single_parent else [doc.metadata['doc_id'] for doc in embed_docs]
        return await self.retriever.aadd_documents(embed_docs, backup_vectorstore=self.backup_vectorstore,
                                                   es_store=self.es_store, ids=ids, single_parent=single_parent)

    async def get_retrieved_documents(self, query: str, partition_keys: List[str], time_record: dict, hybrid_search: bool):
        milvus_start_time = time.perf_counter()
        expr = f'kb_id in {partition_keys}'
        # self.retriever.set_search_kwargs("mmr", k=VECTOR_SEARCH_TOP_K, expr=expr)
        self.retriever.set_search_kwargs("similarity", k=VECTOR_SEARCH_TOP_K, expr=expr)
        query_docs = await self.retriever.aget_relevant_documents(query)
        for doc in query_docs:
            doc.metadata['retrieval_source'] = 'milvus'
        milvus_end_time = time.perf_counter()
        time_record['retriever_search_by_milvus'] = round(milvus_end_time - milvus_start_time, 2)

        if not hybrid_search:
            return query_docs

        try:
            # filter = []
            # for partition_key in partition_keys:
            filter = [{"terms": {"metadata.kb_id.keyword": partition_keys}}]
            es_sub_docs = await self.es_store.asimilarity_search(query, k=ES_TOP_K, filter=filter)
            es_ids = []
            milvus_doc_ids = [d.metadata[self.retriever.id_key] for d in query_docs]
            for d in es_sub_docs:
                if self.retriever.id_key in d.metadata and d.metadata[self.retriever.id_key] not in es_ids and d.metadata[self.retriever.id_key] not in milvus_doc_ids:
                    es_ids.append(d.metadata[self.retriever.id_key])
            es_docs = await self.retriever.docstore.amget(es_ids)
            es_docs = [d for d in es_docs if d is not None]
            for doc in es_docs:
                doc.metadata['retrieval_source'] = 'es'
            time_record['retriever_search_by_es'] = round(time.perf_counter() - milvus_end_time, 2)
            debug_logger.info(f"Got {len(query_docs)} documents from vectorstore and {len(es_sub_docs)} documents from es, total {len(query_docs) + len(es_docs)} merged documents.")
            query_docs.extend(es_docs)
        except Exception as e:
            debug_logger.error(f"Error in get_retrieved_documents on es_search: {e}")
        return query_docs

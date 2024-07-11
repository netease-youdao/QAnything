from qanything_kernel.utils.custom_log import debug_logger, insert_logger
from qanything_kernel.configs.model_config import ES_USER, ES_PASSWORD, ES_URL, ES_INDEX_NAME
from langchain_elasticsearch import ElasticsearchStore


class StoreElasticSearchClient:
    def __init__(self, index_name):
        self.es_store = ElasticsearchStore(
            es_url=ES_URL,
            index_name=index_name.lower(),
            es_user=ES_USER,
            es_password=ES_PASSWORD,
            strategy=ElasticsearchStore.BM25RetrievalStrategy()
        )
        debug_logger.info(f"Init ElasticSearchStore with index_name: {index_name.lower()}")

    def delete(self, docs_ids):
        try:
            res = self.es_store.delete(docs_ids, timeout=60)
            debug_logger.info(f"Delete ES document with number: {len(docs_ids)}, {docs_ids[0]}, res: {res}")
        except Exception as e:
            debug_logger.error(f"Delete ES document failed with error: {e}")

    def delete_files(self, file_ids, file_chunks):
        docs_ids = []
        for file_id, file_chunk in zip(file_ids, file_chunks):
            # doc_id 是file_id + '_' + i，其中i是range(file_chunk)
            docs_ids.extend([file_id + '_' + str(i) for i in range(file_chunk)])
        if docs_ids:
            self.delete(docs_ids)

from qanything_kernel.utils.custom_log import insert_logger
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from qanything_kernel.configs.model_config import UPLOAD_ROOT_PATH
from qanything_kernel.utils.custom_log import debug_logger
from langchain_core.documents import Document
from langchain.storage import InMemoryStore
from typing import (
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar
)
import os
import json
from tqdm import tqdm


V = TypeVar("V")


class MysqlStore(InMemoryStore):
    def __init__(self, mysql_client: KnowledgeBaseManager):
        self.mysql_client = mysql_client
        super().__init__()


    def mset(self, key_value_pairs: Sequence[Tuple[str, V]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs (Sequence[Tuple[str, V]]): A sequence of key-value pairs.

        Returns:
            None
        """
        doc_ids = [doc_id for doc_id, _ in key_value_pairs]
        insert_logger.info(f"add documents: {len(doc_ids)}")
        for doc_id, doc in tqdm(key_value_pairs):
            doc_json = doc.to_json()
            if doc_json['kwargs'].get('metadata') is None:
                doc_json['kwargs']['metadata'] = doc.metadata
            self.mysql_client.add_document(doc_id, doc_json)

    def mget(self, keys: Sequence[str]) -> List[Optional[V]]:
        """Get the values associated with the given keys.

        Args:
            keys (Sequence[str]): A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
       
        docs = []
        for doc_id in keys:
            doc_json = self.mysql_client.get_document_by_doc_id(doc_id)
            if doc_json is None:
                docs.append(None)
                continue
            # debug_logger.info(f'doc_id: {doc_id} get doc_json: {doc_json}')
            user_id, file_id, file_name, kb_id = doc_json['kwargs']['metadata']['user_id'], doc_json['kwargs']['metadata']['file_id'], doc_json['kwargs']['metadata']['file_name'], doc_json['kwargs']['metadata']['kb_id'] 
            doc_idx = doc_id.split('_')[-1]
            upload_path = os.path.join(UPLOAD_ROOT_PATH, user_id)
            local_path = os.path.join(upload_path, kb_id, file_id, file_name.rsplit('.', 1)[0] + '_' + doc_idx + '.json')
            doc = Document(page_content=doc_json['kwargs']['page_content'], metadata=doc_json['kwargs']['metadata'])
            doc.metadata['doc_id'] = doc_id
            if file_name.endswith('.faq'):
                faq_dict = doc.metadata['faq_dict']
                page_content = f"{faq_dict['question']}：{faq_dict['answer']}"
                nos_keys = faq_dict.get('nos_keys')
                doc.page_content = page_content
                doc.metadata['nos_keys'] = nos_keys
            docs.append(doc)
            if not os.path.exists(local_path):
                #  json字符串写入本地文件
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                # debug_logger.info(f'write local_path: {local_path}')
                with open(local_path, 'w') as f:
                    f.write(json.dumps(doc_json, ensure_ascii=False))
        return docs

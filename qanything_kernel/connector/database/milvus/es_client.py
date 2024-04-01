'''
@Description: 
@Author: shenlei
@Date: 2024-03-20 14:16:48
@LastEditTime: 2024-04-01 15:04:05
@LastEditors: shenlei
'''
import numpy as np
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from copy import deepcopy
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.configs.model_config import ES_HOST_LOCAL, ES_CLOUD_ID, ES_USER, ES_PASSWORD, ES_API_KEY, ES_CONNECT_PARAMS, ES_SIGNATURE, ES_BM25_SEARCH_SIZE

try:
    from elasticsearch import Elasticsearch, helpers
except ImportError:
    raise ImportError(
            "Could not import elasticsearch python package. "
            "Please install it with `pip install elasticsearch`."
        )


class ElasticsearchClient:
    """ Elasticsearch client, supporting bm25 retrieval.
    Args:
        index_name: index of the Elasticsearch client. In Qanything, we design index_name to "{user_id}_{kb_id}".
        ...: connection setup to es client.
    """
    def __init__(self, 
        index_name: List[str] = None, 
        url=ES_HOST_LOCAL, cloud_id=ES_CLOUD_ID, 
        user=ES_USER, password=ES_PASSWORD, 
        api_key=ES_API_KEY, 
        connect_params: Dict[str, Any]=ES_CONNECT_PARAMS
        ):

        if index_name is None:
            raise ValueError("Please provide `index_name` in str or list.")
        
        self.index_name = [index.lower() for index in index_name]

        # connect to es
        if url and cloud_id:
            raise ValueError("Both es_url and cloud_id are defined. Please provide only one.")

        connects = deepcopy(connect_params) if connect_params is not None else {}

        if url:
            connects['hosts'] = url
        elif cloud_id:
            connects['cloud_id'] = cloud_id
        else:
            raise ValueError("Please provide either elasticsearch url or cloud_id.")
        
        if api_key:
            connects['api_key'] = api_key
        elif user and password:
            connects['basic_auth'] = (user, password)
        
        self.client = Elasticsearch(**connects, headers={"user-agent": ES_SIGNATURE})

        try:
            debug_logger.info(f"##ES## - success to connect to {self.client.info().body}")
        except Exception as e:
            raise RuntimeError(f"Elasticsearch client initialization failed: {e}\nConnection setup: {connects}")
    
    def _create_index(self):
        for index_name in self.index_name:
            index_name = index_name.lower()
            if self.client.indices.exists(index=index_name):
                debug_logger.info(f"##ES## - Index {index_name} already exists. Skipping creation.")
            else:
                settings = {
                    "index": {
                        "similarity": {
                            "custom_bm25": {
                                "type": "BM25",
                                "k1": "1.3",
                                "b": "0.6"
                            }
                        }
                    }
                }
                mappings = {
                    "properties": {
                        'file_id': {
                            'type': 'keyword',
                            'index': True
                        },
                        "content": {
                            "type": "text",
                            "similarity": "custom_bm25",
                            "index": True,
                            "analyzer": "ik_smart",
                            "search_analyzer": "ik_smart",
                        }
                    }
                }
                debug_logger.info(
                    f"##ES## - Creating index {index_name} with:\nmappings: {mappings}\nsettings: {settings}"
                )
                self.client.indices.create(index=index_name, mappings=mappings, settings=settings)

    async def insert(self, data, refresh=False):
        self._create_index()
        ids = [item['metadata']["chunk_id"] for item in data]
        for index_name in self.index_name:
            index_name = index_name.lower()
            actions = []
            for item in data:
                action = {
                        "_op_type": "index",
                        "_id": item['metadata']["chunk_id"]
                    }
                action.update(item)
                actions.append(action)
            
            try:
                documents_written, errors = helpers.bulk(
                    client=self.client,
                    actions=actions,
                    refresh=False,
                    index=index_name,
                    stats_only=True,
                    raise_on_error=False,
                )
                debug_logger.info(f"##ES## - success to add: {documents_written}\nfail to add to index: {errors}")
                if refresh:
                    self.client.indices.refresh(index=index_name)
                    debug_logger.info(f"##ES## - finish insert chunks!")
            except Exception as e:
                return f"Error adding texts: {e}"

        return f"success to add chunks: {ids[:5]} ... in index: {self.index_name[:5]} ..."

    def search(self, queries, field='content'):
        valid_index_name = []
        for index_name in self.index_name:
            index_name = index_name.lower()
            if self.client.indices.exists(index=index_name):
                valid_index_name.append(index_name)
            else:
                debug_logger.info(f"##ES## - index: {index_name} is empty!")
        if len(valid_index_name) == 0:
            return []
        
        fields = ['file_id', 'content', 'metadata']
        search_results = []
        search_item_seen = set()
        for query in queries:
            if field == 'content':
                query_body = {
                    "query": {
                        "match": {
                            'content': {
                                "query": query,
                                "fuzziness": "AUTO",
                            }
                        }
                    },
                    "size": ES_BM25_SEARCH_SIZE
                }
            elif field == 'file_id':
                query_body = {
                    "query": {
                        "term": {
                            'file_id': query
                        }
                    }
                }
            else:
                raise ValueError(f"##ES## - Please provide valid field: {field}")

            response = self.client.search(
                index=valid_index_name,
                **query_body,
                source=fields,
            )
            
            for hit in response["hits"]["hits"]:
                search_tag = f"{hit['_index']}_{hit['_id']}"
                if search_tag in search_item_seen:
                    continue
                search_item_seen.add(search_tag)

                score = (1-float(1/(1 + np.exp(-hit['_score']/8))))*1.414
                search_item = {'index': hit['_index'], 'id': hit['_id'], 'score': score}
                
                for f in fields:
                    search_item[f] = hit["_source"][f]
                search_results.append(search_item)

        search_results = sorted(search_results, key=lambda x: x['score'])

        debug_logger.info(f"##ES## - success to search: field - {field}\nqueries: {queries}\nsearch num: {len(search_results)}")

        return search_results

    def delete_index(self, index_name):
        index_name = [index.lower() for index in index_name]
        try:
            self.client.indices.delete(index=index_name, ignore_unavailable=True)
            debug_logger.info(f"##ES## - success to delete index: {index_name}")
        except Exception as e:
            debug_logger.error(f"##ES## - fail to delete: {index_name}\nERROR: {e}")
    
    def delete_chunks(self, index_name=None, ids=None):
        if index_name is None or ids is None:
            return "No chunks to delete."

        index_name = index_name.lower()
        try:
            helpers.bulk(
                client=self.client,
                actions=({"_op_type": "delete", "_id": id_} for id_ in ids),
                refresh="wait_for",
                index=index_name,
                stats_only=True,
                raise_on_error=False,
                ignore_status=404
            )
            debug_logger.info(f"##ES## - success to delete chunks ids: {ids}\nfrom index: {index_name}")
        except Exception as e:
            debug_logger.error(f"Error delete chunks: {e}")
        
        debug_logger.info(f"success to delete chunks: {ids} in index: {index_name}")

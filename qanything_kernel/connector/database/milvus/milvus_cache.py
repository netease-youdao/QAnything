from collections import OrderedDict
from pymilvus import (
    connections,
    Collection,
    utility,
)
from pymilvus.client.types import LoadState
from qanything_kernel.utils.custom_log import debug_logger 
from qanything_kernel.utils.general_utils import get_time
from qanything_kernel.configs.model_config import MILVUS_HOST_ONLINE, MILVUS_PORT
from tqdm import tqdm

class MilvusLRUCache:
    def __init__(self, capacity: int):
        self.host = MILVUS_HOST_ONLINE
        self.port = MILVUS_PORT
        self.cache = OrderedDict()
        connections.connect(host=self.host, port=self.port)
        self.capacity = capacity
        self.update_cache()
        self.init_clear()

    @get_time
    def update_cache(self):
        # connections.connect(host=self.host, port=self.port)
        user_ids = utility.list_collections()
        loaded = []
        for user_id in tqdm(user_ids):
            if utility.load_state(user_id) == LoadState.Loaded:
                loaded.append(user_id)
                collection = Collection(name=user_id)
                self.cache[user_id] = collection
                if len(self.cache) > self.capacity * 0.7:
                    break
        debug_logger.info(f"Update Cache! Loaded collections number: {len(loaded)}")
        # connections.disconnect('default')
    
    def init_clear(self):
        while len(self.cache) >= self.capacity:
            debug_logger.info(f"init clear, current cache size: {len(self.cache)}")
            self.evict()

    def get(self, collection_name: str):
        if collection_name not in self.cache:
            return None
        now_state = utility.load_state(collection_name)
        if now_state != LoadState.Loaded:
            # self.update_cache()
            debug_logger.warning(f"{collection_name} not loaded: {now_state}, remove from cache")
            self.cache.pop(collection_name)
            return None
        # 移动到最末尾表示最近使用
        self.cache.move_to_end(collection_name)
        self.cache[collection_name].load(_async=True)  # 防止被其他workers释放了
        return self.cache[collection_name]

    def put(self, collection_name: str, collection, _async=True):
        if len(self.cache) >= self.capacity:
            # LRU策略释放
            self.evict()
        # 添加新的Collection
        self.cache[collection_name] = collection
        self.cache.move_to_end(collection_name)
        debug_logger.info(f"load collection: {collection_name}, async: {_async}")
        collection.load(_async=_async)

    def remove(self, collection_name: str):
        if collection_name in self.cache:
            collection = self.cache.pop(collection_name)
            collection.release()  # 释放资源
        else:
            sess = Collection(name=collection_name) 
            sess.release()  # 防止在其他进程里load了

    def evict(self):
        # 弹出第一个item
        _, collection = self.cache.popitem(last=False)
        debug_logger.info(f"evict collection: {collection.name}")
        collection.release()  # 释放资源

    def clear(self):
        for collection in self.cache.values():
            collection.release()  # 释放资源
        self.cache.clear()
        connections.disconnect('default')

import faiss
import numpy as np
import os
from typing import List
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.configs.model_config import FAISS_INDEX_FILE_PATH, FAISS_NLIST, FAISS_QUANTIZER, FAISS_DIMENSION


class FaissClient:
    def __init__(self, nlist, quantizer=None, index_file_path=None):
        self.dimension = 768
        self.index_file_path = index_file_path or 'faiss_index.idx'
        self.deleted_ids = set()

        if quantizer is None:
            self.quantizer = faiss.IndexFlatL2(self.dimension)
        else:
            self.quantizer = quantizer

        self.index = faiss.IndexIVFFlat(self.quantizer, self.dimension, nlist, faiss.METRIC_L2)
        self.is_trained = False
        self.id_to_index = {}  # 映射向量ID到FAISS索引ID
        self.index_to_id = []  # 从FAISS索引ID映射回向量ID

        if os.path.exists(self.index_file_path):
            self.load_index(self.index_file_path)

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        if not self.is_trained:
            self.index.train(vectors)
            self.is_trained = True
        self.index.add(vectors)
        for i, vid in enumerate(ids):
            idx = len(self.index_to_id)
            self.id_to_index[vid] = idx
            self.index_to_id.append(vid)

    def search_vectors(self, query_vectors: np.ndarray, top_k: int):
        distances, indices = self.index.search(query_vectors, top_k)
        # 将FAISS索引ID映射回向量ID
        return [(self.index_to_id[idx], dist) for dist, idx in zip(distances[0], indices[0]) if
                idx not in self.deleted_ids]

    def remove_vectors(self, ids: List[int]):
        for vid in ids:
            self.deleted_ids.add(self.id_to_index[vid])

    def rebuild_index(self):
        # 从索引中删除标记为删除的向量
        surviving_ids = [vid for vid in self.index_to_id if vid not in self.deleted_ids]
        surviving_vectors = self.index.reconstruct_n(0, len(self.index_to_id))
        surviving_vectors = np.array([surviving_vectors[i] for i in range(len(surviving_vectors)) if
                                      self.index_to_id[i] not in self.deleted_ids])

        # 创建新的索引并添加幸存的向量
        new_index = faiss.IndexIVFFlat(self.quantizer, self.dimension, self.index.nlist, faiss.METRIC_L2)
        new_index.train(surviving_vectors)
        new_index.add(surviving_vectors)

        # 更新内部状态
        self.index = new_index
        self.index_to_id = surviving_ids
        self.id_to_index = {vid: i for i, vid in enumerate(surviving_ids)}
        self.deleted_ids.clear()
        self.is_trained = True

    def save_index(self):
        faiss.write_index(self.index, self.index_file_path)

    def load_index(self, file_path):
        self.index = faiss.read_index(file_path)
        self.is_trained = True


# 示例用法
dimension = 768
nlist = 100
faiss_client = FaissClient(dimension, nlist)

# 添加向量到索引
vectors = np.random.rand(10, dimension).astype('float32')
ids = list(range(10))
faiss_client.add_vectors(vectors, ids)

# 搜索向量
query_vectors = np.random.rand(1, dimension).astype('float32')
results = faiss_client.search_vectors(query_vectors, 4)
print('Search results:', results)

# 删除向量
faiss_client.remove_vectors([0, 1])
faiss_client.rebuild_index()

# 保存索引
faiss_client.save_index()

# 加载索引
faiss_client.load_index('faiss_index.idx')

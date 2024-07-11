import json
import traceback
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, \
    Partition
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import copy
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.utils.general_utils import get_time
from qanything_kernel.configs.model_config import MILVUS_HOST_ONLINE, MILVUS_PORT, CHUNK_SIZE, VECTOR_SEARCH_TOP_K
from langchain.docstore.document import Document
from tqdm import tqdm
import math
from itertools import groupby
from typing import List

from qanything_kernel.utils.general_utils import cur_func_name


class MilvusFailed(Exception):
    """异常基类"""
    pass


class MilvusClient:
    def __init__(self, user_id, kb_ids, milvus_cache, *, threshold=1.1, client_timeout=10):
        self.user_id = user_id
        self.kb_ids = kb_ids
        self.host = MILVUS_HOST_ONLINE
        self.port = MILVUS_PORT
        self.client_timeout = client_timeout
        self.threshold = threshold
        self.sess: Collection = None
        self.partitions: List[Partition] = []
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.top_k = VECTOR_SEARCH_TOP_K
        self.search_params = {"metric_type": "L2", "params": {"nprobe": 128}}
        self.create_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
        # self.create_params = {"metric_type": "L2", "index_type": "GPU_IVF_FLAT", "params": {"nlist": 1024}}  # GPU版本
        self.milvus_cache = milvus_cache
        self.init()


    @get_time 
    def __load_collection(self, collection_name, _async=True):
        # 从缓存中获取Collection，如果不存在则加载
        collection = self.milvus_cache.get(collection_name)
        if not collection:
            # 如果Collection不在缓存中，创建它并添加到缓存
            if not utility.has_collection(collection_name):
                schema = CollectionSchema(self.fields)
                debug_logger.info(f'create collection {self.user_id}')
                collection = Collection(self.user_id, schema)
                collection.create_index(field_name="embedding", index_params=self.create_params)
                # raise MilvusFailed(f"Collection {collection_name} does not exist.")
            else:
                collection = Collection(collection_name)
            # collection.load(_async=True)
            self.milvus_cache.put(collection_name, collection, _async=_async)
        return collection

    @property
    def fields(self):
        fields = [
            FieldSchema(name='chunk_id', dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name='file_id', dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name='file_name', dtype=DataType.VARCHAR, max_length=640),
            FieldSchema(name='file_path', dtype=DataType.VARCHAR, max_length=640),
            FieldSchema(name='timestamp', dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name='content', dtype=DataType.VARCHAR, max_length=4000),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name='source_info', dtype=DataType.VARCHAR, max_length=2000) # 记录速读版pdf解析结果中的段落信息，包括 doc_id, page_id(0为起点), chunk_id（是速读解析中构造出的段落chunk，与上面Schema第一个字段不同）
        ]
        return fields

    @get_time
    def parse_batch_result(self, batch_result):
        debug_logger.info(f'batch_result_length {len(batch_result)}')
        new_result = []
        for batch_idx, result in enumerate(batch_result):
            new_cands = []
            result.sort(key=lambda x: x.score)
            valid_results = [cand for cand in result if cand.score <= self.threshold]
            if len(valid_results) == 0:  # 如果没有合适的结果，就取topk
                valid_results = result[:self.top_k]
            for cand_i, cand in enumerate(valid_results):
                source_info = cand.entity.get('source_info')
                if source_info:
                    source_info = json.loads(source_info)
                if cand.entity.get('file_name').endswith('.faq'):
                    faq_dict = json.loads(cand.entity.get('content'))
                    page_content = f"{faq_dict['question']}：{faq_dict['answer']}"
                    nos_keys = faq_dict.get('nos_keys')
                else:
                    page_content = cand.entity.get('content')
                    nos_keys = ''
                doc = Document(page_content=page_content,
                               metadata={"score": cand.score, "file_id": cand.entity.get('file_id'),
                                         "file_name": cand.entity.get('file_name'),
                                         "chunk_id": cand.entity.get('chunk_id'),
                                         "source_info": source_info, "nos_keys": nos_keys})
                new_cands.append(doc)
            # pdf, csv和xlsx文件不做expand_cand_docs
            need_expand, not_need_expand = [], []
            for doc in new_cands:
                if doc.metadata['file_name'].lower().split('.')[-1] in ['csv', 'xlsx', 'pdf', 'faq']:
                    doc.metadata["kernel"] = doc.page_content
                    not_need_expand.append(doc)
                else:
                    need_expand.append(doc)
            expand_res = self.expand_cand_docs(need_expand)
            new_cands = not_need_expand + expand_res
            new_result.append(new_cands)
        return new_result

    @property
    def output_fields(self):
        if 'source_info' in str(self.sess.schema.fields):
            return ['chunk_id', 'file_id', 'file_name', 'file_path', 'timestamp', 'content', 'source_info']
        else:
            return ['chunk_id', 'file_id', 'file_name', 'file_path', 'timestamp', 'content'] 

    def init(self):
        try:
            connections.connect(host=self.host, port=self.port)  # timeout=3 [cannot set]
            self.sess = self.__load_collection(self.user_id)
            for kb_id in self.kb_ids:
                if not self.sess.has_partition(kb_id):
                    self.sess.create_partition(kb_id)
            self.partitions = [Partition(self.sess, kb_id) for kb_id in self.kb_ids]
        except Exception as e:
            debug_logger.error(f'[{cur_func_name()}] [MilvusClient] traceback = {traceback.format_exc()}')

    def __search_emb_sync(self, embs, expr='', top_k=None, client_timeout=None):
        if not top_k:
            top_k = self.top_k
        
        milvus_records = []
        retry_count = 0
        max_retries = 1
        while retry_count <= max_retries:
            try:
                milvus_records = self.sess.search(data=embs, partition_names=self.kb_ids, anns_field="embedding",
                                                  param=self.search_params, limit=top_k,
                                                  output_fields=self.output_fields, expr=expr, timeout=client_timeout)
                break
            except Exception as e:
                debug_logger.error(e)
                retry_count += 1  # 出错时增加重试计数
                if retry_count > max_retries:
                    debug_logger.error("milvus搜索重试失败，停止尝试。")
                    return []
                else:
                    debug_logger.info("milvus搜索失败，正在尝试重试。")
                    try:
                        self.sess = self.__load_collection(self.user_id, _async=False)
                    except Exception as e:
                        debug_logger.error("重新加载 milvus 集合失败：" + str(e))
                        return []
        return self.parse_batch_result(milvus_records)

    def search_emb_async(self, embs, expr='', top_k=None, client_timeout=None):
        if not top_k:
            top_k = self.top_k
        # 将search_emb_sync函数放入线程池中运行
        future = self.executor.submit(self.__search_emb_sync, embs, expr, top_k, client_timeout)
        return future.result()

    @get_time
    def query_expr_async(self, expr, output_fields=None, client_timeout=None):
        if client_timeout is None:
            client_timeout = self.client_timeout
        if not output_fields:
            output_fields = self.output_fields
        future = self.executor.submit(
            partial(self.sess.query, partition_names=self.kb_ids, output_fields=output_fields, expr=expr,
                    timeout=client_timeout))
        return future.result()
    
    def delete_collection(self):
        self.milvus_cache.remove(self.user_id)
        utility.drop_collection(self.user_id)
    
    def delete_partition(self, partition_name):
        self.milvus_cache.remove(self.user_id)
        if self.sess.has_partition(partition_name):
            self.sess.drop_partition(partition_name)
    
    def delete_files(self, files_id):
        batch_size = 100
        num_docs = len(files_id)
        for batch_start in range(0, num_docs, batch_size):
            batch_end = min(batch_start + batch_size, num_docs)
            self.delete_files_batch(files_id[batch_start:batch_end])
    
    def delete_files_batch(self, files_id):
        res = self.query_expr_async(expr=f"file_id in {files_id}", output_fields=["chunk_id"])
        if res:
            valid_ids = [result['chunk_id'] for result in res]
            self.sess.delete(expr=f"chunk_id in {valid_ids}")
            debug_logger.info('milvus delete files_id: %s', files_id)

    def get_files(self, files_id):
        res = self.query_expr_async(expr=f"file_id in {files_id}", output_fields=["file_id"])
        valid_ids = [result['file_id'] for result in res]
        return valid_ids

    def seperate_list(self, ls: List[int]) -> List[List[int]]:
        lists = []
        ls1 = [ls[0]]
        for i in range(1, len(ls)):
            if ls[i - 1] + 1 == ls[i]:
                ls1.append(ls[i])
            else:
                lists.append(ls1)
                ls1 = [ls[i]]
        lists.append(ls1)
        return lists

    @get_time
    def process_group(self, group):
        new_cands = []
        group.sort(key=lambda x: int(x.metadata['chunk_id'].split('_')[-1]))
        id_set = set()
        file_id = group[0].metadata['file_id']
        file_name = group[0].metadata['file_name']
        group_scores_map = {}
        # 先找出该文件所有需要搜索的chunk_id
        cand_chunks_set = set()  # 使用集合而不是列表
        for cand_doc in group:
            current_chunk_id = int(cand_doc.metadata['chunk_id'].split('_')[-1])
            group_scores_map[current_chunk_id] = cand_doc.metadata['score']
            # 使用 set comprehension 一次性生成区间内所有可能的 chunk_id
            chunk_ids = {file_id + '_' + str(i) for i in range(current_chunk_id - 200, current_chunk_id + 200)}
            # 更新 cand_chunks_set 集合
            cand_chunks_set.update(chunk_ids)

        cand_chunks = list(cand_chunks_set)
        group_relative_chunks = self.query_expr_async(expr=f"file_id == \"{file_id}\" and chunk_id in {cand_chunks}",
                                                      output_fields=["chunk_id", "content"])
        group_chunk_map = {int(item['chunk_id'].split('_')[-1]): item['content'] for item in group_relative_chunks}

        group_file_chunk_num = list(group_chunk_map.keys())
        for cand_doc in group:
            current_chunk_id = int(cand_doc.metadata['chunk_id'].split('_')[-1])
            doc = copy.deepcopy(cand_doc)
            id_set.add(current_chunk_id)
            docs_len = len(doc.page_content)
            for k in range(1, 200):
                break_flag = False
                for expand_index in [current_chunk_id + k, current_chunk_id - k]:
                    if expand_index in group_file_chunk_num:
                        merge_content = group_chunk_map[expand_index]
                        if docs_len + len(merge_content) > CHUNK_SIZE:
                            break_flag = True
                            break
                        else:
                            docs_len += len(merge_content)
                            id_set.add(expand_index)
                if break_flag:
                    break

        id_list = sorted(list(id_set))
        id_lists = self.seperate_list(id_list)
        for id_seq in id_lists:
            for id in id_seq:
                if id == id_seq[0]:
                    doc = Document(page_content=group_chunk_map[id],
                                   metadata={"score": 0, "file_id": file_id,
                                             "file_name": file_name})
                else:
                    doc.page_content += " " + group_chunk_map[id]
            doc_score = min([group_scores_map[id] for id in id_seq if id in group_scores_map])
            doc.metadata["score"] = float(format(1 - doc_score / math.sqrt(2), '.4f'))
            doc.metadata["kernel"] = '|'.join([group_chunk_map[id] for id in id_seq if id in group_scores_map])
            new_cands.append(doc)
        return new_cands

    def expand_cand_docs(self, cand_docs):
        cand_docs = sorted(cand_docs, key=lambda x: x.metadata['file_id'])
        # 按照file_id进行分组
        m_grouped = [list(group) for key, group in groupby(cand_docs, key=lambda x: x.metadata['file_id'])]
        debug_logger.info('当前用户问题搜索到的相关文档数量（非切片数） : %s', len(m_grouped))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            # 对每个分组按照chunk_id进行排序
            for group in m_grouped:
                if not group:
                    continue
                future = executor.submit(self.process_group, group)
                futures.append(future)

            new_cands = []
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    new_cands.extend(result)
            return new_cands

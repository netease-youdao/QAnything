# -*- coding: utf-8 -*-
from __future__ import print_function  # 确保 print 函数在 Python 2 中的行为与 Python 3 一致

def get_run_config_params():
    openai_api_base = "https://api.openai.com/v1"
    openai_api_key = "sk-xxxxxxx"
    openai_api_model_name = "gpt-3.5-turbo-1106"
    openai_api_context_length = "4096"
    workers = 4
    milvus_port = 19530
    qanything_port = 8777
    use_cpu = True
    # 使用 .format() 方法格式化字符串，以兼容 Python 2
    return "{},{},{},{},{},{},{}".format(openai_api_base, openai_api_key, openai_api_model_name,
                                         openai_api_context_length, workers, milvus_port, qanything_port, use_cpu)

# 模型参数
llm_config = {
    # 回答的最大token数，一般来说对于国内模型一个中文不到1个token，国外模型一个中文1.5-2个token
    "max_token": 512,
    # 附带的上下文数目
    "history_len": 2,
    # 总共的token数，如果遇到电脑显存不够的情况可以将此数字改小，如果低于3000仍然无法使用，就更换模型
    "token_window": 4096,
    # 如果报错显示top_p值必须在0到1，可以在这里修改
    "top_p": 1.0
}

# pdf解析参数
pdf_config = {
    # 设置是否使用快速PDF解析器，设置为False时，使用优化后的PDF解析器，但速度下降
    "USE_FAST_PDF_PARSER": True
}

# 一些可以提高性能和速度的参数，对硬件要求更高，请根据实际情况调整
user_defined_configuration = {
    # 设置rerank的batch大小，16GB内存建议设置为8，32GB内存建议设置为16
    "LOCAL_RERANK_BATCH": 8,
    # 设置rerank的多线程worker数量，默认设置为4，根据机器性能调整
    "LOCAL_RERANK_WORKERS": 4,
    # 设置embed的batch大小，16GB内存建议设置为8，32GB内存建议设置为16
    "LOCAL_EMBED_BATCH": 8,
    # 设置embed的多线程worker数量，默认设置为4，根据机器性能调整
    "LOCAL_EMBED_WORKERS": 4
}

#### 一般情况下，除非特殊需要，不要修改一下字段参数 ####
# 解析文档基本参数
model_config = {
    # 文本分句长度
    "SENTENCE_SIZE": 100,
    # 匹配后单段上下文长度
    "CHUNK_SIZE": 800,
    # 知识库检索时返回的匹配内容条数
    "VECTOR_SEARCH_TOP_K": 40,
    # embedding检索的相似度阈值，归一化后的L2距离，设置越大，召回越多，设置越小，召回越少
    "VECTOR_SEARCH_SCORE_THRESHOLD": 1.1
}
text_splitter_config = {
    # 切割文件chunk块的大小
    "chunk_size": 400,
    # 切割文件的相邻文本重合长度
    "chunk_overlap": 100
}
pdf_splitter_config = {
    # 切割文件chunk块的大小
    "chunk_size": 800,
    # 切割文件的相邻文本重合长度
    "chunk_overlap": 0
}
#### 一般情况下，除非特殊需要，不要修改一下字段参数 ####


if __name__ == "__main__":
    import sys
    sys.stdout.write(''.join(get_run_config_params()))
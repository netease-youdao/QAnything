import logging
import sys

from concurrent_log_handler import ConcurrentRotatingFileHandler
import time
import os

# 创建自定义过滤器
class FilterOutMessages(logging.Filter):
    def filter(self, record):

        exclude_keywords = [
            "local doc search retrieval_documents",
            "prompt: 参考信息：",
            "[debug] event_text =",
            "retrieval_documents: [Document(page_content",
            "reranked retrieval_documents: [Document(page_content=",
            "'role': 'user'",
        ]

        # 检查日志消息中是否包含要过滤掉的关键词
        for keyword in exclude_keywords:
            if keyword in record.msg:
                return False  # 如果包含关键词，则不记录该日志消息

        if isinstance(record.msg, list):
            return False

        return True  # 如果日志消息中不包含任何要过滤掉的关键词，则记录该消息



class CustomConcurrentRotatingFileHandler(ConcurrentRotatingFileHandler):
    def doRollover(self):
        """
        Do a rollover, as described in __init__().
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        # Add timestamp to the filename for the old log
        current_time = time.strftime("%Y%m%d_%H%M%S")
        dfn = self.rotation_filename(f"{self.baseFilename}.{current_time}")

        if os.path.exists(dfn):
            os.remove(dfn)
        self.rotate(self.baseFilename, dfn)

        if not self.delay:
            self.stream = self._open()


# 这是LogHandler的代码，用于将日志写入文件
# 获取当前时间作为日志文件名的一部分
current_time = time.strftime("%Y%m%d_%H%M%S")
# 定义日志文件夹路径
debug_log_folder = './logs/debug_logs'
qa_log_folder = './logs/qa_logs'
# 确保日志文件夹存在
if not os.path.exists(debug_log_folder):
    os.makedirs(debug_log_folder)
if not os.path.exists(qa_log_folder):
    os.makedirs(qa_log_folder)
# 定义日志文件的完整路径，包括文件夹和文件名
# log_file = os.path.join(log_folder, f'log_{current_time}.log')

# 创建一个 logger 实例
qa_logger = logging.getLogger('qa_logger')
debug_logger = logging.getLogger('debug_logger')
# 设置 logger 的日志级别为 INFO，即只记录 INFO 及以上级别的日志信息
qa_logger.setLevel(logging.INFO)
debug_logger.setLevel(logging.INFO)

# 创建一个 ConcurrentRotatingFileHandler 实例
# log_file: 日志文件名
# "a": 文件的打开模式，追加模式
# 16*1024*1024: maxBytes，当日志文件达到 512KB 时进行轮转
# 5: backupCount，保留 5 个轮转日志文件的备份
debug_handler = ConcurrentRotatingFileHandler(os.path.join(debug_log_folder, "debug.log"), "a", 16 * 1024 * 1024, 5)
# 定义日志格式
# 创建一个自定义字段，用于表示是主进程还是子进程
process_type = 'MainProcess' if 'SANIC_WORKER_NAME' not in os.environ else os.environ['SANIC_WORKER_NAME']

# 创建一个带有自定义字段的格式器
formatter = logging.Formatter(f"%(asctime)s - [PID: %(process)d][{process_type}] - [Function: %(funcName)s] - %(levelname)s - %(message)s")

# formatter = logging.Formatter("%(asctime)s - %(name)s - [PID: %(process)d] - %(levelname)s - %(message)s")
# 设置日志格式
debug_handler.setFormatter(formatter)

# 将 handler 添加到 logger 中，这样 logger 就可以使用这个 handler 来记录日志了
debug_logger.addHandler(debug_handler)

qa_handler = ConcurrentRotatingFileHandler(os.path.join(qa_log_folder, "qa.log"), "a", 64 * 1024 * 1024, 16)
# 定义日志格式
formatter = logging.Formatter("%(asctime)s %(message)s")
# 设置日志格式
qa_handler.setFormatter(formatter)

# 将 handler 添加到 logger 中，这样 logger 就可以使用这个 handler 来记录日志了
qa_logger.addHandler(qa_handler)


''''''
# filter_out_messages = FilterOutMessages()
# 添加控制台日志处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)  # 可以根据需要设置适当的日志级别
console_handler.setFormatter(formatter)  # 使用与文件日志相同的格式
console_handler.addFilter(FilterOutMessages())
# 添加过滤器到处理器
debug_handler.addFilter(FilterOutMessages())
qa_handler.addFilter(FilterOutMessages())
# 将控制台处理器添加到日志记录器
# qa_logger.addHandler(console_handler)
debug_logger.addHandler(console_handler)



# 原来的 print，但是好像不能输出日志
# print(debug_logger, qa_logger)


# 测试日志输出
# debug_logger.info("This is a debug message")
# qa_logger.info("This is a QA message")


qa_logger.propagate = False  # 关闭日志传播
debug_logger.propagate = False  # 关闭日志传播
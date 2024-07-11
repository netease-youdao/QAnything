import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
import time
import os


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
writing_log_folder = './logs/writing_logs'
debug_log_folder = './logs/debug_logs'
qa_log_folder = './logs/qa_logs'
rerank_log_folder = './logs/rerank_logs'
insert_log_folder = './logs/insert_logs'
# 确保日志文件夹存在
if not os.path.exists(writing_log_folder):
    os.makedirs(writing_log_folder)
if not os.path.exists(debug_log_folder):
    os.makedirs(debug_log_folder)
if not os.path.exists(qa_log_folder):
    os.makedirs(qa_log_folder)
if not os.path.exists(rerank_log_folder):
    os.makedirs(rerank_log_folder)
if not os.path.exists(insert_log_folder):
    os.makedirs(insert_log_folder)
# 定义日志文件的完整路径，包括文件夹和文件名
# log_file = os.path.join(log_folder, f'log_{current_time}.log')

# 创建一个 logger 实例
writing_logger = logging.getLogger('writing_logger')
qa_logger = logging.getLogger('qa_logger')
debug_logger = logging.getLogger('debug_logger')
rerank_logger = logging.getLogger('rerank_logger')
insert_logger = logging.getLogger('insert_logger')
# 设置 logger 的日志级别为 INFO，即只记录 INFO 及以上级别的日志信息
writing_logger.setLevel(logging.INFO)
qa_logger.setLevel(logging.INFO)
debug_logger.setLevel(logging.INFO)
rerank_logger.setLevel(logging.INFO)
insert_logger.setLevel(logging.INFO)

# 创建一个 ConcurrentRotatingFileHandler 实例
# log_file: 日志文件名
# "a": 文件的打开模式，追加模式
# 16*1024*1024: maxBytes，当日志文件达到 512KB 时进行轮转
# 5: backupCount，保留 5 个轮转日志文件的备份
writing_handler = ConcurrentRotatingFileHandler(os.path.join(writing_log_folder, "writing.log"), "a", 64 * 1024 * 1024, 256)
# 定义日志格式
# 创建一个自定义字段，用于表示是主进程还是子进程
process_type = 'MainProcess' if 'SANIC_WORKER_NAME' not in os.environ else os.environ['SANIC_WORKER_NAME']

# 创建一个带有自定义字段的格式器
formatter = logging.Formatter(f"%(asctime)s - [PID: %(process)d][{process_type}] - [Function: %(funcName)s] - %(levelname)s - %(message)s")

# formatter = logging.Formatter("%(asctime)s - %(name)s - [PID: %(process)d] - %(levelname)s - %(message)s")
# 设置日志格式
writing_handler.setFormatter(formatter)

# 将 handler 添加到 logger 中，这样 logger 就可以使用这个 handler 来记录日志了
writing_logger.addHandler(writing_handler)


debug_handler = ConcurrentRotatingFileHandler(os.path.join(debug_log_folder, "debug.log"), "a", 64 * 1024 * 1024, 256)
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

qa_handler = ConcurrentRotatingFileHandler(os.path.join(qa_log_folder, "qa.log"), "a", 64 * 1024 * 1024, 256)
# 定义日志格式
formatter = logging.Formatter("%(asctime)s %(message)s")
# 设置日志格式
qa_handler.setFormatter(formatter)

# 将 handler 添加到 logger 中，这样 logger 就可以使用这个 handler 来记录日志了
qa_logger.addHandler(qa_handler)


rerank_handler = ConcurrentRotatingFileHandler(os.path.join(rerank_log_folder, "rerank.log"), "a", 64 * 1024 * 1024, 256)
# 定义日志格式
formatter = logging.Formatter("%(asctime)s %(message)s")
# 设置日志格式
rerank_handler.setFormatter(formatter)

# 将 handler 添加到 logger 中，这样 logger 就可以使用这个 handler 来记录日志了
rerank_logger.addHandler(rerank_handler)

insert_handler = ConcurrentRotatingFileHandler(os.path.join(insert_log_folder, "insert.log"), "a", 64 * 1024 * 1024, 256)
# 定义日志格式
formatter = logging.Formatter(f"%(asctime)s - [PID: %(process)d][{process_type}] - [Function: %(funcName)s] - %(levelname)s - %(message)s")
# 设置日志格式
insert_handler.setFormatter(formatter)

# 将 handler 添加到 logger 中，这样 logger 就可以使用这个 handler 来记录日志了
insert_logger.addHandler(insert_handler)

print(writing_logger, debug_logger, qa_logger, rerank_logger, insert_logger)

writing_logger.propagate = False  # 关闭日志传播
qa_logger.propagate = False  # 关闭日志传播
debug_logger.propagate = False  # 关闭日志传播
rerank_logger.propagate = False  # 关闭日志传播
insert_logger.propagate = False  # 关闭日志传播

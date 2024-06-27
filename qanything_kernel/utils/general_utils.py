from sanic.request import Request
from sanic.exceptions import BadRequest
import traceback
from urllib.parse import urlparse
import time
import os
import logging
import re
import tiktoken
from io import BytesIO
import pandas as pd

__all__ = ['write_check_file', 'isURL', 'format_source_documents', 'get_time', 'safe_get', 'truncate_filename',
           'read_files_with_extensions', 'validate_user_id', 'get_invalid_user_id_msg', 'num_tokens',
           'simplify_filename', 'check_and_transform_excel'
           ]


def get_invalid_user_id_msg(user_id):
    return "fail, Invalid user_id: {}. user_id 必须只含有字母，数字和下划线且字母开头".format(user_id)


def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()


def isURL(string):
    result = urlparse(string)
    return result.scheme != '' and result.netloc != ''


def format_source_documents(ori_source_documents):
    source_documents = []
    for inum, doc in enumerate(ori_source_documents):
        # for inum, doc in enumerate(answer_source_documents):
        # doc_source = doc.metadata['source']
        file_id = doc.metadata['file_id']
        file_name = doc.metadata['file_name']
        # source_str = doc_source if isURL(doc_source) else os.path.split(doc_source)[-1]
        source_info = {'file_id': doc.metadata['file_id'],
                       'file_name': doc.metadata['file_name'],
                       'content': doc.page_content,
                       'retrieval_query': doc.metadata['retrieval_query'],
                       'kernel': doc.metadata['kernel'],
                       'score': str(doc.metadata['score']),
                       'embed_version': doc.metadata['embed_version']}
        source_documents.append(source_info)
    return source_documents


def get_time(func):
    def inner(*arg, **kwargs):
        s_time = time.time()
        res = func(*arg, **kwargs)
        e_time = time.time()
        print('函数 {} 执行耗时: {} 秒'.format(func.__name__, e_time - s_time))
        return res

    return inner


def safe_get(req: Request, attr: str, default=None):
    try:
        if attr in req.form:
            return req.form.getlist(attr)[0]
        if attr in req.args:
            return req.args[attr]
        if attr in req.json:
            return req.json[attr]
        # if value := req.form.get(attr):
        #     return value
        # if value := req.args.get(attr):
        #     return value
        # """req.json执行时不校验content-type，body字段可能不能被正确解析为json"""
        # if value := req.json.get(attr):
        #     return value
    except BadRequest:
        logging.warning(f"missing {attr} in request")
    except Exception as e:
        logging.warning(f"get {attr} from request failed:")
        logging.warning(traceback.format_exc())
    return default


def truncate_filename(filename, max_length=200):
    # 获取文件名后缀
    file_ext = os.path.splitext(filename)[1]

    # 获取不带后缀的文件名
    file_name_no_ext = os.path.splitext(filename)[0]

    # 计算文件名长度，注意中文字符
    filename_length = len(filename.encode('utf-8'))

    # 如果文件名长度超过最大长度限制
    if filename_length > max_length:
        # 生成一个时间戳标记
        timestamp = str(int(time.time()))
        # 截取文件名
        while filename_length > max_length:
            file_name_no_ext = file_name_no_ext[:-4]
            new_filename = file_name_no_ext + "_" + timestamp + file_ext
            filename_length = len(new_filename.encode('utf-8'))
    else:
        new_filename = filename

    return new_filename


def read_files_with_extensions():
    # 获取当前脚本文件的路径
    current_file = os.path.abspath(__file__)

    # 获取当前脚本文件所在的目录
    current_dir = os.path.dirname(current_file)

    # 获取项目根目录
    project_dir = os.path.dirname(current_dir)

    directory = project_dir + '/data'
    print(f'now reading {directory}')
    extensions = ['.md', '.txt', '.pdf', '.jpg', '.docx', '.xlsx', '.eml', '.csv']
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(tuple(extensions)):
                file_path = os.path.join(root, file)
                yield file_path


def validate_user_id(user_id):
    return True
    # 定义正则表达式模式
    pattern = r'^[A-Za-z][A-Za-z0-9_]*$'
    # 检查是否匹配
    if isinstance(user_id, str) and re.match(pattern, user_id):
        return True
    else:
        return False


def num_tokens(text: str, model: str = 'gpt-3.5-turbo-0613') -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def simplify_filename(filename, max_length=40):
    if len(filename) <= max_length:
        # 如果文件名长度小于等于最大长度，直接返回原文件名
        return filename

    # 分离文件的基本名和扩展名
    name, extension = filename.rsplit('.', 1)
    extension = '.' + extension  # 将点添加回扩展名

    # 计算头部和尾部的保留长度
    part_length = (max_length - len(extension) - 1) // 2  # 减去扩展名长度和破折号的长度
    end_start = -part_length if part_length else None

    # 构建新的简化文件名
    simplified_name = f"{name[:part_length]}-{name[end_start:]}" if part_length else name[:max_length - 1]

    return f"{simplified_name}{extension}"


def check_and_transform_excel(binary_data):
    # 使用BytesIO读取二进制数据
    try:
        data_io = BytesIO(binary_data)
        df = pd.read_excel(data_io)
    except Exception as e:
        return f"读取文件时出错: {e}"

    # 检查列数
    if len(df.columns) != 2:
        return "格式错误：文件应该只有两列"

    # 检查列标题
    if df.columns[0] != "问题" or df.columns[1] != "答案":
        return "格式错误：第一列标题应为'问题'，第二列标题应为'答案'"

    # 检查每行长度
    for index, row in df.iterrows():
        question_len = len(row['问题'])
        answer_len = len(row['答案'])
        if question_len > 512 or answer_len > 2048:
            return f"行{index + 1}长度超出限制：问题长度={question_len}，答案长度={answer_len}"

    # 转换数据格式
    transformed_data = []
    for _, row in df.iterrows():
        transformed_data.append({"question": row['问题'], "answer": row['答案']})

    return transformed_data

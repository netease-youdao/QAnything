# -*- coding: utf-8 -*-
import traceback
import os
import botocore
import boto3
from qanything_kernel.configs.model_config import BUCKET_NAME, ACCESS_KEY, SECRET_KEY, END_POINT, S3_END_POINT, UPLOAD_ROOT_PATH
from qanything_kernel.utils.general_utils import get_time

from qanything_kernel.utils.custom_log import debug_logger, insert_logger

S3_NOS_CONFIG = botocore.config.Config(
    connect_timeout=120,
    max_pool_connections=20,
    retries={"max_attempts": 2},
    s3={'addressing_style': 'virtual'}  # nos公有云只支持子域名的方式
)

S3CLIENT = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    endpoint_url=S3_END_POINT,  # 必须包含scheme，scheme中的http/https决定了使用什么方式上传对象
    config=S3_NOS_CONFIG
)


def construct_nos_key_prefix(vdb_path):
    assert (vdb_path)
    vdb_name = os.path.basename(vdb_path)
    return f"zhiyun/docqa/doc_vdb/{vdb_name}"


def construct_paper_nos_key_prefix(paper):
    return f"zhiyun/docqa/scholar_paper/paper_json"


def construct_qanything_local_file_nos_key_prefix(user_id, kb_id, file_id):
    return f"zhiyun/docqa/qanything/local_file/{user_id}/{kb_id}/{file_id}"


def download_nos_file(nos_key):
    logger = insert_logger
    logger.info(f"download nos file: {nos_key}")
    error = None
    try:
        result = S3CLIENT.get_object(Bucket=BUCKET_NAME, Key=nos_key)
        fp = result.get("Body")
        return fp, error
        # object_str = fp.read()
        # print("object content: ", object_str)
    except botocore.exceptions.ClientError as e:
        error = (
                "ServiceError: %s\n"
                "status_code: %s\n"
                "error_code: %s\n"
                "request_id: %s\n"
                "message: %s\n"
                % (
                    e,
                    e.response['ResponseMetadata']['HTTPStatusCode'],  # 错误http状态码
                    e.response["Error"]['Code'],  # NOS服务器定义错误类型
                    e.response['ResponseMetadata']['HTTPHeaders']['x-nos-request-id'],  # NOS服务器定义错误码
                    e.response['Error']['Message'],  # 请求ID，有利于nos开发人员跟踪异常请求的错误原因
                ))
        return None, error
    except botocore.exceptions.ParamValidationError as e:
        error = (
                "ClientError: %s\n"
                "message: %s\n"
                % (
                    e,
                    e.fmt
                ))
        logger.error(error)
        return error


"""
用boto3 sdk，即 NOS S3 python sdk。
比起 nos-python-sdk 的优势是，支持生成资源下载链接，可用于浏览器下载。
"""


def upload_nos_file_bytes_or_str(nos_key, file_bytes_or_str):
    logger = debug_logger
    try:
        response = S3CLIENT.put_object(Bucket=BUCKET_NAME, Key=nos_key, Body=file_bytes_or_str)
        return response
    except botocore.exceptions.ClientError as e:
        error = (
                "ServiceError: %s\n"
                "status_code: %s\n"
                "error_code: %s\n"
                "request_id: %s\n"
                "message: %s\n"
                % (
                    e,
                    e.response['ResponseMetadata']['HTTPStatusCode'],  # 错误http状态码
                    e.response["Error"]['Code'],  # NOS服务器定义错误类型
                    e.response['ResponseMetadata']['HTTPHeaders']['x-nos-request-id'],  # NOS服务器定义错误码
                    e.response['Error']['Message'],  # 请求ID，有利于nos开发人员跟踪异常请求的错误原因
                ))
        logger.error(error)
        return error
    except botocore.exceptions.ParamValidationError as e:
        error = (
                "ClientError: %s\n"
                "message: %s\n"
                % (
                    e,
                    e.fmt
                ))
        logger.error(error)
        return error


"""
用boto3 sdk，即 NOS S3 python sdk。
比起 nos-python-sdk 的优势是，支持生成资源下载链接，可用于浏览器下载。
"""


def upload_nos_file(nos_key, local_filepath):
    logger = debug_logger
    try:
        response = S3CLIENT.put_object(Body=open(local_filepath, "rb"), Bucket=BUCKET_NAME, Key=nos_key)
        return response
    except botocore.exceptions.ClientError as e:
        error = (
                "ServiceError: %s\n"
                "status_code: %s\n"
                "error_code: %s\n"
                "request_id: %s\n"
                "message: %s\n"
                % (
                    e,
                    e.response['ResponseMetadata']['HTTPStatusCode'],  # 错误http状态码
                    e.response["Error"]['Code'],  # NOS服务器定义错误类型
                    e.response['ResponseMetadata']['HTTPHeaders']['x-nos-request-id'],  # NOS服务器定义错误码
                    e.response['Error']['Message'],  # 请求ID，有利于nos开发人员跟踪异常请求的错误原因
                ))
        logger.error(error)
        return error
    except botocore.exceptions.ParamValidationError as e:
        error = (
                "ClientError: %s\n"
                "message: %s\n"
                % (
                    e,
                    e.fmt
                ))
        logger.error(error)
        return error


def upload_nos_file_retry(nos_key, local_filepath):
    for i in range(3):
        res = upload_nos_file(nos_key, local_filepath)
        if res is None:
            return res
    return f"Retried 3 times, failed uploading: {nos_key}"


@get_time
def upload_nos_file_bytes_or_str_retry(nos_key, file_bytes_or_str):
    res = None
    for i in range(3):
        res = upload_nos_file_bytes_or_str(nos_key, file_bytes_or_str)
        if isinstance(res, dict):
            return res
    return f"Retried 3 times, failed uploading: {res}"


def download_vdb_from_nos_and_save_local_file(vdb_paths):
    assert (len(vdb_paths) > 0)
    vdb_path = vdb_paths[0]
    nos_key_prefix = construct_nos_key_prefix(vdb_path)  # 把 vdb_path 最后一段拿出来，拼出一个 nos_key
    # 专门用于下载和写入 vdb 文件。因为其组织形式是一个 vdb 对应于一个目录下有 index.faiss 和 index.pkl 俩文件。
    # 先创建目录
    try:
        if not os.path.exists(vdb_path):
            os.makedirs(vdb_path)
    except:
        debug_logger.error(f'[download_vdb_to_nos] makedirs failed: {vdb_path}')
    try:
        for suf in ['index.faiss', 'index.pkl']:
            nos_key = f'{nos_key_prefix}/{suf}'
            fp, error = download_nos_file(nos_key)
            if error:
                # break
                return False
            with open(f'{vdb_path}/{suf}', 'wb') as f:
                f.write(fp.read())
            debug_logger.info(f'[download_vdb_to_nos] successfully downloaded: {nos_key}, vdb_path = {vdb_path}')
            fp.close()
        return True
    except:
        debug_logger.error(f'[download_vdb_to_nos] download failed: {vdb_path}, nos_key = {nos_key}')
        debug_logger.error(f'[download_vdb_to_nos] {traceback.format_exc()}')
        return False


def upload_vdb_to_nos(vdb_path):
    assert (vdb_path)
    nos_key_prefix = construct_nos_key_prefix(vdb_path)  # 把 vdb_path 最后一段拿出来，拼出一个 nos_key
    for suf in ['index.pkl', 'index.faiss']:
        nos_key = f'{nos_key_prefix}/{suf}'
        local_filepath = f'{vdb_path}/{suf}'
        error = upload_nos_file(nos_key, local_filepath)
        if error:
            break
        debug_logger.info(f'[upload_vdb_to_nos] successfully uploaded: {nos_key}')


def construct_nos_key_for_user_pdf(file_id):
    return f"zhiyun/docqa/qanything/{file_id}.pdf"


def construct_nos_key_for_user_pdf_chunks_json(file_id):
    return f"zhiyun/docqa/qanything/chunks_json/{file_id}.json"


def construct_nos_key_for_local_file(user_id, kb_id, file_id, file_name):
    return f"{construct_qanything_local_file_nos_key_prefix(user_id, kb_id, file_id)}/{file_name}"


def from_nos_key_to_local_path(nos_key):
    return os.path.join(UPLOAD_ROOT_PATH, '/'.join(nos_key.split('/')[-3:]))


"""
boto3，生成 nosURL，供前端下载
"""


def gen_nos_url(nos_key):
    url = S3CLIENT.generate_presigned_url(
        ClientMethod='get_object',
        Params={
            'Bucket': BUCKET_NAME,
            'Key': nos_key
        }  # ,
        # ExpiresIn = 10 # 单位s，默认3600s
    )
    return url

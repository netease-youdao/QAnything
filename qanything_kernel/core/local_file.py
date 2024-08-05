from typing import Union, Tuple, Dict
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from sanic.request import File
from qanything_kernel.configs.model_config import UPLOAD_ROOT_PATH
import uuid
import os


class LocalFile:
    def __init__(self, user_id, kb_id, file: Union[File, str, Dict], file_name):
        self.user_id = user_id
        self.kb_id = kb_id
        self.file_id = uuid.uuid4().hex
        self.file_name = file_name
        self.file_url = ''
        if isinstance(file, Dict):
            self.file_location = "FAQ"
            self.file_content = b''
        elif isinstance(file, str):
            self.file_location = "URL"
            self.file_content = b''
            self.file_url = file
        else:
            self.file_content = file.body
            # nos_key = construct_nos_key_for_local_file(user_id, kb_id, self.file_id, self.file_name)
            # debug_logger.info(f'file nos_key: {self.file_id}, {self.file_name}, {nos_key}')
            # self.file_location = nos_key
            # upload_res = upload_nos_file_bytes_or_str_retry(nos_key, self.file_content)
            # if 'failed' in upload_res:
            #     debug_logger.error(f'failed init localfile {self.file_name}, {upload_res}')
            # else:
            #     debug_logger.info(f'success init localfile {self.file_name}, {upload_res}')
            upload_path = os.path.join(UPLOAD_ROOT_PATH, user_id)
            file_dir = os.path.join(upload_path, self.kb_id, self.file_id)
            os.makedirs(file_dir, exist_ok=True)
            self.file_location = os.path.join(file_dir, self.file_name)
            #  如果文件不存在：
            if not os.path.exists(self.file_location):
                with open(self.file_location, 'wb') as f:
                    f.write(self.file_content)

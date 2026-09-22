from typing import Union, Tuple, Dict
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from sanic.request import File
from qanything_kernel.configs.model_config import UPLOAD_ROOT_PATH
from qanything_kernel.utils.path_security import safe_join, validate_filename
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
            validate_filename(self.file_name)
            # nos_key = construct_nos_key_for_local_file(user_id, kb_id, self.file_id, self.file_name)
            # debug_logger.info(f'file nos_key: {self.file_id}, {self.file_name}, {nos_key}')
            # self.file_location = nos_key
            # upload_res = upload_nos_file_bytes_or_str_retry(nos_key, self.file_content)
            # if 'failed' in upload_res:
            #     debug_logger.error(f'failed init localfile {self.file_name}, {upload_res}')
            # else:
            #     debug_logger.info(f'success init localfile {self.file_name}, {upload_res}')
            upload_path = safe_join(UPLOAD_ROOT_PATH, user_id)
            file_dir = safe_join(upload_path, self.kb_id, self.file_id)
            os.makedirs(file_dir, exist_ok=True)
            # Resolve from the trusted upload root again after directory
            # creation, so a raced symlink cannot turn ``file_dir`` into a
            # new trust boundary.
            self.file_location = safe_join(upload_path, self.kb_id, self.file_id, self.file_name)
            # Create once, without following/replacing an existing file.
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(self.file_location, flags, 0o600)
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, 'wb') as output_file:
                    output_file.write(self.file_content)

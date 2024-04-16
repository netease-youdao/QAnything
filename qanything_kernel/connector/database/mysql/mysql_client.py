import sqlite3
from qanything_kernel.configs.model_config import SQLITE_DATABASE 
from qanything_kernel.utils.custom_log import debug_logger
import uuid


class KnowledgeBaseManager:
    def __init__(self):
        self.database = SQLITE_DATABASE
        self.create_tables_()
        debug_logger.info("[SUCCESS] 数据库{}连接成功".format(self.database))

    def execute_query_(self, query, params, commit=False, fetch=False):
        conn = sqlite3.connect(self.database)
        # 启用外键约束
        conn.execute('PRAGMA foreign_keys = ON')
        cursor = conn.cursor()
        cursor.execute(query, params)

        if commit:
            conn.commit()

        if fetch:
            result = cursor.fetchall()
        else:
            result = None

        cursor.close()
        conn.close()

        return result

    def create_tables_(self):
        query = """
            CREATE TABLE IF NOT EXISTS User (
                user_id VARCHAR(255) PRIMARY KEY,
                user_name VARCHAR(255)
            );
        """

        self.execute_query_(query, (), commit=True)
        query = """
            CREATE TABLE IF NOT EXISTS KnowledgeBase (
                kb_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255),
                kb_name VARCHAR(255),
                deleted INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES User(user_id) ON DELETE CASCADE
            );

        """
        self.execute_query_(query, (), commit=True)
        query = """
            CREATE TABLE IF NOT EXISTS File (
                file_id VARCHAR(255) PRIMARY KEY,
                kb_id VARCHAR(255),
                file_name VARCHAR(255),
                status VARCHAR(255),
                timestamp VARCHAR(255),
                deleted INTEGER DEFAULT 0,
                file_size INT DEFAULT -1,
                content_length INT DEFAULT -1,
                chunk_size INT DEFAULT -1,
                FOREIGN KEY (kb_id) REFERENCES KnowledgeBase(kb_id) ON DELETE CASCADE
            );

        """
        self.execute_query_(query, (), commit=True)

        query = """
            CREATE TABLE IF NOT EXISTS Document (
                docstore_id VARCHAR(64) PRIMARY KEY,
                chunk_id VARCHAR(64),
                file_id VARCHAR(64),
                file_name VARCHAR(640),
                kb_id VARCHAR(64)
            );
        """
        self.execute_query_(query, (), commit=True)

        query = """
            CREATE TABLE IF NOT EXISTS QanythingBot (
                bot_id          VARCHAR(64) PRIMARY KEY,
                user_id         VARCHAR(255),
                bot_name        VARCHAR(512),
                description     VARCHAR(512),
                head_image      VARCHAR(512),
                prompt_setting  LONGTEXT,
                welcome_message LONGTEXT,
                model           VARCHAR(100),
                kb_ids_str      VARCHAR(1024),
                deleted         INT DEFAULT 0,
                create_time     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                update_time     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
        self.execute_query_(query, (), commit=True)

        query = """
            CREATE TABLE IF NOT EXISTS Faqs (
                faq_id  VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                kb_id VARCHAR(255) NOT NULL,
                question VARCHAR(512) NOT NULL, 
                answer VARCHAR(2048) NOT NULL, 
                nos_keys VARCHAR(768) 
            );
        """
        self.execute_query_(query, (), commit=True)

    def check_user_exist_(self, user_id):
        query = "SELECT user_id FROM User WHERE user_id = ?"
        result = self.execute_query_(query, (user_id,), fetch=True)
        debug_logger.info("check_user_exist {}".format(result))
        return result is not None and len(result) > 0

    def check_kb_exist(self, user_id, kb_ids):
        # 使用参数化查询
        placeholders = ','.join(['?'] * len(kb_ids))
        query = "SELECT kb_id FROM KnowledgeBase WHERE kb_id IN ({}) AND deleted = 0 AND user_id = ?".format(
            placeholders)
        query_params = kb_ids + [user_id]
        result = self.execute_query_(query, query_params, fetch=True)
        debug_logger.info("check_kb_exist {}".format(result))
        valid_kb_ids = [kb_info[0] for kb_info in result]
        unvalid_kb_ids = list(set(kb_ids) - set(valid_kb_ids))
        return unvalid_kb_ids

    def check_bot_is_exist(self, user_id, bot_id):
        # 使用参数化查询
        query = "SELECT bot_id FROM QanythingBot WHERE bot_id = ? AND user_id = ? AND deleted = 0"
        result = self.execute_query_(query, (bot_id, user_id), fetch=True)
        debug_logger.info("check_bot_exist {}".format(result))
        return result is not None and len(result) > 0

    def get_file_by_status(self, kb_ids, status):
        # query = "SELECT file_name FROM File WHERE kb_id = ? AND deleted = 0 AND status = ?"
        kb_ids_str = ','.join("'{}'".format(str(x)) for x in kb_ids)
        query = "SELECT file_id, file_name FROM File WHERE kb_id IN ({}) AND deleted = 0 AND status = ?".format(kb_ids_str)
        result = self.execute_query_(query, (status,), fetch=True)
        # result = self.execute_query_(query, (kb_id, "gray"), fetch=True)
        return result

    def check_file_exist(self, user_id, kb_id, file_ids):
        # 筛选出有效的文件
        if not file_ids:
            debug_logger.info("check_file_exist skipped because of empty file_ids")
            return []
        
        file_ids_str = ','.join("'{}'".format(str(x)) for x in file_ids)
        query = """SELECT file_id, status FROM File 
                 WHERE deleted = 0
                 AND file_id IN ({})
                 AND kb_id = ? 
                 AND kb_id IN (SELECT kb_id FROM KnowledgeBase WHERE user_id = ?)""".format(file_ids_str)
        result = self.execute_query_(query, (kb_id, user_id), fetch=True)
        debug_logger.info("check_file_exist {}".format(result))
        return result

    def check_file_exist_by_name(self, user_id, kb_id, file_names):
        results = []
        batch_size = 100  # 根据实际情况调整批次大小

        # 分批处理file_names
        for i in range(0, len(file_names), batch_size):
            batch_file_names = file_names[i:i + batch_size]
            placeholders = ','.join(['?'] * len(batch_file_names))
            query = """
                SELECT file_id, file_name, file_size, status FROM File 
                WHERE deleted = 0
                AND file_name IN ({})
                AND kb_id = ? 
                AND kb_id IN (SELECT kb_id FROM KnowledgeBase WHERE user_id = ?)
            """.format(placeholders)

            query_params = batch_file_names + [kb_id, user_id]
            batch_result = self.execute_query_(query, query_params, fetch=True)
            debug_logger.info("check_file_exist_by_name batch {}: {}".format(i // batch_size, batch_result))
            results.extend(batch_result)

        return results

    # 对外接口不需要增加用户，新建知识库的时候增加用户就可以了
    def add_user_(self, user_id, user_name=None):
        query = "INSERT INTO User (user_id, user_name) VALUES (?, ?)"
        self.execute_query_(query, (user_id, user_name), commit=True)
        return user_id

    def add_document(self, docstore_id, chunk_id, file_id, file_name, kb_id):
        query = "INSERT INTO Document (docstore_id, chunk_id, file_id, file_name, kb_id) VALUES (?, ?, ?, ?, ?)"
        self.execute_query_(query, (docstore_id, chunk_id, file_id, file_name, kb_id), commit=True)

    def get_documents_by_kb_id(self, kb_id):
        query = "SELECT docstore_id FROM Document WHERE kb_id = ?"
        return self.execute_query_(query, (kb_id,), fetch=True)

    def get_documents_by_file_ids(self, file_ids):
        # 使用参数化查询
        placeholders = ','.join(['?'] * len(file_ids))
        query = "SELECT docstore_id FROM Document WHERE file_id IN ({})".format(placeholders)
        return self.execute_query_(query, file_ids, fetch=True)

    def new_knowledge_base(self, kb_id, user_id, kb_name, user_name=None):
        if not self.check_user_exist_(user_id):
            self.add_user_(user_id, user_name)
        query = "INSERT INTO KnowledgeBase (kb_id, user_id, kb_name) VALUES (?, ?, ?)"
        self.execute_query_(query, (kb_id, user_id, kb_name), commit=True)
        return kb_id, "success"

    def new_qanything_bot(self, bot_id, user_id, bot_name, description, head_image, prompt_setting, welcome_message, model, kb_ids_str):
        query = "INSERT INTO QanythingBot (bot_id, user_id, bot_name, description, head_image, prompt_setting, welcome_message, model, kb_ids_str) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        self.execute_query_(query, (bot_id, user_id, bot_name, description, head_image, prompt_setting, welcome_message, model, kb_ids_str), commit=True)
        return bot_id, "success"

    # [知识库] 获取指定用户的所有知识库 
    def get_knowledge_bases(self, user_id):
        query = "SELECT kb_id, kb_name FROM KnowledgeBase WHERE user_id = ? AND deleted = 0"
        return self.execute_query_(query, (user_id,), fetch=True)
    
    def get_users(self):
        query = "SELECT user_id FROM User"
        return self.execute_query_(query, (), fetch=True)

    # [知识库] 获取指定kb_ids的知识库
    def get_knowledge_base_name(self, kb_ids):
        # 使用参数化查询
        placeholders = ','.join(['?'] * len(kb_ids))
        query = "SELECT user_id, kb_id, kb_name FROM KnowledgeBase WHERE kb_id IN ({}) AND deleted = 0".format(placeholders)
        query_params = kb_ids
        return self.execute_query_(query, query_params, fetch=True)

    # [知识库] 删除指定知识库
    def delete_knowledge_base(self, user_id, kb_ids):
        # 使用参数化查询
        placeholders = ','.join(['?'] * len(kb_ids))
        query = "UPDATE KnowledgeBase SET deleted = 1 WHERE user_id = ? AND kb_id IN ({})".format(placeholders)
        query_params = [user_id] + kb_ids
        self.execute_query_(query, query_params, commit=True)

        # 更新文件的删除状态也需要使用参数化查询
        query = "UPDATE File SET deleted = 1 WHERE kb_id IN ({}) AND kb_id IN (SELECT kb_id FROM KnowledgeBase WHERE user_id = ?)".format(placeholders)
        debug_logger.info("delete_knowledge_base: {}".format(kb_ids))
        self.execute_query_(query, query_params, commit=True)

    def delete_bot(self, user_id, bot_id):
        # 使用参数化查询
        query = "UPDATE QanythingBot SET deleted = 1 WHERE user_id = ? AND bot_id = ?"
        self.execute_query_(query, (user_id, bot_id), commit=True)
    
    # [知识库] 重命名知识库
    def rename_knowledge_base(self, user_id, kb_id, kb_name):
        query = "UPDATE KnowledgeBase SET kb_name = ? WHERE kb_id = ? AND user_id = ?"
        debug_logger.info("rename_knowledge_base: {}".format(kb_id))
        self.execute_query_(query, (kb_name, kb_id, user_id), commit=True)

    # [文件] 向指定知识库下面增加文件
    def add_file(self, user_id, kb_id, file_name, timestamp, status="gray"):
        # 如果他传回来了一个id, 那就说明这个表里肯定有
        if not self.check_user_exist_(user_id):
            return None, "invalid user_id, please check..."
        not_exist_kb_ids = self.check_kb_exist(user_id, [kb_id])
        if not_exist_kb_ids:
            return None, f"invalid kb_id, please check {not_exist_kb_ids}"
        file_id = uuid.uuid4().hex
        query = "INSERT INTO File (file_id, kb_id, file_name, status, timestamp) VALUES (?, ?, ?, ?, ?)"
        self.execute_query_(query, (file_id, kb_id, file_name, status, timestamp), commit=True)
        debug_logger.info("add_file: {}".format(file_id))
        return file_id, "success"

    def add_faq(self, faq_id, user_id, kb_id, question, answer, nos_keys):
        # debug_logger.info(f"add_faq: {faq_id}, {user_id}, {kb_id}, {question}, {answer}, {nos_keys}")
        query = "INSERT INTO Faqs (faq_id, user_id, kb_id, question, answer, nos_keys) VALUES (?, ?, ?, ?, ?, ?)"
        self.execute_query_(query, (faq_id, user_id, kb_id, question, answer, nos_keys), commit=True)

    #  更新file中的file_size
    def update_file_size(self, file_id, file_size):
        query = "UPDATE File SET file_size = ? WHERE file_id = ?"
        self.execute_query_(query, (file_size, file_id), commit=True)
    
    #  更新file中的content_length
    def update_content_length(self, file_id, content_length):
        query = "UPDATE File SET content_length = ? WHERE file_id = ?"
        self.execute_query_(query, (content_length, file_id), commit=True)
    
    #  更新file中的chunk_size
    def update_chunk_size(self, file_id, chunk_size):
        query = "UPDATE File SET chunk_size = ? WHERE file_id = ?"
        self.execute_query_(query, (chunk_size, file_id), commit=True)

    def update_file_status(self, file_id, status):
        query = "UPDATE File SET status = ? WHERE file_id = ?"
        self.execute_query_(query, (status, file_id), commit=True)

    def from_status_to_status(self, file_ids, from_status, to_status):
        file_ids_str = ','.join("'{}'".format(str(x)) for x in file_ids)
        query = "UPDATE File SET status = ? WHERE file_id IN ({}) AND status = ?".format(file_ids_str)
        self.execute_query_(query, (to_status, from_status), commit=True)
        

    # [文件] 获取指定知识库下面所有文件的id和名称
    def get_files(self, user_id, kb_id):
        query = "SELECT file_id, file_name, status, file_size, content_length, timestamp FROM File WHERE kb_id = ? AND kb_id IN (SELECT kb_id FROM KnowledgeBase WHERE user_id = ?) AND deleted = 0"
        return self.execute_query_(query, (kb_id, user_id), fetch=True)

    # [文件] 删除指定文件
    def delete_files(self, kb_id, file_ids):
        file_ids_str = ','.join("'{}'".format(str(x)) for x in file_ids)
        query = "UPDATE File SET deleted = 1 WHERE kb_id = ? AND file_id IN ({})".format(file_ids_str)
        debug_logger.info("delete_files: {}".format(file_ids))
        self.execute_query_(query, (kb_id,), commit=True)

    def get_bot(self, user_id, bot_id):
        if not bot_id:
            query = "SELECT bot_id, bot_name, description, head_image, prompt_setting, welcome_message, model, kb_ids_str, update_time FROM QanythingBot WHERE user_id = ? AND deleted = 0"
            return self.execute_query_(query, (user_id,), fetch=True)
        else:
            query = "SELECT bot_id, bot_name, description, head_image, prompt_setting, welcome_message, model, kb_ids_str, update_time FROM QanythingBot WHERE user_id = ? AND bot_id = ? AND deleted = 0"
            return self.execute_query_(query, (user_id, bot_id), fetch=True)

    def update_bot(self, user_id, bot_id, bot_name, description, head_image, prompt_setting, welcome_message, model,
                   kb_ids_str, update_time):
        query = "UPDATE QanythingBot SET bot_name = ?, description = ?, head_image = ?, prompt_setting = ?, welcome_message = ?, model = ?, kb_ids_str = ?, update_time = ? WHERE user_id = ? AND bot_id = ? AND deleted = 0"
        self.execute_query_(query, (bot_name, description, head_image, prompt_setting, welcome_message, model, kb_ids_str, update_time, user_id, bot_id), commit=True)

    def get_faq(self, faq_id) -> tuple:
        query = "SELECT user_id, kb_id, question, answer, nos_keys FROM Faqs WHERE faq_id = ?"
        faq_all = self.execute_query_(query, (faq_id,), fetch=True)
        if faq_all:
            faq = faq_all[0]
            debug_logger.info(f"get_faq: faq_id: {faq_id}, mysql res: {faq}")
            return faq
        else:
            debug_logger.error(f"get_faq: faq_id: {faq_id} not found")
            return None

    def delete_faqs(self, faq_ids):
        # 分批，因为多个faq_id的加一起可能会超过sql的最大长度
        batch_size = 100
        total_deleted = 0
        for i in range(0, len(faq_ids), batch_size):
            batch_faq_ids = faq_ids[i:i+batch_size]
            placeholders = ','.join(['%s'] * len(batch_faq_ids))
            query = "DELETE FROM Faqs WHERE faq_id IN ({})".format(placeholders)
            res = self.execute_query_(query, (batch_faq_ids), commit=True, check=True)
            total_deleted += res
        debug_logger.info(f"delete_faqs count: {total_deleted}")


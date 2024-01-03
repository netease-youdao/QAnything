import mysql.connector
from mysql.connector import pooling


class KnowledgeBaseManager:
    def __init__(self, host='mysql-container-local', port='3306', user='root', password='123456', database='qanything'):
        self.check_database_(host, port, user, password, database)
        dbconfig = {
            "host": host,
            "user": user,
            "port": port,
            "password": password,
            "database": database,
        }
        self.cnxpool = pooling.MySQLConnectionPool(pool_size=5, pool_reset_session=True, **dbconfig)
        self.create_tables_()
        print("[SUCCESS] 数据库{}连接成功".format(database))

    def check_database_(self, host, port, user, password, database_name):
        # 连接 MySQL 服务器
        cnx = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password
        )

        # 检查数据库是否存在
        cursor = cnx.cursor(buffered=True)
        cursor.execute('SHOW DATABASES')
        databases = [database[0] for database in cursor]

        if database_name not in databases:
            # 如果数据库不存在，则新建数据库
            cursor.execute('CREATE DATABASE IF NOT EXISTS {}'.format(database_name))
            print("数据库{}新建成功或已存在".format(database_name))
        print("[SUCCESS] 数据库{}检查通过".format(database_name))
        # 关闭游标
        cursor.close()
        # 连接到数据库
        cnx.database = database_name
        # 关闭数据库连接
        cnx.close()

    def execute_query_(self, query, params, commit=False, fetch=False):
        conn = self.cnxpool.get_connection()
        cursor = conn.cursor(buffered=True)
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
                deleted BOOL DEFAULT 0,
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
                deleted BOOL DEFAULT 0,
                file_size INT DEFAULT -1,
                content_length INT DEFAULT -1,
                chunk_size INT DEFAULT -1,
                FOREIGN KEY (kb_id) REFERENCES KnowledgeBase(kb_id) ON DELETE CASCADE
            );

        """
        self.execute_query_(query, (), commit=True)

    def get_statistics_by_user(self, user_id):
        query = """
            SELECT SUM(file_size) AS total_file_size,
                   SUM(content_length) AS total_content_length,
                   SUM(chunk_size) AS total_chunk_size
            FROM File
            WHERE file_id IN (
                SELECT f.file_id
                FROM File f
                INNER JOIN KnowledgeBase kb ON f.kb_id = kb.kb_id
                WHERE kb.user_id = %s AND f.deleted = 0
            );
        """
        result = self.execute_query_(query, (user_id,), fetch=True)
        if result:
            return {
                'total_file_size': result[0][0] or 0,
                'total_content_length': result[0][1] or 0,
                'total_chunk_size': result[0][2] or 0
            }
        else:
            return None

    def get_statistics_by_user_and_kb(self, user_id, kb_ids):
        # 将kb_ids列表转换成字符串，用逗号分隔
        kb_ids_str = ','.join(["%s"] * len(kb_ids))
        query = f"""
            SELECT SUM(file_size) AS total_file_size,
                SUM(content_length) AS total_content_length,
                SUM(chunk_size) AS total_chunk_size,
                COUNT(file_id) AS total_files_count
            FROM File
            WHERE kb_id IN ({kb_ids_str}) AND deleted = 0
            AND kb_id IN (
                SELECT kb_id
                FROM KnowledgeBase
                WHERE user_id = %s AND deleted = 0
            );
        """
        # 合并user_id和kb_ids为一个参数列表
        params = kb_ids + [user_id]
        result = self.execute_query_(query, params, fetch=True)
        # 查询知识库名称
        kb_names_query = f"""
            SELECT kb_id, kb_name
            FROM KnowledgeBase
            WHERE kb_id IN ({kb_ids_str});
        """
        # 只需要传递 kb_ids 参数查询 kb_name
        kb_names_result = self.execute_query_(kb_names_query, kb_ids, fetch=True)

        # 构建 kb_id 到 kb_name 的映射
        kb_names = {kb_id: kb_name for kb_id, kb_name in kb_names_result}

        if result:
            return {
                'total_file_size': result[0][0] or 0,
                'total_content_length': result[0][1] or 0,
                'total_chunk_size': result[0][2] or 0,
                'total_files_count': result[0][3] or 0,
                'kb_names': kb_names  # 添加 kb_names 到返回结果
            }
        else:
            return None


def main(user_id, kb_ids=None):
    # 创建 KnowledgeBaseManager 的实例
    manager = KnowledgeBaseManager()

    # 根据输入判断调用哪个函数
    if kb_ids:
        # 如果提供了 kb_id，则获取特定 user_id 和 kb_id 的统计信息
        stats = manager.get_statistics_by_user_and_kb(user_id, kb_ids)
    else:
        # 如果没有提供 kb_id，只根据 user_id 获取统计信息
        stats = manager.get_statistics_by_user(user_id)

    # 打印统计信息
    if stats:
        print(f"用户ID: {user_id}")
        # if kb_ids:
        #     print(f"知识库ID: {kb_ids}")
        # stats['total_file_size']是字节，换算成MB，保留3位小数
        stats['total_file_size'] = round(stats['total_file_size'] / 1024 / 1024, 3)
        print(stats['kb_names'])
        print(f"总文件数量: {stats['total_files_count']}")
        print(f"总文件大小: {stats['total_file_size']}MB")
        print(f"总内容长度（字数）: {stats['total_content_length'] / 10000}万字")
        print(f"总chunk数量: {stats['total_chunk_size']}")
        # 平均每个chunk对应的字数
        print(f"平均每个chunk对应的字数: {stats['total_content_length'] / stats['total_chunk_size']}")
    else:
        print("没有找到统计信息。")


# 示例调用
if __name__ == "__main__":
    # 这里需要替换为真实的 user_id 和 kb_id
    user_id_input = "liujx_265"
    kb_ids_input = ["KBc2440f13e98f4736b5ef81cfaebef3a9", "KB6c2b097d83be430ab809e361fa8dcc8b",
                    "KB69331d593f5b4b5bb555a0ea1b145e5b", "KB3cdc79f8c8d24a14bffd27e6570c33da",
                    "KBb78af28c73f74fb4ae6ad44b3c53302f", "KBf46828db208c4289a120a34f0fc96147"]  # 或者 None 如果没有 kb_id

    main(user_id_input, kb_ids_input)

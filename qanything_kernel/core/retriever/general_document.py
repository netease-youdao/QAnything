from qanything_kernel.utils.general_utils import get_time_async, num_tokens, get_time, get_table_infos
from typing import List
from qanything_kernel.configs.model_config import UPLOAD_ROOT_PATH, LOCAL_OCR_SERVICE_URL, PARENT_CHUNK_SIZE
from langchain.docstore.document import Document
from qanything_kernel.utils.loader.my_recursive_url_loader import MyRecursiveUrlLoader
from qanything_kernel.utils.custom_log import insert_logger
from langchain_community.document_loaders import UnstructuredFileLoader, TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredEmailLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from qanything_kernel.utils.loader import UnstructuredPaddlePDFLoader
from qanything_kernel.utils.loader.self_pdf_loader import PdfLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qanything_kernel.utils.loader.csv_loader import CSVLoader
from qanything_kernel.utils.loader.markdown_parser import convert_markdown_to_langchaindoc
from qanything_kernel.utils.loader.pdf_data_parser import \
    convert_markdown_to_langchaindoc as convert_pdf_data_to_langchaindoc
import base64
import pandas as pd
import os
import json
import requests
import threading
import re
import newspaper
import shutil
import uuid
import traceback


def get_ocr_result_sync(image_data):
    try:
        response = requests.post(f"http://{LOCAL_OCR_SERVICE_URL}/ocr", data=image_data)
        response.raise_for_status()  # 如果请求返回了错误状态码，将会抛出异常
        ocr_res = response.text
        # insert_logger.info(f"ocr_res[:100]: {ocr_res[:100]}")
        ocr_res = json.loads(ocr_res)
        return ocr_res['result']
    except Exception as e:
        insert_logger.warning(f"ocr error: {traceback.format_exc()}")
        return None


def get_pdf_to_markdown(file_path, file_id):
    try:
        base64_pdf_file = base64.b64encode(open(file_path, 'rb').read()).decode()
        data = {"pdf_data": base64_pdf_file, "uuid": file_id}
        response = requests.post('http://rag-parse.inner.youdao.com/parse_pdf', json=data, timeout=480)
        response.raise_for_status()  # 如果请求返回了错误状态码，将会抛出异常
        pdf2markdown_res = response.text
        pdf2markdown_res = json.loads(pdf2markdown_res)
        if pdf2markdown_res['status'] != 'success':
            insert_logger.warning(f"pdf2markdown_res error: {data['uuid']}")
            return None
        data = json.loads(pdf2markdown_res['res_data'])
        insert_logger.info(f"pdf2markdown_res: {data['pages'][:1]}")
        return data['pages']
    except Exception as e:
        insert_logger.warning(f"pdf2markdown_res error: {traceback.format_exc()}")
        # file_path文件复制到UPLOAD_ROOT_PATH下的error_pdfs文件夹
        error_pdf_dir = os.path.join(UPLOAD_ROOT_PATH, 'error_pdfs')
        os.makedirs(error_pdf_dir, exist_ok=True)
        shutil.copy(file_path, error_pdf_dir)
        return None


def get_word_to_markdown(file_path, file_id):
    try:
        files = {'docx_file': open(file_path, 'rb')}
        data = {"uuid": file_id}
        response = requests.post('http://rag-parse.inner.youdao.com/parse_docx', files=files, data=data)
        response.raise_for_status()  # 如果请求返回了错误状态码，将会抛出异常
        word2markdown_res = response.text
        word2markdown_res = json.loads(word2markdown_res)
        if word2markdown_res['status'] != 'success':
            insert_logger.warning(f"word2markdown_res error: {data['uuid']}")
            return None
        return word2markdown_res['res_data']
    except Exception as e:
        insert_logger.warning(f"word2markdown_res error: {traceback.format_exc()}")
        # file_path文件复制到UPLOAD_ROOT_PATH下的error_pdfs文件夹
        error_docx_dir = os.path.join(UPLOAD_ROOT_PATH, 'error_docxs')
        os.makedirs(error_docx_dir, exist_ok=True)
        shutil.copy(file_path, error_docx_dir)
        return None


pdf_text_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=0,
                                                   length_function=num_tokens)


class LocalFileForInsert:
    def __init__(self, user_id, kb_id, file_id, file_location, file_name, file_url, mysql_client):
        self.user_id = user_id
        self.kb_id = kb_id
        self.file_id = file_id
        self.docs: List[Document] = []
        self.embs = []
        self.file_name = file_name
        self.file_location = file_location
        self.file_url = ""
        self.faq_dict = {}
        self.error = None
        self.file_path = ""
        self.mysql_client = mysql_client
        if self.file_location == 'FAQ':
            faq_info = self.mysql_client.get_faq(self.file_id)
            user_id, kb_id, question, answer, nos_keys = faq_info
            self.faq_dict = {'question': question, 'answer': answer, 'nos_keys': nos_keys}
        elif self.file_location == 'URL':
            self.file_url = file_url
            upload_path = os.path.join(UPLOAD_ROOT_PATH, user_id)
            file_dir = os.path.join(upload_path, self.kb_id, self.file_id)
            os.makedirs(file_dir, exist_ok=True)
            self.file_path = os.path.join(file_dir, self.file_name)
        else:
            self.file_path = self.file_location
        self.event = threading.Event()

    @staticmethod
    @get_time
    def image_ocr_txt(filepath, dir_path="tmp_files"):
        full_dir_path = os.path.join(os.path.dirname(filepath), dir_path)
        if not os.path.exists(full_dir_path):
            os.makedirs(full_dir_path)
        filename = os.path.split(filepath)[-1]

        # 读取图片
        img_np = open(filepath, 'rb').read()

        img_data = {
            "img64": base64.b64encode(img_np).decode("utf-8"),
        }

        result = get_ocr_result_sync(img_data)

        ocr_result = [line for line in result if line]
        ocr_result = '\n'.join(ocr_result)

        insert_logger.info(f'ocr_res[:100]: {ocr_result[:100]}')

        # 写入结果到文本文件
        txt_file_path = os.path.join(full_dir_path, "%s.txt" % (filename))
        with open(txt_file_path, 'w', encoding='utf-8') as fout:
            fout.write(ocr_result)

        return txt_file_path

    @staticmethod
    def pdf_process(docs: List[Document]):
        # pdf解析后结果太碎片了，需要合并处理，优先合并doc.metadata['page_id']一致的doc，再判断page_content长度是否大于阈值，如果小于就合并两页
        merged_docs = []
        # 先合页
        for i, doc in enumerate(docs):
            if i == 0:
                merged_docs.append(doc)
            else:
                if doc.metadata['source_info']['page_id'] == merged_docs[-1].metadata['source_info']['page_id']:
                    merged_docs[-1].page_content += '\n' + doc.page_content
                elif doc.metadata['source_info']['page_id'] > merged_docs[-1].metadata['source_info']['page_id']:
                    merged_docs.append(doc)
                else:  # 后续的重叠文本不需要了
                    break

        # TODO 后续需要更复杂的逻辑去优化
        insert_logger.info(f"pdf_process merged_docs: {len(docs)}->{len(merged_docs)}")
        return merged_docs

    @staticmethod
    def table_process(doc):
        table_infos = get_table_infos(doc.page_content)
        title_lst = doc.metadata['title_lst']
        new_docs = []
        if table_infos is not None:
            tmp_content = '\n'.join(title_lst) + '\n' + doc.page_content
            if num_tokens(tmp_content) <= PARENT_CHUNK_SIZE:
                doc.page_content = tmp_content
                return [doc]
            head_line = table_infos['head_line']
            end_line = table_infos['end_line']
            table_head = table_infos['head']
            if head_line != 0:
                tmp_doc = Document(
                    page_content='\n'.join(title_lst) + '\n' + '\n'.join(table_infos['lines'][:head_line]),
                    metadata=doc.metadata)
                new_docs.append(tmp_doc)
            # 将head_line + 2:end_line的表格切分，每PARENT_CHUNK_SIZE的长度切分一个doc
            tmp_doc = '\n'.join(title_lst) + '\n' + table_head
            for line in table_infos['lines'][head_line + 2:end_line + 1]:
                tmp_doc += '\n' + line
                if num_tokens(tmp_doc) + num_tokens(line) > PARENT_CHUNK_SIZE:
                    tmp_doc = Document(page_content=tmp_doc, metadata=doc.metadata)
                    new_docs.append(tmp_doc)
                    tmp_doc = '\n'.join(title_lst) + '\n' + table_head
            if tmp_doc != '\n'.join(title_lst) + '\n' + table_head:
                tmp_doc = Document(page_content=tmp_doc, metadata=doc.metadata)
                new_docs.append(tmp_doc)
            if end_line != len(table_infos['lines']) - 1:
                tmp_doc = Document(
                    page_content='\n'.join(title_lst) + '\n' + '\n'.join(table_infos['lines'][end_line:]),
                    metadata=doc.metadata)
                new_docs.append(tmp_doc)
            insert_logger.info(f"TABLE SLICES: {new_docs[:2]}")
        else:
            return None
        return new_docs

    @staticmethod
    def get_page_id(doc, pre_page_id):
        # 查找 page_id 标志行
        lines = doc.page_content.split('\n')
        for line in lines:
            if re.match(r'^#+ 当前页数:\d+$', line):
                try:
                    page_id = int(line.split(':')[-1])
                    return page_id
                except ValueError:
                    continue
        return pre_page_id

    @staticmethod
    def markdown_process(docs: List[Document]):
        new_docs = []
        for doc in docs:
            if 'coord_lst' in doc.metadata:
                content_list = [para for para in doc.metadata['coord_lst'] if para[2] == 'content']
                if content_list:
                    doc.metadata['page_id'] = content_list[0][0]
                    doc.metadata['bbox'] = content_list[0][1]  # (x,y,w,h)
                else:
                    doc.metadata['page_id'] = doc.metadata['coord_lst'][0][0]
                    doc.metadata['bbox'] = doc.metadata['coord_lst'][0][1]  # (x,y,w,h)

            title_lst = doc.metadata['title_lst']
            # 删除所有仅有多个#的title
            title_lst = [t for t in title_lst if t.replace('#', '') != '']
            has_table = doc.metadata['has_table']
            if has_table:
                table_docs = LocalFileForInsert.table_process(doc)
                if table_docs:
                    new_docs.extend(table_docs)
                    continue
            if doc.page_content is "":  # page_content为空时把title_lst当做正文
                cleaned_list = [re.sub(r'^#+\s*', '', item) for item in title_lst]
                doc.page_content = '\n'.join(cleaned_list)
                doc.metadata['title_lst'] = []  # 清空title_lst
                slices = pdf_text_splitter.split_documents([doc])
                new_docs.extend(slices)
            else: 
                slices = pdf_text_splitter.split_documents([doc])
                # insert_logger.info(f"pdf_text_splitter: {len(slices)}")
                if len(slices) == 1:
                    slices[0].page_content = '\n\n'.join(title_lst) + '\n\n' + slices[0].page_content
                else:
                    for idx, slice in enumerate(slices):
                        slice.page_content = '\n\n'.join(title_lst) + f'\n\n###### 第{idx+1}段内容如下：\n' + slice.page_content
                new_docs.extend(slices)
        return new_docs

    def set_file_images(self, docs):
        for doc in docs:
            lines = doc.page_content.split('\n')
            image_lines = [(idx, line) for idx, line in enumerate(lines) if line.startswith('![figure]')]
            if not image_lines:
                continue
            # 把figure后面的nos_key提取出来
            for idx, image_line in image_lines:
                nos_key = image_line.split('](')[1].split(')')[0]
                image_uuid = uuid.uuid4().hex
                lines[idx] = f'![figure]({image_uuid})'
                self.mysql_client.add_file_images(image_uuid, self.file_id, self.user_id, self.kb_id, nos_key)
            doc.page_content = '\n'.join(lines)
            doc.metadata['images_number'] = len(image_lines)
            insert_logger.info(f"set_file_images: {doc.metadata['images_number']}")

    @staticmethod
    @get_time
    def pdf_to_documents(file_path, file_name, file_id, dir_path="tmp_files"):
        full_dir_path = os.path.join(os.path.dirname(file_path), dir_path)
        if not os.path.exists(full_dir_path):
            os.makedirs(full_dir_path)
        pages = get_pdf_to_markdown(file_path, file_id)
        if not pages:
            return None
        pages = sorted(pages, key=lambda x: x['page_id'])

        if pages:
            json_file_path = os.path.join(full_dir_path, "%s_mark.json" % (file_name))
            # 写入json文件
            with open(json_file_path, 'w', encoding='utf-8') as fout:
                json.dump(pages, fout, ensure_ascii=False)
            try:
                docs = convert_pdf_data_to_langchaindoc(file_name, pages)
                docs = LocalFileForInsert.markdown_process(docs)
                return docs
            except Exception as e:
                insert_logger.error(f"convert_pdf_data_to_langchaindoc error: {file_path}, {traceback.format_exc()}")
                return None
        else:
            return None

    @staticmethod
    @get_time
    def word_to_documents(file_path, file_name, file_id, dir_path="tmp_files"):
        full_dir_path = os.path.join(os.path.dirname(file_path), dir_path)
        if not os.path.exists(full_dir_path):
            os.makedirs(full_dir_path)
        markdown_str = get_word_to_markdown(file_path, file_id)
        if markdown_str is not None:
            md_file_path = os.path.join(full_dir_path, "%s.md" % (file_name))
            with open(md_file_path, 'w', encoding='utf-8') as fout:
                fout.write(markdown_str)
            try:
                docs = convert_markdown_to_langchaindoc(md_file_path)
                docs = LocalFileForInsert.markdown_process(docs)
            except Exception as e:
                insert_logger.error(f"convert_markdown_to_langchaindoc error: {file_path}, {traceback.format_exc()}")
                return None 
            return docs
        return None

    @staticmethod
    @get_time
    def url_to_documents(file_path, file_name, file_url, dir_path="tmp_files"):
        full_dir_path = os.path.join(os.path.dirname(file_path), dir_path)
        if not os.path.exists(full_dir_path):
            os.makedirs(full_dir_path)
        try:
            jina_response = requests.get(f"https://r.jina.ai/{file_url}", timeout=120)
            if jina_response.status_code == 200:
                parts = jina_response.text.split('\n')
                metadata = {}
                for part in parts:
                    if 'Title: ' in part:
                        metadata['title'] = part.replace('Title: ', '')
                    elif 'URL Source: ' in part:
                        metadata['url_source'] = part.replace('URL Source: ', '')
                    elif 'Published Time: ' in part:
                        metadata['published_time'] = part.replace('Published Time: ', '')
                    elif 'Markdown Content\n:' in part:
                        break
                markdown_str = jina_response.text.split("Markdown Content\n:")[-1]
                md_file_path = os.path.join(full_dir_path, "%s.md" % (file_name))
                with open(md_file_path, 'w', encoding='utf-8') as fout:
                    fout.write(markdown_str)
                docs = convert_markdown_to_langchaindoc(md_file_path)
                docs = LocalFileForInsert.markdown_process(docs)
                return docs
            else:
                insert_logger.warning(f"jina get url error: {file_url}, {jina_response.text}")
        except Exception as e:
            insert_logger.warning(f"jina get url error: {file_url}, {traceback.format_exc()}")
        return None

    @get_time
    def split_file_to_docs(self):
        insert_logger.info(f"start split file to docs, file_path: {self.file_name}")
        if self.faq_dict:
            docs = [Document(page_content=self.faq_dict['question'], metadata={"faq_dict": self.faq_dict})]
        elif self.file_url:
            insert_logger.info("load url: {}".format(self.file_url))
            docs = self.url_to_documents(self.file_path, self.file_name, self.file_url)
            if docs is None:
                try:
                    article = newspaper.article(self.file_url, timeout=120)
                    docs = [Document(page_content=article.text, metadata={"title": article.title, "url": self.file_url})]
                except Exception as e:
                    insert_logger.error(f"newspaper get url error: {self.file_url}, {traceback.format_exc()}")
                    loader = MyRecursiveUrlLoader(url=self.file_url)
                    docs = loader.load()
        elif self.file_path.lower().endswith(".md"):
            try:
                docs = convert_markdown_to_langchaindoc(self.file_path)
                docs = self.markdown_process(docs)
            except Exception as e:
                insert_logger.error(f"convert_markdown_to_langchaindoc error: {self.file_path}, {traceback.format_exc()}")
                loader = UnstructuredFileLoader(self.file_path, strategy="fast")
                docs = loader.load()
        elif self.file_path.lower().endswith(".txt"):
            loader = TextLoader(self.file_path, autodetect_encoding=True)
            docs = loader.load()
        elif self.file_path.lower().endswith(".pdf"):
            try:
                loader = PdfLoader(filename=self.file_path, save_dir=os.path.dirname(self.file_path))
                markdown_dir = loader.load_to_markdown()
                docs = convert_markdown_to_langchaindoc(markdown_dir)
                docs = self.markdown_process(docs)
            except Exception as e:
                insert_logger.warning(f'Error in Powerful PDF parsing: {traceback.format_exc()}, use fast PDF parser instead.')
                loader = UnstructuredPaddlePDFLoader(self.file_path, strategy="fast")
                docs = loader.load()
        elif self.file_path.lower().endswith(".jpg") or self.file_path.lower().endswith(
                ".png") or self.file_path.lower().endswith(".jpeg"):
            txt_file_path = self.image_ocr_txt(filepath=self.file_path)
            loader = TextLoader(txt_file_path, autodetect_encoding=True)
            docs = loader.load()
        elif self.file_path.lower().endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(self.file_path, strategy="fast")
            docs = loader.load()
        elif self.file_path.lower().endswith(".xlsx"):
            docs = []
            excel_file = pd.ExcelFile(self.file_path)
            sheet_names = excel_file.sheet_names
            for idx, sheet_name in enumerate(sheet_names):
                xlsx = pd.read_excel(self.file_path, sheet_name=sheet_name, engine='openpyxl')
                csv_file_path = self.file_path[:-5] + f'_{idx}.csv'
                xlsx.to_csv(csv_file_path, index=False)
                insert_logger.info('xlsx2csv: %s', csv_file_path)
                loader = CSVLoader(csv_file_path, autodetect_encoding=True,
                                   csv_args={"delimiter": ",", "quotechar": '"'})
                docs.extend(loader.load())
        elif self.file_path.lower().endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(self.file_path, strategy="fast")
            docs = loader.load()
        elif self.file_path.lower().endswith(".eml"):
            loader = UnstructuredEmailLoader(self.file_path, strategy="fast")
            docs = loader.load()
        elif self.file_path.lower().endswith(".csv"):
            loader = CSVLoader(self.file_path, autodetect_encoding=True, csv_args={"delimiter": ",", "quotechar": '"'})
            docs = loader.load()
        else:
            raise TypeError("文件类型不支持，目前仅支持：[md,txt,pdf,jpg,png,jpeg,docx,xlsx,pptx,eml,csv]")

        self.inject_metadata(docs)

    def inject_metadata(self, docs: List[Document]):
        # 这里给每个docs片段的metadata里注入file_id
        new_docs = []
        for doc in docs:
            page_content = re.sub(r'[\n\t]+', '\n', doc.page_content).strip()
            new_doc = Document(page_content=page_content)
            new_doc.metadata["user_id"] = self.user_id
            new_doc.metadata["kb_id"] = self.kb_id
            new_doc.metadata["file_id"] = self.file_id
            new_doc.metadata["file_name"] = self.file_name
            new_doc.metadata["nos_key"] = self.file_location
            new_doc.metadata["file_url"] = self.file_url
            new_doc.metadata["title_lst"] = doc.metadata.get("title_lst", [])
            new_doc.metadata["has_table"] = doc.metadata.get("has_table", False)
            new_doc.metadata["images_number"] = doc.metadata.get("images_number", 0)

            if 'faq_dict' not in doc.metadata:
                new_doc.metadata['faq_dict'] = {}
            else:
                new_doc.metadata['faq_dict'] = doc.metadata['faq_dict']
            if 'source_info' not in doc.metadata:
                new_doc.metadata['source_info'] = {'page_id': doc.metadata.get('page_id', 0),
                                                   'pdf_nos_key': self.file_location, 'chunk_id': -1,
                                                   'doc_id': self.file_id, 'bbox': doc.metadata.get('bbox', None)}
            else:
                new_doc.metadata['source_info'] = doc.metadata['source_info']
                # insert_logger.info(f"pdf_source_info: {new_doc.metadata['source_info']}")
            new_docs.append(new_doc)
        if new_docs:
            insert_logger.info('langchain analysis content head: %s', new_docs[0].page_content[:100])
        else:
            insert_logger.info('langchain analysis docs is empty!')
        self.docs = new_docs

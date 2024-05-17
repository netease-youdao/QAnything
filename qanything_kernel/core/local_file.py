from qanything_kernel.utils.general_utils import *
from typing import List, Union, Callable
from qanything_kernel.configs.model_config import UPLOAD_ROOT_PATH, SENTENCE_SIZE, ZH_TITLE_ENHANCE, USE_FAST_PDF_PARSER
from langchain.docstore.document import Document
from qanything_kernel.utils.loader.my_recursive_url_loader import MyRecursiveUrlLoader
from langchain_community.document_loaders import UnstructuredFileLoader, TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredEmailLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from qanything_kernel.utils.loader.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qanything_kernel.utils.custom_log import debug_logger, qa_logger
from qanything_kernel.utils.splitter import ChineseTextSplitter
from qanything_kernel.utils.loader import UnstructuredPaddleImageLoader, UnstructuredPaddlePDFLoader, UnstructuredPaddleAudioLoader
from qanything_kernel.utils.splitter import zh_title_enhance
from qanything_kernel.utils.loader.self_pdf_loader import PdfLoader
from qanything_kernel.utils.loader.markdown_parser import convert_markdown_to_langchaindoc
from sanic.request import File
import pandas as pd
import os
import re

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " ", ""],
    chunk_size=400,
    chunk_overlap=100,
    length_function=num_tokens,
)

pdf_text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200, length_function=num_tokens)


class LocalFile:
    def __init__(self, user_id, kb_id, file: Union[File, str, dict], file_id, file_name, embedding, is_url=False, in_milvus=False):
        self.user_id = user_id
        self.kb_id = kb_id
        self.file_id = file_id
        self.docs: List[Document] = []
        self.embs = []
        self.emb_infer = embedding
        self.use_cpu = embedding.use_cpu
        self.url = None
        self.in_milvus = in_milvus
        self.file_name = file_name
        if is_url:
            self.url = file
            self.file_path = "URL"
            self.file_content = b''
        elif isinstance(file, dict):
            self.file_path = "FAQ"
            self.file_content = file
        else:
            if isinstance(file, str):
                self.file_path = file
                with open(file, 'rb') as f:
                    self.file_content = f.read()
            else:
                upload_path = os.path.join(UPLOAD_ROOT_PATH, user_id)
                file_dir = os.path.join(upload_path, self.file_id)
                os.makedirs(file_dir, exist_ok=True)
                self.file_path = os.path.join(file_dir, self.file_name)
                self.file_content = file.body
            with open(self.file_path, "wb+") as f:
                f.write(self.file_content)
        debug_logger.info(f'success init localfile {self.file_name}')

    @staticmethod
    def pdf_process(dos: List[Document]):
        new_docs = []
        for doc in dos:
            # metadata={'title_lst': ['#樊昊天个人简历', '##教育经历'], 'has_table': False}
            title_lst = doc.metadata['title_lst']
            # 删除所有仅有多个#的title
            title_lst = [t for t in title_lst if t.replace('#', '') != '']
            has_table = doc.metadata['has_table']
            if has_table:
                doc.page_content = '\n'.join(title_lst) + '\n本段为表格，内容如下：' + doc.page_content
                new_docs.append(doc)
                continue
            # doc.page_content = '\n'.join(title_lst) + '\n' + doc.page_content
            slices = pdf_text_splitter.split_documents([doc])
            for idx, slice in enumerate(slices):
                slice.page_content = '\n'.join(title_lst) + f'\n######第{idx+1}段内容如下：\n' + slice.page_content
            new_docs.extend(slices)
        return new_docs

    @get_time
    def split_file_to_docs(self, ocr_engine: Callable, sentence_size=SENTENCE_SIZE,
                           using_zh_title_enhance=ZH_TITLE_ENHANCE):
        if self.url:
            debug_logger.info("load url: {}".format(self.url))
            loader = MyRecursiveUrlLoader(url=self.url)
            textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
            docs = loader.load_and_split(text_splitter=textsplitter)
        elif self.file_path == 'FAQ':
            docs = [Document(page_content=self.file_content['question'], metadata={"faq_dict": self.file_content})]
        elif self.file_path.lower().endswith(".md"):
            loader = UnstructuredFileLoader(self.file_path)
            docs = loader.load()
        elif self.file_path.lower().endswith(".txt"):
            loader = TextLoader(self.file_path, autodetect_encoding=True)
            texts_splitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
            docs = loader.load_and_split(texts_splitter)
        elif self.file_path.lower().endswith(".pdf"):
            if USE_FAST_PDF_PARSER:
                loader = UnstructuredPaddlePDFLoader(self.file_path, ocr_engine, self.use_cpu)
                texts_splitter = ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
                docs = loader.load_and_split(texts_splitter)
            else:
                loader = PdfLoader(filename=self.file_path, root_dir=os.path.dirname(self.file_path))
                markdown_dir = loader.load_to_markdown()
                docs = convert_markdown_to_langchaindoc(markdown_dir)
                docs = self.pdf_process(docs)
                # print(docs)
        elif self.file_path.lower().endswith(".jpg") or self.file_path.lower().endswith(
                ".png") or self.file_path.lower().endswith(".jpeg"):
            loader = UnstructuredPaddleImageLoader(self.file_path, ocr_engine, self.use_cpu)
            texts_splitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
            docs = loader.load_and_split(text_splitter=texts_splitter)
        elif self.file_path.lower().endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(self.file_path)
            texts_splitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
            docs = loader.load_and_split(texts_splitter)
        elif self.file_path.lower().endswith(".xlsx"):
            # loader = UnstructuredExcelLoader(self.file_path, mode="elements")
            csv_file_path = self.file_path[:-5] + '.csv'
            xlsx = pd.read_excel(self.file_path, engine='openpyxl')
            xlsx.to_csv(csv_file_path, index=False)
            loader = CSVLoader(csv_file_path, csv_args={"delimiter": ",", "quotechar": '"'})
            docs = loader.load()
        elif self.file_path.lower().endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(self.file_path)
            docs = loader.load()
        elif self.file_path.lower().endswith(".eml"):
            loader = UnstructuredEmailLoader(self.file_path)
            docs = loader.load()
        elif self.file_path.lower().endswith(".csv"):
            loader = CSVLoader(self.file_path, csv_args={"delimiter": ",", "quotechar": '"'})
            docs = loader.load()
        elif self.file_path.lower().endswith(".mp3") or self.file_path.lower().endswith(".wav"):
            loader = UnstructuredPaddleAudioLoader(self.file_path, self.use_cpu)
            docs = loader.load()
        else:
            debug_logger.info("file_path: {}".format(self.file_path))
            raise TypeError("文件类型不支持，目前仅支持：[md,txt,pdf,jpg,png,jpeg,docx,xlsx,pptx,eml,csv]")
        if using_zh_title_enhance:
            debug_logger.info("using_zh_title_enhance %s", using_zh_title_enhance)
            docs = zh_title_enhance(docs)
        print('docs number:', len(docs))
        # print(docs)
        # 不是csv，xlsx和FAQ的文件，需要再次分割
        if not self.file_path.lower().endswith(".csv") and not self.file_path.lower().endswith(".xlsx") and not self.file_path == 'FAQ':
            new_docs = []
            min_length = 200
            for doc in docs:
                if not new_docs:
                    new_docs.append(doc)
                else:
                    last_doc = new_docs[-1]
                    if len(last_doc.page_content) + len(doc.page_content) < min_length:
                        last_doc.page_content += '\n' + doc.page_content
                    else:
                        new_docs.append(doc)
            debug_logger.info(f"before 2nd split doc lens: {len(new_docs)}")
            if self.file_path.lower().endswith(".pdf"):
                if USE_FAST_PDF_PARSER:
                    docs = pdf_text_splitter.split_documents(new_docs)
            else:
                docs = text_splitter.split_documents(new_docs)
            debug_logger.info(f"after 2nd split doc lens: {len(docs)}")

        # 这里给每个docs片段的metadata里注入file_id
        new_docs = []
        for idx, doc in enumerate(docs):
            page_content = re.sub(r'[\n\t]+', '\n', doc.page_content).strip()
            new_doc = Document(page_content=page_content)
            new_doc.metadata["user_id"] = self.user_id
            new_doc.metadata["kb_id"] = self.kb_id
            new_doc.metadata["file_id"] = self.file_id
            new_doc.metadata["file_name"] = self.url if self.url else self.file_name
            new_doc.metadata["chunk_id"] = idx
            new_doc.metadata["file_path"] = self.file_path
            if 'faq_dict' not in doc.metadata:
                new_doc.metadata['faq_dict'] = {}
            else:
                new_doc.metadata['faq_dict'] = doc.metadata['faq_dict']
            new_docs.append(new_doc)

        if new_docs:
            debug_logger.info('langchain analysis content head: %s', new_docs[0].page_content[:100])
        else:
            debug_logger.info('langchain analysis docs is empty!')
        self.docs = new_docs

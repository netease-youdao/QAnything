"""Loader that loads image files."""
from typing import List, Callable

from langchain.document_loaders.unstructured import UnstructuredFileLoader
from unstructured.partition.text import partition_text
import os
import fitz
from tqdm import tqdm
from typing import Union, Any
import numpy as np
import cv2


class UnstructuredPaddlePDFLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load image files, such as PNGs and JPGs."""
    def __init__(
        self,
        file_path: Union[str, List[str]],
        mode: str = "single",
        ocr_engine: Callable[[Any], List[str]] = None,
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""
        self.ocr_engine = ocr_engine
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        def pdf_ocr_txt(filepath, dir_path="tmp_files"):
            full_dir_path = os.path.join(os.path.dirname(filepath), dir_path)
            if not os.path.exists(full_dir_path):
                os.makedirs(full_dir_path)
            doc = fitz.open(filepath)
            txt_file_path = os.path.join(full_dir_path, "{}.txt".format(os.path.split(filepath)[-1]))
            with open(txt_file_path, 'w', encoding='utf-8') as fout:
                for i in tqdm(range(doc.page_count)):
                    page = doc.load_page(i)
                    result = page.get_text()
                    # 如果页面中有图片或 OCR 返回空，需要检查图片内容
                    pix = page.get_pixmap(dpi=300)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))
                    
                    _, img_encoded = cv2.imencode('.png', img)
                    img_bytes = img_encoded.tobytes()
                    files = {'file': ('image.png', img_bytes, 'image/png')}

                    # 调用 OCR 引擎处理图片内容
                    ocr_result = self.ocr_engine(files)
                    # OCR 引擎可能返回 None，需要合理处理
                    if ocr_result is not None:
                        ocr_result = '\n'.join([line for line in ocr_result if line])
                        # 如果文本内容和 OCR 内容同时存在，拼接两部分
                        if result.strip():  # 页面有文本内容
                            result = result.strip() + '\n' + ocr_result.strip()
                        else:  # 仅有 OCR 内容
                            result = ocr_result.strip()

                    # 如果 OCR 也未返回任何内容，保留当前的文本内容（可能为空）
                    result = result.strip()
                    fout.write(result + '\n\n')

            return txt_file_path

        txt_file_path = pdf_ocr_txt(self.file_path)
        return partition_text(filename=txt_file_path, **self.unstructured_kwargs)

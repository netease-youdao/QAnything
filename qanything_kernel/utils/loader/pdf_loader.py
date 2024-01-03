"""Loader that loads image files."""
from typing import List, Callable

from langchain.document_loaders.unstructured import UnstructuredFileLoader
import os
import fitz
from tqdm import tqdm
from typing import Union, Any


class UnstructuredPaddlePDFLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load image files, such as PNGs and JPGs."""
    def __init__(
        self,
        file_path: Union[str, List[str]],
        ocr_engine: Callable,
        mode: str = "single",
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
            img_name = os.path.join(full_dir_path, 'tmp.png')
            with open(txt_file_path, 'w', encoding='utf-8') as fout:
                for i in tqdm(range(doc.page_count)):
                    page = doc[i]
                    text = page.get_text("")
                    fout.write(text)
                    fout.write("\n")

                    img_list = page.get_images()
                    for img in img_list:
                        pix = fitz.Pixmap(doc, img[0])
                        if pix.n - pix.alpha >= 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        pix.save(img_name)

                        result = self.ocr_engine(img_name)
                        result = [line for line in result if line]
                        ocr_result = [i[1][0] for line in result for i in line]
                        # with open(txt_file_path, 'w', encoding='utf-8') as fout:
                        #     fout.write("\n".join(ocr_result))
                        fout.write("\n".join(ocr_result))
            if os.path.exists(img_name):
                os.remove(img_name)
            return txt_file_path

        txt_file_path = pdf_ocr_txt(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(filename=txt_file_path, **self.unstructured_kwargs)

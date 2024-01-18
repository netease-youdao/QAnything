"""Loader that loads image files."""
from typing import List, Callable

from langchain.document_loaders.unstructured import UnstructuredFileLoader
import os
from typing import Union, Any
import cv2
import base64


class UnstructuredPaddleImageLoader(UnstructuredFileLoader):
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
        def image_ocr_txt(filepath, dir_path="tmp_files"):
            full_dir_path = os.path.join(os.path.dirname(filepath), dir_path)
            if not os.path.exists(full_dir_path):
                os.makedirs(full_dir_path)
            filename = os.path.split(filepath)[-1]
            img_np = cv2.imread(filepath)
            h, w, c = img_np.shape
            img_data = {"img64": base64.b64encode(img_np).decode("utf-8"), "height": h, "width": w, "channels": c}
            result = self.ocr_engine(img_data)
            result = [line for line in result if line]

            ocr_result = [i[1][0] for line in result for i in line]
            txt_file_path = os.path.join(full_dir_path, "%s.txt" % (filename))
            with open(txt_file_path, 'w', encoding='utf-8') as fout:
                fout.write("\n".join(ocr_result))
            return txt_file_path

        txt_file_path = image_ocr_txt(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(filename=txt_file_path, **self.unstructured_kwargs)

"""Loader that loads image files."""
from typing import List, Callable

from langchain.document_loaders.unstructured import UnstructuredFileLoader
from unstructured.partition.text import partition_text
import os
from typing import Union, Any
import cv2
import base64

class UnstructuredPaddleImageLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load image files, such as PNGs and JPGs."""
    def __init__(
        self,
        file_path: Union[str, List[str]],
        txt_file_path: str,
        mode: str = "single",
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""
        self.txt_file_path = txt_file_path
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        return partition_text(filename=self.txt_file_path, **self.unstructured_kwargs)
      
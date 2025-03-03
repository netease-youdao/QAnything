import json
from typing import Any, Dict, List, Optional, Sequence
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.helpers import detect_file_encodings
from io import TextIOWrapper


class JSONLoader(BaseLoader):
    """Load a `JSON` file into a list of Documents.

    Each document represents one item in the JSON file. Nested JSON structures are flattened
    into a key/value pair for every field in the structure. The result is outputted as a new line 
    in the document's page_content.

    The source for each document loaded from JSON is set to the value of the
    `file_path` argument for all documents by default.
    You can override this by setting the `source_column` argument to the
    name of a column in the JSON file (if it exists).
    The source of each document will then be set to the value of the column with the name specified in `source_column`.
    """

    def __init__(
            self,
            file_path: str,
            source_column: Optional[str] = None,
            metadata_columns: Sequence[str] = (),
            encoding: Optional[str] = None,
            autodetect_encoding: bool = False,
    ):
        """
        Args:
            file_path: The path to the JSON file.
            source_column: The name of the column in the JSON file to use as the source.
              Optional. Defaults to None.
            metadata_columns: A sequence of column names to use as metadata. Optional.
            encoding: The encoding of the JSON file. Optional. Defaults to None.
            autodetect_encoding: Whether to try to autodetect the file encoding.
        """
        self.file_path = file_path
        self.source_column = source_column
        self.metadata_columns = metadata_columns
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding

    def load(self) -> List[Document]:
        """Load data into document objects."""

        docs = []
        try:
            with open(self.file_path, "r", encoding=self.encoding) as jsonfile:
                docs = self.__read_file(jsonfile)
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(self.file_path)
                for encoding in detected_encodings:
                    try:
                        with open(self.file_path, "r", encoding=encoding.encoding) as jsonfile:
                            docs = self.__read_file(jsonfile)
                            break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {self.file_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

        return docs

    def __read_file(self, jsonfile: Any) -> List[Document]:
        docs = []
        data = json.load(jsonfile)

        # If the JSON is a list of objects, iterate through it
        if isinstance(data, list):
            for i, item in enumerate(data):
                docs.append(self.__process_item(item, i))
        else:
            docs.append(self.__process_item(data, 0))  # If it's a single object

        return docs

    def __process_item(self, item: Dict[str, Any], index: int) -> Document:
        """Process a single item (either a dictionary or a list) from the JSON file."""

        # Flatten the nested JSON structure
        line_contents = self.__flatten_json(item)
        
        content = '------------------------\n' + ' & '.join(line_contents) + '\n------------------------'
        
        # Determine source
        source = self.file_path
        if self.source_column and self.source_column in item:
            source = item[self.source_column]

        metadata = {"source": source, "index": index}
        for col in self.metadata_columns:
            if col in item:
                metadata[col] = item[col]

        return Document(page_content=content, metadata=metadata)

    def __flatten_json(self, item: Dict[str, Any], parent_key: str = '') -> List[str]:
        """Flatten a nested JSON structure."""
        items = []
        for k, v in item.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.__flatten_json(v, new_key))  # Recursively flatten
            elif isinstance(v, list):
                for i, sub_item in enumerate(v):
                    items.extend(self.__flatten_json({f"{i}": sub_item}, new_key))  # Handle list elements
            else:
                items.append(f"{new_key}: {v}")
        return items

if __name__ == "__main__":
    loader = JSONLoader("test/test.json")
    docs = loader.load()
    print(docs)
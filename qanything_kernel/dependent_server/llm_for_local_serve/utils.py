import enum
import logging
import time
from datetime import datetime
from enum import Enum

def log_timestamp() -> str:
    return datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S.%f")

@enum.unique
class CODES(Enum):
    SUCCESS = 0, "success"
    TYPE_ERROR = 3, "Type Error"
    RUNTIME_ERROR = 2, "Runtime Error"
    UNKNOWN_ERROR = 1, "Unknown Error"
    TRITON_INFERENCE_ERROR = 4, "Triton Inference Error"
    TRITON_CALLBACK_ERROR = 5, "Triton Callback Error"
    JSON_FORMAT_ERROR = 6, "Json Format Error"
    ILLEGAL_QUERY = 10, "Illegal Query Error"
    TRITON_TIMEOUT = 20, "Triton Timeout Error"
    INVALID_PARAMS = 30, "Invalid Parameters Error"
    MISSING_CONTENT = 40, "Missing Content Error"

    def __init__(self, code, desc) -> None:
        self._code_ = code
        self._desc_ = desc

    @property
    def desc(self) -> str:
        return self._desc_

    @property
    def code(self) -> int:
        return self._code_


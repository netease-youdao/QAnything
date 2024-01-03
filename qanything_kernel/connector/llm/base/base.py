from abc import ABC, abstractmethod
from typing import Optional, List


class AnswerResult:
    """
    消息实体
    """
    history: List[List[str]] = []
    llm_output: Optional[dict] = None
    prompt: str = ""


class BaseAnswer(ABC):
    """上层业务包装器.用于结果生成统一api调用"""

    @property
    @abstractmethod
    def _history_len(self) -> int:
        """Return _history_len of llm."""

    @abstractmethod
    def set_history_len(self, history_len: int) -> None:
        """Return _history_len of llm."""

    def generatorAnswer(self, prompt: str,
                        history: List[List[str]] = [],
                        streaming: bool = False):
        pass

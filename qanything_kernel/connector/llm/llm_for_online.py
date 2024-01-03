from abc import ABC
import tiktoken
import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, List
from qanything_kernel.connector.llm.base import (BaseAnswer, AnswerResult)

load_dotenv()


class OpenAILLM(BaseAnswer, ABC):
    model: str = "gpt-3.5-turbo"
    token_window: int = 4096
    max_token: int = 512
    offcut_token: int = 50
    truncate_len: int = 50
    temperature: float = 0
    history: List[List[str]] = []
    history_len: int = 2

    def __init__(self):
        super().__init__()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @property
    def _llm_type(self) -> str:
        return "OpenAILLM"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def num_tokens_from_messages(self, message_texts):
        encoding = tiktoken.encoding_for_model(self.model)
        num_tokens = 0
        for message in message_texts:
            num_tokens += len(encoding.encode(message, disallowed_special=()))
        return num_tokens

    def num_tokens_from_docs(self, docs):
        encoding = tiktoken.encoding_for_model(self.model)
        num_tokens = 0
        for doc in docs:
            num_tokens += len(encoding.encode(doc.page_content, disallowed_special=()))
        return num_tokens

    def _call(self, prompt: str, history: List[List[str]]) -> str:
        messages = []
        for pair in history:
            question, answer = pair
            messages.append({"role": "user", "text": question})
            messages.append({"role": "assistant", "text": answer})
        messages.append({"role": "user", "text": prompt})
        print(messages)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_token,
                temperature=self.temperature,
                stop=None
            )
            return response['choices'][0]['text'] if response['choices'] else ""
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return ""

    def generatorAnswer(self, prompt: str,
                        history: List[List[str]] = [],
                        streaming: bool = False) -> AnswerResult:
        response_text = self._call(prompt, history)
        answer_result = AnswerResult()
        answer_result.llm_output = {"answer": response_text}
        answer_result.prompt = prompt
        yield answer_result

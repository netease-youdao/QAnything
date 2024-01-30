from abc import ABC
from langchain.llms.base import LLM
from typing import Optional, List
from qanything_kernel.connector.llm.base import (BaseAnswer,
                                                 AnswerResult)
import json
import requests
from requests.exceptions import RequestException
from collections import OrderedDict
import tiktoken
from qanything_kernel.utils.custom_log import debug_logger, qa_logger
import os
from dotenv import load_dotenv
load_dotenv()


class ZiyueLLM(BaseAnswer, LLM, ABC):
    model_name: str = "ZiyueLLM"
    model: str = "yd_gpt"
    token_window: int = 4096
    max_token: int = 300  # 300
    offcut_token: int = int(os.getenv("OFFCUT_TOKEN", 50))
    truncate_len: int = 50
    temperature: float = 0.6
    top_p: float = 1.0
    top_k: int = 4
    repetition_penalty: float = 1.2
    check_in: int = 0
    url: str = "http://0.0.0.0:36001/worker_generate_stream"

    history: List[List[str]] = []
    history_len: int = 2

    def __init__(self):
        super().__init__()
        debug_logger.info("ZiyueLLM offcut_token: %s", self.offcut_token)

    @property
    def _llm_type(self) -> str:
        return "ZiyueLLM"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.chat(
            prompt,
            history=[]
        )
        return response

    def num_tokens_from_messages(self, message_texts):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
        num_tokens = 0
        for message in message_texts:
            num_tokens += len(encoding.encode(message, disallowed_special=()))
        return num_tokens

    def num_tokens_from_docs(self, docs):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
        num_tokens = 0
        for doc in docs:
            num_tokens += len(encoding.encode(doc.page_content, disallowed_special=()))
        return num_tokens

    def generatorAnswer(self, prompt: str,
                        history=None,
                        streaming: bool = False):
        if history is None:
            history = []
        print("self.history_len:", self.history_len)
        print("prompt:\n", prompt)
        print("prompt tokens:", self.num_tokens_from_messages([prompt]))
        print("streaming:", streaming)
        if streaming:
            history += [[]]
            complete_answer = ""
            for stream_resp in self.stream_chat(prompt, history=history[:-1]):
                print("stream res:", stream_resp, flush=True)
                if stream_resp:
                    chunk_str = stream_resp[6:]
                    # 如果没有以[DONE]结尾咋办
                    if not chunk_str.startswith("[DONE]"):
                        chunk_js = json.loads(chunk_str)
                        complete_answer += chunk_js["answer"]
                
                history[-1] = [prompt, complete_answer]
                answer_result = AnswerResult()
                answer_result.history = history
                answer_result.llm_output = {"answer": stream_resp}
                answer_result.prompt = prompt
                yield answer_result
        else:
            response = self.chat(prompt, history=history[-self.history_len:] if self.history_len > 0 else [])
            history += [[prompt, response]]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": response}
            answer_result.prompt = prompt
            yield answer_result

    def chat(self, prompt, history):
        print('one time chat')
        hist_messages = OrderedDict()
        for k, msg in enumerate(history):
            hist_messages[k] = {"user": msg[0] if msg[0] != None else "", "chatbot": msg[1] if msg[1] != None else ""}

        print("hist_messages", hist_messages)
        data_raw = {
            "model": self.model,
            "prompt": prompt,
            "hist_messages": hist_messages,
            "temperature": self.temperature,
            "max_new_tokens": self.max_token,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "check_in": self.check_in,
            "stop": None}

        response = self.retry_requests(data_raw=data_raw, headers={"User-Agent": "fastchat Client"})

        return response
    
    def stream_chat(self, prompt, history):
        print('stream chat', flush=True)
        hist_messages = OrderedDict()
        for k, msg in enumerate(history):
            hist_messages[k] = {"user": msg[0] if msg[0] != None else "", "chatbot": msg[1] if msg[1] != None else ""}

        print("hist_messages", hist_messages)
        data_raw = {
            "model": self.model,
            "prompt": prompt,
            "hist_messages": hist_messages,
            "temperature": self.temperature,
            "max_new_tokens": self.max_token,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "check_in": self.check_in,
            "stop": None}

        for res in self.retry_stream_requests(data_raw=data_raw, headers={"User-Agent": "fastchat Client"}):
            yield res

    def retry_stream_requests(self, data_raw, headers):
        response = requests.post(
            self.url,
            headers=headers,
            json=data_raw,
            timeout=60,
            stream=True
        )
        try:
            response.raise_for_status()
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\n\n"):
                delta = {"answer": ""}
                if chunk:
                    data = chunk.decode('utf-8')[6:]
                    data = json.loads(data)
                    if data["error_code"] == 0:         
                        text = data['text']           
                    else:
                        text = data["text"] + f" (error_code: {data['error_code']})"
                        print(f"stream error: {text}")
                    delta["answer"] = text
                    yield "data: " + json.dumps(delta, ensure_ascii=False)
        except RequestException as e:
            print("Error sending request: {}".format(e))
            yield "data: " + json.dumps({"answer": "ERROR: request for llm failed."}, ensure_ascii=False)
        yield f"data: [DONE]\n\n"

    def retry_requests(self, data_raw, headers):
        response = requests.post(
            self.url,
            headers=headers,
            json=data_raw,
            timeout=60,
            stream=True
        )
        try:
            response.raise_for_status()
            final_response = ""
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\n\n"):
                if chunk:
                    data = chunk.decode('utf-8')[6:]
                    data = json.loads(data)
                    if data["error_code"] == 0:         
                        text = data['text']           
                    else:
                        text = data["text"] + f" (error_code: {data['error_code']})"
                        print(f"stream error: {text}")
                    final_response += text
        except RequestException as e:
            print("Error sending request: {}".format(e))
            final_response = "ERROR: request for llm failed."
        return final_response

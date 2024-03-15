from abc import ABC
import tiktoken
import os
from dotenv import load_dotenv
from typing import Optional, List
import sys
import json
import requests

sys.path.append("../../../")
from qanything_kernel.connector.llm.base import (BaseAnswer, AnswerResult)
from qanything_kernel.utils.custom_log import debug_logger
from llama_cpp import Llama
from huggingface_hub import hf_hub_download


class LlamaCPPCustomLLM(BaseAnswer, ABC):
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    token_window: int = 4096
    max_token: int = 512
    offcut_token: int = 50
    truncate_len: int = 50
    model_folder: str = "Qwen1.5-7B-Chat/"
    model_name: str = "qwen1_5-7b-chat-q4_k_m.gguf"
    temperature: float = 0
    stop_words: str = None
    history: List[List[str]] = []
    history_len: int = 3

    def __init__(self):
        super().__init__()
        self._get_model()
        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx
        )

    def _get_model(self) -> None:
        current_script_path = os.path.abspath(__file__)
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
        self.model_path = root_path + '/assets/custom_models/' + self.model_folder + self.model_name
        os.makedirs(root_path + '/assets/custom_models/' + self.model_folder, exist_ok=True)
        ori_path = os.getcwd()
        os.chdir(root_path + '/assets/custom_models/' + self.model_folder)
        if not os.path.exists(self.model_name):
            hf_hub_download(repo_id='Qwen/Qwen1.5-7B-Chat-GGUF', filename=self.model_name, cache_dir='./', local_dir='./')
        os.chdir(ori_path)

    @property
    def _llm_type(self) -> str:
        return "CustomLLM using LlamaCPP backend"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def num_tokens_from_messages(self, message_texts):
        num_tokens = 0
        for message in message_texts:
            num_tokens += len(self.llm.tokenizer().encode(message, add_bos=False, special=False))
        return num_tokens

    def num_tokens_from_docs(self, docs):
        num_tokens = 0
        for doc in docs:
            num_tokens += len(self.llm.tokenizer().encode(doc.page_content, add_bos=False, special=False))
        return num_tokens

    async def _call(self, prompt: str, history: List[List[str]], streaming: bool = False) -> str:
        messages = []
        for pair in history:
            question, answer = pair
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": prompt})
        debug_logger.info(messages)

        if streaming:

            results = self.llm.create_chat_completion(messages=messages,
                                                      max_tokens=self.max_token,
                                                      stream=True)

            for chunk in results:
                if isinstance(chunk['choices'], List) and len(chunk['choices']) > 0:
                    if 'content' in chunk['choices'][0]['delta']:
                        chunk_text = chunk['choices'][0]['delta']['content']
                        if isinstance(chunk_text, str) and chunk_text != "":
                            # debug_logger.info(f"[debug] event_text = [{event_text}]")
                            delta = {'answer': chunk_text}
                            yield "data: " + json.dumps(delta, ensure_ascii=False)

        else:
            results = self.llm.create_chat_completion(messages=messages,
                                                      max_tokens=self.max_token,
                                                      stream=False)
            if isinstance(results['choices'], List) and len(results['choices']) > 0:
                text = results['choices'][0]['message']['content']
                if isinstance(text, str) and text != "":
                    # debug_logger.info(f"[debug] event_text = [{event_text}]")
                    delta = {'answer': text}
                    yield "data: " + json.dumps(delta, ensure_ascii=False)

        yield f"data: [DONE]\n\n"

    async def generatorAnswer(self, prompt: str,
                              history: List[List[str]] = [],
                              streaming: bool = False) -> AnswerResult:

        if history is None or len(history) == 0:
            history = [[]]
        debug_logger.info(f"history_len: {self.history_len}")
        debug_logger.info(f"prompt: {prompt}")
        debug_logger.info(f"streaming: {streaming}")

        response = self._call(prompt, history[:-1], streaming)
        complete_answer = ""
        async for response_text in response:
            if response_text:
                chunk_str = response_text[6:]
                if not chunk_str.startswith("[DONE]"):
                    chunk_js = json.loads(chunk_str)
                    complete_answer += chunk_js["answer"]

            history[-1] = [prompt, complete_answer]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": response_text}
            answer_result.prompt = prompt
            yield answer_result


import shutil
from abc import ABC
from threading import Thread

import tiktoken
import os

import torch
from dotenv import load_dotenv
from typing import Optional, List
import sys
import json
import requests

sys.path.append("../../../")
from qanything_kernel.connector.llm.base import (BaseAnswer, AnswerResult)
from qanything_kernel.utils.custom_log import debug_logger
from transformers import AutoTokenizer, TextIteratorStreamer
from bigdl.llm.transformers import AutoModelForCausalLM
from bigdl.llm import optimize_model
import intel_extension_for_pytorch as ipex
import numpy as np


class BigDLCustomLLM(BaseAnswer, ABC):
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    token_window: int = 4096
    max_token: int = 512
    offcut_token: int = 50
    truncate_len: int = 50
    model_folder: str = "Qwen1.5-7B-Chat/"
    model_name: str = "int4_model"
    temperature: float = 0
    stop_words: str = None
    history: List[List[str]] = []
    history_len: int = 3

    def __init__(self):
        super().__init__()
        self._get_model()

    def _get_model(self) -> None:
        current_script_path = os.path.abspath(__file__)
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
        self.model_path = root_path + '/assets/custom_models/' + self.model_folder + self.model_name
        self.tokenizer_path = root_path + '/assets/custom_models/' + self.model_folder + 'tokenizer'
        os.makedirs(root_path + '/assets/custom_models/' + self.model_folder, exist_ok=True)
        ori_path = os.getcwd()
        os.chdir(root_path + '/assets/custom_models/' + self.model_folder)
        if not os.path.exists(self.model_name):
            os.makedirs('./tmp', exist_ok=True)
            self.llm = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-7B-Chat', load_in_4bit=True,
                                                            trust_remote_code=True, cache_dir='./tmp')
            self.llm.save_low_bit(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-7B-Chat',
                                                           trust_remote_code=True, cache_dir='./tmp')
            self.tokenizer.save_pretrained('./tokenizer')
            shutil.rmtree('./tmp')
        else:
            self.llm = AutoModelForCausalLM.load_low_bit(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        self.llm = self.llm.to('xpu')
        os.chdir(ori_path)

    @property
    def _llm_type(self) -> str:
        return "CustomLLM using BigDL backend"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def num_tokens_from_messages(self, message_texts):
        num_tokens = 0
        for message in message_texts:
            num_tokens += len(self.tokenizer(message, add_special_tokens=False))
        return num_tokens

    def num_tokens_from_docs(self, docs):
        num_tokens = 0
        for doc in docs:
            num_tokens += len(self.tokenizer(doc.page_content, add_special_tokens=False))
        return num_tokens

    async def _call(self, prompt: str, history: List[List[str]], streaming: bool = False) -> str:
        messages = []
        for pair in history:
            question, answer = pair
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": prompt})
        debug_logger.info(messages)

        with torch.inference_mode():
            if streaming:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = self.tokenizer([text], return_tensors="pt").to("xpu")
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                generate_kwargs = dict(
                    model_inputs,
                    streamer=streamer,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=self.max_token
                )
                thread = Thread(target=self.llm.generate, kwargs=generate_kwargs)
                thread.start()
                for stream_out in streamer:
                    if isinstance(stream_out, str) and stream_out != "":
                        delta = {'answer': stream_out}
                        yield "data: " + json.dumps(delta, ensure_ascii=False)
                thread.join()

            else:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = self.tokenizer([text], return_tensors="pt").to("xpu")

                generated_ids = self.llm.generate(
                    model_inputs.input_ids,
                    max_new_tokens=self.max_token
                )
                torch.xpu.synchronize()
                generated_ids = generated_ids.cpu()
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                if isinstance(response, str) and response != "":
                    delta = {'answer': response}
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


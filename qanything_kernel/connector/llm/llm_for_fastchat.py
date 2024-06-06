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
from qanything_kernel.configs.model_config import DT_CONV_7B_TEMPLATE, DT_CONV_3B_TEMPLATE
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid
from qanything_kernel.configs.conversation import get_conv_template
from qanything_kernel.utils.custom_log import debug_logger
import config

load_dotenv()


class OpenAICustomLLM(BaseAnswer, ABC):
    token_window: int = config.llm_config['token_window']
    max_token: int = config.llm_config['max_token']
    offcut_token: int = 50
    truncate_len: int = 50
    temperature: float = 0
    stop_words: str = None
    history: List[List[str]] = []
    history_len: int = config.llm_config['history_len']

    def __init__(self, args):
        super().__init__()
        # self.llm = LLM(model=DT_MODEL_PATH)
        # parser = argparse.ArgumentParser()
        # parser = AsyncEngineArgs.add_cli_args(parser)
        args.trust_remote_code = True
        engine_args = AsyncEngineArgs.from_cli_args(args)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.sampling_params = SamplingParams(temperature=0.6, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
        if args.model_size == '3B':
            self.conv_template = DT_CONV_3B_TEMPLATE
        else:
            self.conv_template = DT_CONV_7B_TEMPLATE
        debug_logger.info(f"conv_template: {self.conv_template}")

    @property
    def _llm_type(self) -> str:
        return "CustomLLM using FastChat w/ huggingface transformers or vllm backend"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def token_check(self, query: str) -> int:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
        return len(encoding.encode(query, disallowed_special=()))

    def num_tokens_from_messages(self, message_texts):
        num_tokens = 0
        for message in message_texts:
            num_tokens += self.token_check(message)
        return num_tokens

    def num_tokens_from_docs(self, docs):
        num_tokens = 0
        for doc in docs:
            num_tokens += self.token_check(doc.page_content)
        return num_tokens

    async def _call(self, prompt: str, history: List[List[str]], streaming: bool = False) -> str:
        conv = get_conv_template(self.conv_template)
        for pair in history:
            question, answer = pair
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], answer)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        results_generator = self.engine.generate(prompt, self.sampling_params, request_id=random_uuid())
        if streaming:
            pre_text_len = 0
            async for request_output in results_generator:
                delta = {"answer": request_output.outputs[0].text[pre_text_len:]}
                pre_text_len += len(delta['answer'])
                # delta = {"answer": request_output.outputs[0].text}
                yield "data: " + json.dumps(delta, ensure_ascii=False)

            yield f"data: [DONE]\n\n"
        else:
            delta = {"answer": ""}
            async for request_output in results_generator:
                delta["answer"] = request_output.outputs[0].text
            yield "data: " + json.dumps(delta, ensure_ascii=False)

    async def generatorAnswer(self, prompt: str,
                              history: List[List[str]] = [],
                              streaming: bool = False) -> AnswerResult:

        if history is None or len(history) == 0:
            history = [[]]
        debug_logger.info(f"history_len: {self.history_len}")
        debug_logger.info(f"prompt: {prompt}")
        debug_logger.info(f"prompt tokens: {self.num_tokens_from_messages([prompt])}")
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

from qanything_kernel.connector.llm.base import (BaseAnswer, AnswerResult)
from qanything_kernel.configs.model_config import DT_CONV_7B_TEMPLATE, DT_CONV_3B_TEMPLATE
from qanything_kernel.configs.conversation import get_conv_template
from qanything_kernel.utils.custom_log import debug_logger
from llama_cpp import Llama
from abc import ABC
from typing import List
import json
import config


class LlamaCPPCustomLLM(BaseAnswer, ABC):
    n_gpu_layers: int = -1
    n_ctx: int = 4096
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
        self.llm = Llama(
            model_path=args.model,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx
        )
        if args.model_size == '3B':
            self.conv_template = DT_CONV_3B_TEMPLATE
        else:
            self.conv_template = DT_CONV_7B_TEMPLATE
        debug_logger.info(f"conv_template: {self.conv_template}, {args.model_size}")

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
        # for pair in history:
        #     question, answer = pair
        #     messages.append({"role": "user", "content": question})
        #     messages.append({"role": "assistant", "content": answer})
        # messages.append({"role": "user", "content": prompt})

        conv = get_conv_template(self.conv_template)
        for pair in history:
            question, answer = pair
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], answer)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        content = conv.get_prompt()
        messages.append({"role": "user", "content": content})
        # debug_logger.info('content: %s', content)
        debug_logger.info(messages)

        if streaming:

            results = self.llm.create_chat_completion(messages=messages,
                                                      max_tokens=self.max_token,
                                                      stream=True,
                                                      temperature=0.7,
                                                      top_p=0.8
                                                      )
            for chunk in results:
                if isinstance(chunk['choices'], List) and len(chunk['choices']) > 0:
                    if 'content' in chunk['choices'][0]['delta']:
                        chunk_text = chunk['choices'][0]['delta']['content']
                        debug_logger.info(f"[debug] event_text = [{chunk_text}]")
                        if isinstance(chunk_text, str) and chunk_text != "":
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


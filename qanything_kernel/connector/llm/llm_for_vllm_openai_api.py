import tiktoken
import os
import json
import logging
from abc import ABC
from openai import OpenAI
from typing import Optional, List
from qanything_kernel.utils.custom_log import debug_logger, qa_logger
from qanything_kernel.configs.model_config import VLLM_OPENAI_API_KEY, VLLM_OPENAI_API_BASE, VLLM_OPENAI_API_MODEL_NAME, VLLM_OPENAI_API_CONTEXT_LENGTH
from qanything_kernel.configs.conversation import get_conv_template

OPENAI_API_KEY = VLLM_OPENAI_API_KEY
OPENAI_API_BASE = VLLM_OPENAI_API_BASE
OPENAI_API_MODEL_NAME = VLLM_OPENAI_API_MODEL_NAME
OPENAI_API_CONTEXT_LENGTH = VLLM_OPENAI_API_CONTEXT_LENGTH

if isinstance(OPENAI_API_CONTEXT_LENGTH, str) and OPENAI_API_CONTEXT_LENGTH != '':
    OPENAI_API_CONTEXT_LENGTH = int(OPENAI_API_CONTEXT_LENGTH)
debug_logger.info(f"OPENAI_API_BASE = {OPENAI_API_BASE}")
debug_logger.info(f"OPENAI_API_MODEL_NAME = {OPENAI_API_MODEL_NAME}")

class AnswerResult:
    """
    消息实体
    """
    history: List[List[str]] = []
    llm_output: Optional[dict] = None
    prompt: str = ""


class OpenAILLM(ABC):
    model: str = OPENAI_API_MODEL_NAME
    token_window: int = OPENAI_API_CONTEXT_LENGTH
    max_token: int = 1024
    offcut_token: int = 50
    truncate_len: int = 50
    temperature: float = 0.6
    top_p: float = 0.8 # top_p must be (0,1]
    top_k: int = -1
    stop_words: str = None
    extra_body: dict = {"repetition_penalty": 1.05}
    history: List[List[str]] = []
    history_len: int = 2

    def __init__(self):
        super().__init__()
        self.client = OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)
        self.conv_template = "qwen-7b-chat"

    @property
    def _llm_type(self) -> str:
        return "using OpenAI API serve as LLM backend"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len
        
    def _get_sampling_params(self, model_name):
        
        sampling_params = {
            'youdao-local-llm-qanything-std': {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'repetition_penalty': 1.05, 'stop': ["<|im_end|>"]},
            'youdao-local-llm-qanything-pro': {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'repetition_penalty': 1.05, 'stop': ["<|im_end|>"]},
            'youdao-local-llm-qanything-turbo': {'temperature': 0.6, 'top_p': 0.8, 'top_k': -1, 'repetition_penalty': 1.05, 'stop': ["<|endoftext|>"]},
        }
        extra_body = self.extra_body
        if model_name not in sampling_params:
            extra_body = {'temperature': 0.6, 'top_p': 0.8, 'top_k': -1, 'repetition_penalty': 1.05, 'stop': None}
        else:
            extra_body = sampling_params[model_name]
        
        return extra_body

    # 定义函数 num_tokens_from_messages，该函数返回由一组消息所使用的token数
    def _num_tokens_from_messages(self, messages, model=None):
        """Return the number of tokens used by a list of messages. From https://github.com/DjangoPeng/openai-quickstart/blob/main/openai_api/count_tokens_with_tiktoken.ipynb"""
        
        if model is None:
            model = self.model
        # 尝试获取模型的编码
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # 如果模型没有找到，使用 cl100k_base 编码并给出警告
            debug_logger.warning("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        # 针对不同的模型设置token数量
        if model in {
            "gpt-3.5-turbo-0613",
            # "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4-32k",
            # "gpt-4-1106-preview",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # 每条消息遵循 {role/name}\n{content}\n 格式
            tokens_per_name = -1  # 如果有名字，角色会被省略
        elif "gpt-3.5-turbo" in model:
            # 对于 gpt-3.5-turbo 模型可能会有更新，此处返回假设为 gpt-3.5-turbo-0613 的token数量，并给出警告
            debug_logger.warning("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            # 对于 gpt-4 模型可能会有更新，此处返回假设为 gpt-4-0613 的token数量，并给出警告
            debug_logger.warning("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")

        else:
            # 对于没有实现的模型，抛出未实现错误
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
            
        num_tokens = 0
        # 计算每条消息的token数
        for message in messages:
            if isinstance(message, dict):
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
            elif isinstance(message, str):
                num_tokens += len(encoding.encode(message))
            else:
                NotImplementedError(
                f"""num_tokens_from_messages() is not implemented message type {type(message)}. """
            )

        num_tokens += 3  # 每条回复都以助手为首
        return num_tokens

    def _num_tokens_from_docs(self, docs):
        
        # 尝试获取模型的编码
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # 如果模型没有找到，使用 cl100k_base 编码并给出警告
            debug_logger.warning("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        for doc in docs:
            num_tokens += len(encoding.encode(doc.page_content, disallowed_special=()))
        return num_tokens
    
    def num_tokens_from_messages(self, messages):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
        num_tokens = 0
        tokens_per_message = 3
        tokens_per_name = 1
        for message in messages:
            if isinstance(message, dict):
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
            elif isinstance(message, str):
                num_tokens += len(encoding.encode(message))
            else:
                NotImplementedError(
                f"""num_tokens_from_messages() is not implemented message type {type(message)}. """
            )
        return num_tokens

    def num_tokens_from_docs(self, docs):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
        num_tokens = 0
        for doc in docs:
            num_tokens += len(encoding.encode(doc.page_content, disallowed_special=()))
        return num_tokens

    def _call(self, prompt: str, history: List[List[str]], streaming: bool=False) -> str:
        messages = []
        for pair in history:
            question, answer = pair
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
            
        messages.append({"role": "user", "content": prompt})

        try:

            if streaming:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    max_tokens=self.max_token,
                    extra_body=self._get_sampling_params(self.model)
                )
                for event in response:
                    if not isinstance(event, dict):
                        event = event.model_dump()

                    if isinstance(event['choices'], List) and len(event['choices']) > 0:
                        event_text = event["choices"][0]['delta']['content']
                        if isinstance(event_text, str) and event_text != "":
                            delta = {'answer': event_text}
                            yield "data: " + json.dumps(delta, ensure_ascii=False)

            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    max_tokens=self.max_token,
                    extra_body=self._get_sampling_params(self.model)
                )
                
                event_text = response.choices[0].message.content if response.choices else ""
                delta = {'answer': event_text}
                yield "data: " + json.dumps(delta, ensure_ascii=False)

        except Exception as e:
            debug_logger.error(f"Error calling OpenAI API: {e}")
            delta = {'answer': f"{e}"}
            yield "data: " + json.dumps(delta, ensure_ascii=False)

        finally:
            yield f"data: [DONE]\n\n"

    def generatorAnswer(self, prompt: str,
                        history: List[List[str]] = [],
                        streaming: bool = False) -> AnswerResult:
        conv = get_conv_template(self.conv_template)
        for pair in history:
            question, answer = pair
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], answer)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if history is None or len(history) == 0:
            history = [[]]
        else:
            history += [[]]
        qa_logger.info(f"history_len: {self.history_len}")
        qa_logger.info(f"prompt: {prompt}")
        qa_logger.info(f"prompt tokens: {self.num_tokens_from_messages([{'content': prompt}])}")
        qa_logger.info(f"streaming: {streaming}")
                
        response = self._call(prompt, history[:-1], streaming)
        complete_answer = ""
        for response_text in response:

            if response_text:
                chunk_str = response_text[6:]
                if not chunk_str.startswith("[DONE]"):
                    chunk_js = json.loads(chunk_str)
                    complete_answer += chunk_js["answer"]
                    
            history[-1] = [prompt, complete_answer]
            answer_result = AnswerResult()
            answer_result.history = history
            if streaming:
                answer_result.llm_output = {"answer": response_text}
            else:
                answer_result.llm_output = {"answer": complete_answer}
            answer_result.prompt = prompt
            yield answer_result


if __name__ == "__main__":

    llm = OpenAILLM()
    streaming = True
    chat_history = []
    prompt = """参考信息：
中央纪委国家监委网站讯 据山西省纪委监委消息：山西转型综合改革示范区党工委副书记、管委会副主任董良涉嫌严重违纪违法，目前正接受山西省纪委监委纪律审查和监察调查。\\u3000\\u3000董良简历\\u3000\\u3000董良，男，汉族，1964年8月生，河南鹿邑人，在职研究生学历，邮箱random@xxx.com，联系电话131xxxxx909，1984年3月加入中国共产党，1984年8月参加工作\\u3000\\u3000历任太原经济技术开发区管委会副主任、太原武宿综合保税区专职副主任，山西转型综合改革示范区党工委委员、管委会副主任。2021年8月，任山西转型综合改革示范区党工委副书记、管委会副主任。(山西省纪委监委)
---
我的问题或指令：
帮我提取上述人物的中文名，英文名，性别，国籍，现任职位，最高学历，毕业院校，邮箱，电话
---
请根据上述参考信息回答我的问题或回复我的指令。前面的参考信息可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复,
你的回复："""
    PROMPT_TEMPLATE_FOR_VLLM = """<SYSTEM>
You are a helpful assistant. You are always a reliable assistant that can answer questions with the help of external documents.
</SYSTEM>
<INSTRUCTIONS>
- All contents between <DOCUMENTS> and </DOCUMENTS> are reference information retrieved from an external knowledge base.
- If you cannot answer based on the given information, you will return the sentence \"抱歉，检索到的参考信息并未提供充足的信息，因此无法回答。\".
- Answer the following question based on the retrieved documents:
{question}
- Return your answer in the same language as the question "{question}".
</INSTRUCTIONS>

<DOCUMENTS>
{context}
</DOCUMENTS>

<INSTRUCTIONS>
- All contents between <DOCUMENTS> and </DOCUMENTS> are reference information retrieved from an external knowledge base.
- If you cannot answer based on the given information, you will return the sentence \"抱歉，检索到的参考信息并未提供充足的信息，因此无法回答。\".
- Now, answer the following question based on the above retrieved documents:
{question}
- Return your answer in the same language as the question "{question}".
</INSTRUCTIONS>
"""

    question = "帮我提取上述人物的中文名，英文名，性别，国籍，现任职位，最高学历，毕业院校，邮箱，电话"
    context = """中央纪委国家监委网站讯 据山西省纪委监委消息：山西转型综合改革示范区党工委副书记、管委会副主任董良涉嫌严重违纪违法，目前正接受山西省纪委监委纪律审查和监察调查。\\u3000\\u3000董良简历\\u3000\\u3000董良，男，汉族，1964年8月生，河南鹿邑人，在职研究生学历，邮箱random@xxx.com，联系电话131xxxxx909，1984年3月加入中国共产党，1984年8月参加工作\\u3000\\u3000历任太原经济技术开发区管委会副主任、太原武宿综合保税区专职副主任，山西转型综合改革示范区党工委委员、管委会副主任。2021年8月，任山西转型综合改革示范区党工委副书记、管委会副主任。(山西省纪委监委)"""
    
    prompt = PROMPT_TEMPLATE_FOR_VLLM.replace("{question}", f"{question}").replace("{context}", f"{context}")
    print(prompt)
    final_result = ""
    for answer_result in llm.generatorAnswer(prompt=prompt,
                                             history=chat_history,
                                             streaming=streaming):
        resp = answer_result.llm_output["answer"]
        print(resp)
        if "DONE" not in resp:
            final_result += json.loads(resp[6:])["answer"]
        qa_logger.info(resp)
    print(final_result)
    qa_logger.info(f"final_result = {final_result}")


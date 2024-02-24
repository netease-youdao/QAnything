from abc import ABC
import tiktoken
import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, List
import sys
import json
import requests
import logging
sys.path.append("../../../")
from qanything_kernel.connector.llm.base import (BaseAnswer, AnswerResult)
from qanything_kernel.configs.model_config import LOCAL_LLM_SERVICE_URL, LOCAL_LLM_MODEL_NAME, LOCAL_LLM_MAX_LENGTH

load_dotenv()

logging.basicConfig(level=logging.INFO)

class OpenAICustomLLM(BaseAnswer, ABC):
    model: str = LOCAL_LLM_MODEL_NAME
    token_window: int = LOCAL_LLM_MAX_LENGTH
    max_token: int = 512
    offcut_token: int = 50
    truncate_len: int = 50
    temperature: float = 0
    stop_words: str = None
    history: List[List[str]] = []
    history_len: int = 2

    def __init__(self):
        super().__init__()
        # self.client = OpenAI(base_url="http://localhost:7802/v1", api_key="EMPTY")
        if LOCAL_LLM_SERVICE_URL.startswith("http://"):
            base_url = f"{LOCAL_LLM_SERVICE_URL}/v1" 
        else:
            base_url = f"http://{LOCAL_LLM_SERVICE_URL}/v1" 
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")

    @property
    def _llm_type(self) -> str:
        return "CustomLLM using FastChat w/ huggingface transformers or vllm backend"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def token_check(self, query: str) -> int:
        
        if LOCAL_LLM_SERVICE_URL.startswith("http://"):
            base_url = f"{LOCAL_LLM_SERVICE_URL}/api/v1/token_check" 
        else:
            base_url = f"http://{LOCAL_LLM_SERVICE_URL}/api/v1/token_check" 

        headers = {"Content-Type": "application/json"}
        
        response = requests.post(
            base_url, 
            data=json.dumps(
                {'prompts': [{'model': self.model, 'prompt': query, 'max_tokens': self.max_token}]}
            ),
            headers=headers)

        # {'prompts': [{'fits': True, 'tokenCount': 317, 'contextLength': 8192}]}
        result = response.json()
        token_num = 0
        try:
            token_num = result['prompts'][0]['tokenCount']
            return token_num
        except Exception as e:
            logging.error(f"token_check Exception {base_url} w/ {e}")
            return token_num

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

    def _call(self, prompt: str, history: List[List[str]], streaming: bool=False) -> str:
        messages = []
        for pair in history:
            question, answer = pair
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": prompt})
        logging.info(messages)

        try:

            if streaming:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    max_tokens=self.max_token,
                    # temperature=self.temperature,
                    stop=[self.stop_words] if self.stop_words is not None else None,
                )

                for event in response:
                    if not isinstance(event, dict):
                        event = event.model_dump()

                    event_text = event["choices"][0]['delta']['content']
                    if isinstance(event_text, str) and event_text != "":
                        # logging.info(f"[debug] event_text = [{event_text}]")
                        delta = {'answer': event_text}
                        yield "data: " + json.dumps(delta, ensure_ascii=False)

            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    max_tokens=self.max_token,
                    # temperature=self.temperature,
                    stop=[self.stop_words] if self.stop_words is not None else None,
                )
                
                # logging.info(f"[debug] response.choices = [{response.choices}]")
                event_text = response.choices[0].message.content if response.choices else ""
                delta = {'answer': event_text}
                yield "data: " + json.dumps(delta, ensure_ascii=False)

        except Exception as e:
            logging.info(f"Error calling API: {e}")
            delta = {'answer': f"{e}"}
            yield "data: " + json.dumps(delta, ensure_ascii=False)

        finally:
            # logging.info("[debug] try-finally")
            yield f"data: [DONE]\n\n"

    def generatorAnswer(self, prompt: str,
                        history: List[List[str]] = [],
                        streaming: bool = False) -> AnswerResult:

        if history is None or len(history) == 0:
            history = [[]]
        logging.info(f"history_len: {self.history_len}")
        logging.info(f"prompt: {prompt}")
        logging.info(f"prompt tokens: {self.num_tokens_from_messages([prompt])}")
        logging.info(f"streaming: {streaming}")
                
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

    base_url = f"http://{LOCAL_LLM_SERVICE_URL}/api/v1/token_check" 
    headers = {"Content-Type": "application/json"}
    query = "hello"
    response = requests.post(
        base_url, 
        data=json.dumps(
            {'prompts': [{'model': LOCAL_LLM_MODEL_NAME, 'prompt': query, 'max_tokens': 512}]}
        ),
        headers=headers)

    # {'prompts': [{'fits': True, 'tokenCount': 317, 'contextLength': 8192}]}
    result = response.json()
    logging.info(f"[debug] result = {result}")


    llm = OpenAICustomLLM()
    streaming = True
    chat_history = []
    prompt = "你是谁"
    prompt = """参考信息：
中央纪委国家监委网站讯 据山西省纪委监委消息：山西转型综合改革示范区党工委副书记、管委会副主任董良涉嫌严重违纪违法，目前正接受山西省纪委监委纪律审查和监察调查。\\u3000\\u3000董良简历\\u3000\\u3000董良，男，汉族，1964年8月生，河南鹿邑人，在职研究生学历，邮箱random@xxx.com，联系电话131xxxxx909，1984年3月加入中国共产党，1984年8月参加工作\\u3000\\u3000历任太原经济技术开发区管委会副主任、太原武宿综合保税区专职副主任，山西转型综合改革示范区党工委委员、管委会副主任。2021年8月，任山西转型综合改革示范区党工委副书记、管委会副主任。(山西省纪委监委)
---
我的问题或指令：
帮我提取上述人物的中文名，英文名，性别，国籍，现任职位，最高学历，毕业院校，邮箱，电话
---
请根据上述参考信息回答我的问题或回复我的指令。前面的参考信息可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复,
你的回复："""
    final_result = ""
    for answer_result in llm.generatorAnswer(prompt=prompt,
                                                      history=chat_history,
                                                      streaming=streaming):
        resp = answer_result.llm_output["answer"]
        if "DONE" not in resp:
            final_result += json.loads(resp[6:])["answer"]
        # logging.info(resp)

    logging.info(f"final_result = {final_result}")
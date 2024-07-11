from typing import Optional, List

import json
import requests
import time
import random
import string
import hashlib
from requests.exceptions import RequestException
from qanything_kernel.utils.custom_log import debug_logger, qa_logger
from qanything_kernel.utils.general_utils import cur_func_name, shorten_data
import tiktoken
import os
from dotenv import load_dotenv
import copy
import traceback

load_dotenv()

openai_base_url = os.getenv('OPENAI_API_BASE')
openai_base_url = '/'.join(openai_base_url.split('/')[:-1])


class AnswerResult:
    """
    消息实体
    """
    history: List[List[str]] = []
    llm_output: Optional[dict] = None
    prompt: str = ""
    total_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0


def signedHeaders(appId, appKey):
    nonce = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    timestamp = str(int(time.time()))
    str2Sign = "appId={}&nonce={}&timestamp={}&appkey={}".format(appId, nonce, timestamp, appKey)
    sign = hashlib.md5(str2Sign.encode('utf-8')).hexdigest().upper()

    headers = {}
    headers['appId'] = appId
    headers['nonce'] = nonce
    headers['timestamp'] = timestamp
    headers['sign'] = sign
    headers['version'] = 'v2'
    headers["Content-Type"] = "application/json"
    headers["projectId"] = 'swpxGzN6Tixb95joisgSNuFaiqEAOhT94DDt'
    return headers


def retry_stream_requests(data_raw, headers):
    # base_url = "https://aigc-api.hz.netease.com/openai"  # 稳定正式服
    HANGYAN_CHATGPT_URL = openai_base_url + "/api/v2/text/chat-stream"  # 非流式
    linestr = ''
    try:
        response = requests.post(
            HANGYAN_CHATGPT_URL,
            headers=headers,
            json=data_raw,
            timeout=600,
            stream=True
        )
        response.raise_for_status()
        for line in response.iter_lines():
            # line 有可能是空的
            delta = {"answer": ""}
            if line:
                linestr = line.decode("utf-8")[6:]  # 跳过开头的"data: "

                if linestr.startswith("[DONE]"):  # 说明已经结束了
                    yield f"data: [DONE]\n\n"
                    break
                line_js = json.loads(linestr)
                if not line_js['detail']['choices']:
                    delta['total_tokens'] = line_js['detail']['usage']['totalTokens']
                    delta['completion_tokens'] = line_js['detail']['usage']['completionTokens']
                    delta['prompt_tokens'] = line_js['detail']['usage']['promptTokens']
                    yield "data: " + json.dumps(delta, ensure_ascii=False)
                    continue
                choice = line_js['detail']['choices'][0]
                if 'role' in choice['delta'] and choice['delta']['role']:
                    yield f"data: " + json.dumps(delta, ensure_ascii=False)
                    continue
                if 'content' not in choice["delta"]:
                    yield "data: " + json.dumps(delta, ensure_ascii=False)
                    continue
                text = choice["delta"]["content"]
                if "finish_reason" in choice:
                    finish_reason = choice["finish_reason"]
                    if finish_reason:
                        yield "data: " + json.dumps(delta, ensure_ascii=False)
                        continue
                delta["answer"] = text
                yield "data: " + json.dumps(delta, ensure_ascii=False)
    except Exception as e:
        # yield "data: " + json.dumps({"answer": "ERROR: request for hangyan llm failed."}, ensure_ascii=False)
        debug_logger.error("llm_for_online stream failed: {}".format(traceback.format_exc()))
        debug_logger.error(linestr)
        # yield "data: " + json.dumps({"answer": "ERROR: request for hangyan llm failed."}, ensure_ascii=False)
        raise RequestException("Error sending request: {}".format(e))


def retry_requests(data_raw, headers):
    MAX_RETRIES = 3
    # base_url = "https://aigc-api.hz.netease.com/openai"  # 稳定正式服
    HANGYAN_CHATGPT_URL = openai_base_url + "/api/v2/text/chat"  # 非流式
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.post(
                HANGYAN_CHATGPT_URL,
                headers=headers,
                json=data_raw,
                timeout=600
            )
            response.raise_for_status()
            return response
        except Exception as e:
            debug_logger.error("llm_for_online no stream failed: {}".format(traceback.format_exc()))
            retries += 1
    raise RequestException("No Stream LLM_FOR_ONLINE Max retries exceeded")
    # response = requests.Response()
    # response.status_code = 500
    # response._content = b"Max retries exceeded"
    # return response


class CustomLLM:
    api_key: str = "youdao-ai-IQHKj6rshAgud6oR"
    tokens_manager = {"gpt-3.5-turbo":{"token_window":4096,"max_token":512},
                    "gpt-3.5-turbo-0613":{"token_window":4096,"max_token":512},
                    "gpt-3.5-turbo-16k":{"token_window":8192,"max_token":1024},
                    "gpt-3.5-turbo-1106":{"token_window":8192,"max_token":1024},
                    "gpt-4":{"token_window":8192,"max_token":1024},
                    "gpt-4-1106-preview":{"token_window":8192,"max_token":1024},
                    "gpt-4-0125-preview":{"token_window":8192,"max_token":1024}}
    # model: str = "gpt-3.5-turbo-0613"
    # token_window: int = 4096
    # max_token: int = 512
    offcut_token: int = 50
    truncate_len: int = 50
    temperature: float = 0
    top_p: float = 1.0
    history: List[List[str]] = []
    history_len: int = 3
    appId: str = os.getenv("OPENAI_APPID")
    appKey: str = os.getenv("OPENAI_APPKEY")

    def __init__(self, model_name, max_token):
        self.model = model_name
        if max_token is None:
            self.max_token = self.tokens_manager[self.model]['max_token']
        else:
            self.max_token = max_token
        self.token_window = self.tokens_manager[self.model]['token_window']
        debug_logger.info(f"CustomLLM init! model: {self.model}, max_token: {self.max_token}, token_window: {self.token_window}")

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

    def generatorAnswer(self, prompt: str, history=None, streaming: bool = False):
        # history 需要复制一遍，防止引用导致的错误
        history = copy.deepcopy(history)

        if history is None:
            history = []

        if streaming:
            history += [[]]
            complete_answer = ""
            total_tokens = 0
            prompt_tokens = 0
            completion_tokens = 0
            for stream_resp in self.stream_chat(
                    prompt,
                    history=history[:-1],
                    max_length=self.max_token,
                    temperature=self.temperature,
            ):
                if stream_resp:
                    # print(stream_resp)
                    chunk_str = stream_resp[6:]
                    # 如果没有以[DONE]结尾咋办
                    if not chunk_str.startswith("[DONE]"):
                        chunk_js = json.loads(chunk_str)
                        complete_answer += chunk_js["answer"]
                        total_tokens = chunk_js.get("total_tokens", 0)
                        prompt_tokens = chunk_js.get("prompt_tokens", 0)
                        completion_tokens = chunk_js.get("completion_tokens", 0)

                history[-1] = [prompt, complete_answer]
                answer_result = AnswerResult()
                answer_result.history = history
                answer_result.total_tokens = total_tokens
                answer_result.prompt_tokens = prompt_tokens
                answer_result.completion_tokens = completion_tokens
                answer_result.llm_output = {"answer": stream_resp}
                answer_result.prompt = prompt
                yield answer_result
        else:
            response, total_tokens, prompt_tokens, completion_tokens = self.hangyan_chat(
                prompt,
                history,
                max_length=self.max_token,
                temperature=self.temperature
            )
            history += [[prompt, response]]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.total_tokens = total_tokens
            answer_result.prompt_tokens = prompt_tokens
            answer_result.completion_tokens = completion_tokens
            answer_result.llm_output = {"answer": response}
            answer_result.prompt = prompt
            yield answer_result

    def stream_chat(self,
                    prompt,
                    history,
                    max_length,
                    temperature):

        messages = []
        # print("stream_chat history: ", history)
        for msg in history:
            user_msg = {"role": "user", "content": msg[0] if msg[0] is not None else ""}
            assistant_msg = {"role": "assistant", "content": msg[1]}
            messages.append(user_msg)
            messages.append(assistant_msg)
        messages.append({"role": "user", "content": prompt})
        data_raw = {
            "messages": messages,
            "model": self.model,
            "maxTokens": max_length,
            "temperature": temperature,
            "topP": 1,
            "stop": None,
            "presencePenalty": 0,
            "frequencyPenalty": 0}

        MAX_RETRIES = 3
        retries = 0
        while retries < MAX_RETRIES:
            try:
                for res in retry_stream_requests(data_raw=data_raw,
                                                 headers=signedHeaders(self.appId, self.appKey)):
                    yield res
            except Exception as e:
                # debug_logger.error("llm_for_online stream failed: {}".format(traceback.format_exc()))
                retries += 1
        raise RequestException("Stream LLM_FOR_ONLINE Max retries exceeded")

    def hangyan_chat(self,
                     prompt,
                     history,
                     max_length,
                     temperature):

        messages = []
        for msg in history:
            user_msg = {"role": "user", "content": msg[0] if msg[0] != None else ""}
            assistant_msg = {"role": "assistant", "content": msg[1]}
            messages.append(user_msg)
            messages.append(assistant_msg)
        messages.append({"role": "user", "content": prompt})
        data_raw = {
            "messages": messages,
            "model": self.model,
            "maxTokens": max_length,
            "temperature": temperature,
            "topP": 1,
            "stop": None,
            "presencePenalty": 0,
            "frequencyPenalty": 0}

        debug_logger.info("hangyan data_raw: \n {}".format(shorten_data(data_raw)))
        response = retry_requests(data_raw=data_raw,
                                  headers=signedHeaders(self.appId, self.appKey))
        prompt_response = "ERROR: request for llm failed."
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        if response.status_code == 200:
            res = response.json()
            total_tokens = res["detail"]["usage"]["totalTokens"]
            prompt_tokens = res["detail"]["usage"]["promptTokens"]
            completion_tokens = res["detail"]["usage"]["completionTokens"] 
            response_dict = res["detail"]["choices"]
            if response_dict and len(response_dict) > 0:
                prompt_response = response_dict[0]["message"]["content"]
        return prompt_response, total_tokens, prompt_tokens, completion_tokens 

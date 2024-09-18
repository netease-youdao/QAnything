import traceback
from openai import OpenAI
from typing import List, Optional
import json
from qanything_kernel.connector.llm.base import AnswerResult
from qanything_kernel.utils.custom_log import debug_logger
import tiktoken


class OpenAILLM:
    offcut_token: int = 50
    stop_words: Optional[List[str]] = None

    def __init__(self, model, max_token, api_base, api_key, api_context_length, top_p, temperature):
        base_url = api_base
        api_key = api_key

        if max_token is not None:
            self.max_token = max_token
        if model is not None:
            self.model = model
        if api_context_length is not None:
            self.token_window = api_context_length
        if top_p is not None:
            self.top_p = top_p
        if temperature is not None:
            self.temperature = temperature
        self.use_cl100k_base = False
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except Exception as e:
            debug_logger.warning(f"{model} not found in tiktoken, using cl100k_base!")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.use_cl100k_base = True


        self.client = OpenAI(base_url=base_url, api_key=api_key)
        debug_logger.info(f"OPENAI_API_KEY = {api_key}")
        debug_logger.info(f"OPENAI_API_BASE = {base_url}")
        debug_logger.info(f"OPENAI_API_MODEL_NAME = {self.model}")
        debug_logger.info(f"OPENAI_API_CONTEXT_LENGTH = {self.token_window}")
        debug_logger.info(f"OPENAI_API_MAX_TOKEN = {self.max_token}")
        debug_logger.info(f"TOP_P = {self.top_p}")
        debug_logger.info(f"TEMPERATURE = {self.temperature}")

    @property
    def _llm_type(self) -> str:
        return "using OpenAI API serve as LLM backend"

    # 定义函数 num_tokens_from_messages，该函数返回由一组消息所使用的token数
    def num_tokens_from_messages(self, messages):
        total_tokens = 0
        for message in messages:
            if isinstance(message, dict):
                # 对于字典类型的消息，我们假设它包含 'role' 和 'content' 键
                for key, value in message.items():
                    total_tokens += 3  # role的开销(key的开销)
                    if isinstance(value, str):
                        tokens = self.tokenizer.encode(value, disallowed_special=())
                        total_tokens += len(tokens)
            elif isinstance(message, str):
                # 对于字符串类型的消息，直接编码
                tokens = self.tokenizer.encode(message, disallowed_special=())
                total_tokens += len(tokens)
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        if self.use_cl100k_base:
            total_tokens *= 1.2
        else:
            total_tokens *= 1.1  # 保留一定余量，由于metadata信息的嵌入导致token比计算的会多一些
        return int(total_tokens)

    def num_tokens_from_docs(self, docs):
        total_tokens = 0
        for doc in docs:
            # 对每个文本进行分词
            tokens = self.tokenizer.encode(doc.page_content, disallowed_special=())
            # 累加tokens数量
            total_tokens += len(tokens)
        if self.use_cl100k_base:
            total_tokens *= 1.2
        else:
            total_tokens *= 1.1  # 保留一定余量，由于metadata信息的嵌入导致token比计算的会多一些
        return int(total_tokens)

    async def _call(self, messages: List[dict], streaming: bool = False) -> str:
        try:

            if streaming:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    max_tokens=self.max_token,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stop=self.stop_words
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
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stop=self.stop_words
                )

                event_text = response.choices[0].message.content if response.choices else ""
                delta = {'answer': event_text}
                yield "data: " + json.dumps(delta, ensure_ascii=False)

        except Exception as e:
            debug_logger.info(f"Error calling OpenAI API: {traceback.format_exc()}")
            delta = {'answer': f"{e}"}
            yield "data: " + json.dumps(delta, ensure_ascii=False)

        finally:
            # debug_logger.info("[debug] try-finally")
            yield f"data: [DONE]\n\n"

    async def generatorAnswer(self, prompt: str,
                              history: List[List[str]] = [],
                              streaming: bool = False) -> AnswerResult:

        if history is None or len(history) == 0:
            history = [[]]
        # debug_logger.info(f"history_len: {self.history_len}")
        # debug_logger.info(f"prompt: {prompt}")
        debug_logger.info(f"prompt tokens: {self.num_tokens_from_messages([{'content': prompt}])}")
        # debug_logger.info(f"streaming: {streaming}")

        messages = []
        for pair in history[:-1]:
            question, answer = pair
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": prompt})
        # debug_logger.info(messages)
        prompt_tokens = self.num_tokens_from_messages(messages)
        total_tokens = 0
        completion_tokens = 0

        response = self._call(messages, streaming)
        complete_answer = ""
        async for response_text in response:
            if response_text:
                chunk_str = response_text[6:]
                if not chunk_str.startswith("[DONE]"):
                    chunk_js = json.loads(chunk_str)
                    complete_answer += chunk_js["answer"]
                completion_tokens = self.num_tokens_from_messages([complete_answer])
                total_tokens = prompt_tokens + completion_tokens

            history[-1] = [prompt, complete_answer]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": response_text}
            answer_result.prompt = prompt
            answer_result.total_tokens = total_tokens
            answer_result.completion_tokens = completion_tokens
            answer_result.prompt_tokens = prompt_tokens
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
    final_result = ""
    for answer_result in llm.generatorAnswer(prompt=prompt, history=chat_history, streaming=streaming):
        resp = answer_result.llm_output["answer"]
        if "DONE" not in resp:
            final_result += json.loads(resp[6:])["answer"]
        debug_logger.info(resp)

    debug_logger.info(f"final_result = {final_result}")

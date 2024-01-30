import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import os
import psutil
import queue
import string
import signal
import sys
import traceback
import time
import threading

import sanic
from sanic import Sanic, Request
from sanic.response import ResponseStream
from collections import OrderedDict
from datetime import datetime
from transformers import AutoTokenizer
from tritonclient.utils import InferenceServerException
from typing import List, Tuple, Dict, Optional, Any
from urllib.parse import unquote

WORKER_VERSION = "llm_v1.0.0_231221_fc212a"

sys.path.append("./")
from modeling_qwen import QwenTritonModel
from utils import log_timestamp, CODES

logging.getLogger().setLevel(logging.INFO)
global_counter = 0
model_semaphore = None

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=36001)
parser.add_argument(
    "--model-path",
    type=str,
    default="tokenizer_assets/",
    help="The path to the tokenizer weights",
)
parser.add_argument(
    "--model-url",
    type=str,
    default="localhost:10001",
    help="url of tritonserver",
)
parser.add_argument(
    "--limit-model-concurrency",
    type=int,
    default=40,
    help="limit to the maximum number of semaphore"
)
args = parser.parse_args()

model = QwenTritonModel(model_url=args.model_url)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, do_lower_case=False,
                                            strip_accents=False)

app = Sanic("LLMService")


def signal_handler(signum, frame) -> None:
    signal_ = "unknown signal"
    if signum == signal.SIGINT:
        signal_ = "signal.SIGINT"
    elif signum == signal.SIGTERM:
        signal_ = "signal.SIGTERM"

    for proc_ in mp.active_children():
        os.kill(proc_.pid, signal.SIGINT)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def is_process_running(pid: int) -> bool:
    try:
        psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    else:
        return True


def generator_llm(params: OrderedDict) -> str:
    def insert_error(resp_data: dict, error_enum) -> None:
        resp_data["text"] = error_enum.desc
        resp_data["error_code"] = error_enum.code

    def get_response(resp_data: Dict[str, Any]) -> str:
        return "data: " + json.dumps(resp_data) + "\n\n"

    def parse_params(params: OrderedDict) -> Tuple:

        assert (isinstance(params, dict) or isinstance(params,
                                                       OrderedDict)), "params were expected as dict or OrderedDict, but got {}.".format(
            type(params))

        if type(params.get('hist_messages', {})) == str:
            unquote_messages = unquote(params.get('hist_messages', {}))
            params['hist_messages'] = json.loads(unquote_messages)

        prompt = params.get('prompt', "")
        if params.get('url_encode', False):
            params['prompt'] = unquote(prompt)
        else:
            params['prompt'] = prompt

        request_id = str(params.get("request_id", "-1"))
        max_new_tokens = int(params.get("max_new_tokens", model.max_new_tokens))
        max_new_tokens = min(max_new_tokens, model.max_new_tokens)
        temperature = float(params.get("temperature", 0.6))
        repetition_penalty = float(params.get("repetition_penalty", 1.2))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", 4))
        random_seed_ = int(params.get("random_seed", -1))
        if random_seed_ == -1:
            random_seed_ = 231221
        if request_id == "-1":
            request_id = random_seed_

        params["request_id"] = request_id
        params["random_seed"] = random_seed_

        infer_decode_args = {
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "random_seed_": random_seed_,
            "request_id": request_id
        }

        return tuple([infer_decode_args, params])

    ## 解析参数
    infer_decode_args, params = parse_params(params)
    request_id = infer_decode_args.get("request_id")
    check_in = int(params.get("check_in", 1))
    chunk_out = bool(params.get("chunk_out", True))

    ## 构造 Prompt
    message = params.get("prompt", "")
    hist_messages = params.get("hist_messages", OrderedDict())
    max_new_tokens = infer_decode_args.get("max_new_tokens", model.max_new_tokens)
    query_prompt_tuple = model.get_multiround_template(message, max_new_tokens, hist_messages=hist_messages,
                                                       prefix=None, response="")
    messages = query_prompt_tuple[1]
    query = query_prompt_tuple[0]
    input_len = len(query)

    ## 请求 TritonServer 获取流式推理结果
    response_data = {
        "random_seed": infer_decode_args.get("random_seed", -1),
        "request_id": request_id,
        "version": WORKER_VERSION,
    }

    result_queue = queue.Queue()
    proc = threading.Thread(target=model.chat_stream, args=(query, result_queue), kwargs=infer_decode_args)
    proc.start()
    proc_pid = threading.get_native_id()
    request_id = "{}_{}".format(request_id, proc_pid)

    bytes_ids = []
    decode_len = 0
    bytes_len = 0
    punc = string.punctuation
    try:
        while True:
            res = result_queue.get()
            if res is None:
                break
            if isinstance(res, List):

                if chunk_out:
                    bytes_ids += res[input_len + decode_len + bytes_len:]
                    decoding = tokenizer.decode(bytes_ids, skip_special_tokens=True)
                    if isinstance(decoding, bytes) or (isinstance(decoding, str) and '�' in decoding):
                        bytes_len = len(bytes_ids)
                        continue
                    else:
                        decode_len += len(bytes_ids)
                        bytes_len = 0
                        bytes_ids = []
                else:
                    output_len = len(res[input_len:])
                    decoding = tokenizer.decode(res[input_len:], skip_special_tokens=True)
                    decoding = model.process_response(decoding)

                response_data["text"] = decoding
                response_data["error_code"] = CODES.SUCCESS.code

                try:
                    response_instance = get_response(response_data)
                except:
                    insert_error(response_data, CODES.JSON_FORMAT_ERROR)
                    response_instance = get_response(response_data)

                yield response_instance

            elif isinstance(res, InferenceServerException):
                insert_error(response_data, CODES.TRITON_INFERENCE_ERROR)
                yield get_response(response_data)

            elif isinstance(res, Tuple):
                insert_error(response_data, CODES.TRITON_CALLBACK_ERROR)
                yield get_response(response_data)

            else:
                insert_error(response_data, CODES.UNKNOWN_ERROR)
                yield get_response(response_data)

        try:
            proc.join()
        except Exception as e:
            traceback.print_exc()

    except Exception as e:
        if isinstance(e, RuntimeError):
            exception_enum = CODES.RUNTIME_ERROR
        elif isinstance(e, TypeError):
            exception_enum = CODES.TYPE_ERROR
        else:
            exception_enum = CODES.UNKNOWN_ERROR
        insert_error(response_data, exception_enum)

        yield get_response(response_data)


class WorkerStatus(object):

    def __init__(self, limit_model_concurrency: int) -> None:
        self.limit_model_concurrency = limit_model_concurrency

    def _get_queue_length(self) -> int:
        global model_semaphore

        if (
                model_semaphore is None
                or model_semaphore._value is None
                or model_semaphore._waiters is None
        ):
            return 0
        else:
            return (
                    self.limit_model_concurrency
                    - model_semaphore._value
                    + len(model_semaphore._waiters)
            )

    def get_status(self) -> Dict:
        global model_semaphore, current_connections, qps
        """
        信号量限流参数说明:
        limit_model_concurrency 表示允许同时访问模型的请求数量，并发请求限制的最大数量
        model_semaphore._value 表示当前可获取到信号量许可的数量
        len(model_semaphore._waiters) 表示正在等待获取信号量许可的请求数量
        
        """
        return {
            "time_stamp": datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "limit_model_concurrency": self.limit_model_concurrency,
            "semaphore_value": model_semaphore._value,
            "semaphore_locked": model_semaphore.locked(),
            "semaphore_waiters": len(model_semaphore._waiters),
            "queue_length": self._get_queue_length(),
        }


status = WorkerStatus(args.limit_model_concurrency)

async def release_model_semaphore():
    global model_semaphore, global_counter
    global_counter -= 1
    model_semaphore.release()


@app.post("/worker_check_input")
async def check_input(request: Request):
    params = await request.json()
    assert (isinstance(params, dict) or isinstance(params,
                                                   OrderedDict)), "params were expected as dict or OrderedDict, but got {}.".format(
        type(params))

    if type(params.get('hist_messages', {})) == str:
        unquote_messages = unquote(params.get('hist_messages', {}))
        params['hist_messages'] = json.loads(unquote_messages)

    prompt = params.get('prompt', "")
    if params.get('url_encode', False):
        params['prompt'] = unquote(prompt)
    else:
        params['prompt'] = prompt

    message = params.get("prompt", "")
    hist_messages = params.get("hist_messages", OrderedDict())
    max_new_tokens = params.get("max_new_tokens", model.max_new_tokens)

    ## 返回 prompt + hist_messages 的 token 总数
    tokens_number = model.check_query_tokens(message, max_new_tokens, hist_messages=hist_messages, prefix=None,
                                             response="")

    return sanic.response.text(str(tokens_number))


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = request.json
    print("params:", params)

    if model_semaphore is None:
        limit_model_concurrency = args.limit_model_concurrency
        model_semaphore = asyncio.Semaphore(limit_model_concurrency)
    await model_semaphore.acquire()

    async def generate_answer(response):
        for chunk in generator_llm(params):
            await response.write(chunk)
            await asyncio.sleep(0.001)
        await release_model_semaphore()
        await response.eof()
        # await asyncio.sleep(0.001)

    response = ResponseStream(generate_answer, content_type='text/event-stream')
    # response.headers['Cache-Control'] = 'no-cache'
    return response


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return sanic.response.json(status.get_status())


@app.get("/health/ready")
async def api_health_check(request: Request):
    return sanic.response.text("HTTP/1.1 200 OK")


if __name__ == "__main__":
    app.run(host=args.host, port=args.port, workers=4, debug=False)

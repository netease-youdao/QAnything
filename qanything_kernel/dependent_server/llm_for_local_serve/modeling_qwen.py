import enum
import google.protobuf.json_format
import json
import numpy as np
import queue
import re
import sys
import threading
import tritonclient

from copy import deepcopy
from functools import partial
from tritonclient import grpc as grpcclient
from tritonclient.grpc.service_pb2 import ModelInferResponse
from tritonclient.utils import np_to_triton_dtype
from typing import List, Tuple, Optional
from collections import OrderedDict
from transformers import AutoTokenizer

sys.path.append("/")
from template import get_template_and_fix_tokenizer


@enum.unique
class ErrorCode(enum.Enum):
    UNKNOWN_ERROR = 1
    INFERENCE_ERROR = 2
    CALLBACK_ERROR = 3

    # 将本地的错误码翻译成上层的错误码
    def to_codes(local_error_code: int) -> str:
        if local_error_code == ErrorCode.INFERENCE_ERROR.value:
            return CODES.TRITON_INFERENCE_ERROR
        elif local_error_code == ErrorCode.CALLBACK_ERROR.value:
            return CODES.TRITON_CALLBACK_ERROR
        return CODES.UNKNOWN_ERROR


class QwenTritonModel(object):
    def __init__(self, model_url: str="localhost:10001", model_path: str="tokenizer_assets") -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, do_lower_case=False, strip_accents=False)
        tokenizer.pad_token_id = 151643
        self.start_id = tokenizer.bos_token_id
        self.end_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        self.template = get_template_and_fix_tokenizer("chatml", tokenizer)
        
        self.model_url = model_url
        self.model_name = "base"
        self.seq_length = 4096
        self.max_new_tokens = 300

    def _fill_input(self, name: str, data: np.ndarray) -> grpcclient.InferInput:
        infer_input = grpcclient.InferInput(name, data.shape, np_to_triton_dtype(data.dtype))
        infer_input.set_data_from_numpy(data)
        return infer_input

    def _stream_callback(self, queue: queue.Queue, request_id: str, result: grpcclient.InferResult, error: Optional[tritonclient.utils.InferenceServerException]=None) -> None:
        try:
            if error:
                queue.put(error)
            else:
                res = result.get_response(as_json=True)
                message = ModelInferResponse()
                google.protobuf.json_format.Parse(json.dumps(res), message)
                result = grpcclient.InferResult(message)

                idx = result.as_numpy("sequence_length")[0, 0]
                tokens = result.as_numpy("output_ids")[0, 0, :idx]
                output = tokens.tolist()
                queue.put(output)
                
        except Exception as e:
            queue.put(tuple([ErrorCode.CALLBACK_ERROR.name, ErrorCode.CALLBACK_ERROR.value]))

    def process_response(self, response: str) -> str:
        
        response = response.strip()
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]

        ## 中文 response 字符串半角标点替换成全角标点
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response

    def check_query_tokens(self, query: str, max_new_tokens: int, hist_messages: Optional[OrderedDict]=None, prefix: Optional[str]=None, response: Optional[str]="") -> int:

        history = []
        if hist_messages is not None:
            suffix_hist = "\n---\n"
            hist_contents = "历史对话："
            for iter, content in hist_messages.items():
                question = content.get("user", "")
                answer = content.get("chatbot", "") 
                hist_contents += f"\n问：{question}\n答：{answer}"

            query = hist_contents + suffix_hist + query
        else:
            query = f"历史对话：\n无\n---\n" + query

        input_ids = []
        target_ids = []
        prompt = ""
        input_ids, target_ids = self.template.encode_multiturn(self.tokenizer, query, response, history, prefix)[0]
        
        return len(input_ids)
    
    def get_multiround_template(self, query: str, max_new_tokens: int, hist_messages: Optional[OrderedDict]=None, prefix: Optional[str]=None, response: Optional[str]="") -> Tuple:

        history = []
        if hist_messages is not None:            
            suffix_hist = "\n---\n"
            hist_contents = "历史对话："
            for _, content in hist_messages.items():
                question = content.get("user", "")
                answer = content.get("chatbot", "") 
                hist_contents += f"\n问：{question}\n答：{answer}"
            query = hist_contents + suffix_hist + query
        else:
            query = f"历史对话：\n无\n---\n" + query
        
        # system prefix
        prefix_ids = [151643, 151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151643, 198, 151644, 872, 198]
        suffix_ids = [151645, 198, 151644, 77091, 198]

        input_ids = []
        target_ids = []
        prompt = ""
        input_ids, target_ids = self.template.encode_multiturn(self.tokenizer, query, response, history, prefix)[0]
        prompt = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        query = deepcopy(input_ids)

        q_len = len(query)
        max_new_tokens = min(max_new_tokens, self.max_new_tokens)
        if q_len >= self.seq_length - max_new_tokens:
            query = prefix_ids + query[-(self.seq_length - max_new_tokens):]
            q_len = len(query)

        if q_len == 0:
            query = deepcopy(prefix_ids)
            q_len = len(query)

        return tuple([query, prompt])

    def chat_stream(self, query: List[int], result_queue: queue.Queue, max_new_tokens: int=300, top_k: int=4, top_p: float=1.0, temperature: float=0.6, repetition_penalty: float=1.2, random_seed_: int=0, request_id: str="231221") -> None:

        tid = threading.get_native_id()
        request_id = "{}_{}".format(request_id, tid)
        q_len = len(query)

        if max_new_tokens < 0:
            max_new_tokens = 0

        with grpcclient.InferenceServerClient(self.model_url, verbose=False) as client:
            request_data = []
            request = np.array([query]).astype(np.uint32)
            request_len = np.array([[len(query)]]).astype(np.uint32)
            request_output_len = np.array([[min(self.max_new_tokens, max_new_tokens)]]).astype(np.uint32)
            top_k = np.array([[top_k]]).astype(np.uint32)
            top_p = np.array([[top_p]]).astype(np.float32)
            temperature = np.array([[temperature]]).astype(np.float32)
            repetition_penalty = np.array([[repetition_penalty]]).astype(np.float32)
            random_seed =  random_seed_ * np.ones([request.shape[0], 1]).astype(np.uint64)
            start_id = self.start_id * np.ones([request.shape[0], 1]).astype(np.uint32)
            end_id = self.end_id * np.ones([request.shape[0], 1]).astype(np.uint32)

            request_data.append(self._fill_input('random_seed', random_seed))
            request_data.append(self._fill_input('input_ids', request))
            request_data.append(self._fill_input('input_lengths', request_len))
            request_data.append(self._fill_input('request_output_len', request_output_len))
            request_data.append(self._fill_input('start_id', start_id))
            request_data.append(self._fill_input('end_id', end_id))
            request_data.append(self._fill_input('runtime_top_k', top_k))
            request_data.append(self._fill_input('runtime_top_p', top_p))
            request_data.append(self._fill_input('temperature', temperature))
            request_data.append(self._fill_input('repetition_penalty', repetition_penalty))

            client.start_stream(callback=partial(self._stream_callback, result_queue, request_id), compression_algorithm='gzip')
            client.async_stream_infer(self.model_name, request_data, request_id="rid_{}".format(request_id))
        
        result_queue.put(None)



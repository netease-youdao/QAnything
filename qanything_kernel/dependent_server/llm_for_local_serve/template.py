## Reference to: https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llmtuner/data/template.py

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging as logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@dataclass
class Template:

    prefix: List[Union[str, Dict[str, str]]]
    prompt: List[Union[str, Dict[str, str]]]
    sep: List[Union[str, Dict[str, str]]]
    stop_words: List[str]
    use_history: bool

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        prefix: Optional[str] = None
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        prefix, history = self._format(query, resp, history, prefix)
        encoded_pairs = self._encode(tokenizer, prefix, history)
        prompt_ids = []
        for query_ids, resp_ids in encoded_pairs[:-1]:
            prompt_ids = prompt_ids + query_ids + resp_ids
        prompt_ids = prompt_ids + encoded_pairs[-1][0]
        return prompt_ids, encoded_pairs[-1][1]

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        prefix: Optional[str] = None
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        prefix, history = self._format(query, resp, history, prefix)
        encoded_pairs = self._encode(tokenizer, prefix, history)
        return encoded_pairs

    def _format(
        self,
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        prefix: Optional[str] = None
    ) -> Tuple[List[Union[str, Dict[str, str]]], List[Tuple[str, str]]]:
        r"""
        Aligns inputs to a special format.
        """
        prefix = [prefix] if prefix else self.prefix # use prefix if provided
        history = history if (history and self.use_history) else []
        history = history + [(query, resp)]
        return prefix, history

    def _get_special_ids(
        self,
        tokenizer: "PreTrainedTokenizer"
    ) -> Tuple[List[int], List[int]]:
        if tokenizer.bos_token_id:
            bos_ids = [tokenizer.bos_token_id]
        else:
            bos_ids = [] # bos token is optional

        if tokenizer.eos_token_id:
            eos_ids = [tokenizer.eos_token_id]
        else:
            raise ValueError("EOS token is required.")

        return bos_ids, eos_ids

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        prefix: List[Union[str, Dict[str, str]]],
        history: List[Tuple[str, str]]
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        """
        bos_ids, eos_ids = self._get_special_ids(tokenizer)
        sep_ids = self._convert_inputs_to_ids(tokenizer, context=self.sep)
        encoded_pairs = []
        for turn_idx, (query, resp) in enumerate(history):
            if turn_idx != 0:
                prefix_ids = sep_ids
            elif prefix:
                prefix_ids = self._convert_inputs_to_ids(tokenizer, context=prefix) + eos_ids + sep_ids
            else:
                prefix_ids = []

            query_ids = self._convert_inputs_to_ids(tokenizer, context=self.prompt, query=query)
            resp_ids = self._convert_inputs_to_ids(tokenizer, context=[resp])
            encoded_pairs.append((bos_ids + prefix_ids + query_ids, resp_ids + eos_ids))
        return encoded_pairs

    def _convert_inputs_to_ids(
        self,
        tokenizer: "PreTrainedTokenizer",
        context: List[Union[str, Dict[str, str]]],
        query: Optional[str] = ""
    ) -> List[int]:
        r"""
        Converts context to token ids.
        """
        if hasattr(tokenizer, "tokenizer"): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=False)

        token_ids = []
        for elem in context:
            if isinstance(elem, str):
                elem = elem.replace("{{query}}", query, 1)
                token_ids = token_ids + tokenizer.encode(elem, **kwargs)
            elif isinstance(elem, dict):
                token_ids = token_ids + [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            else:
                raise NotImplementedError
        return token_ids


templates: Dict[str, Template] = {}


def register_template(
    name: str,
    prefix: List[Union[str, Dict[str, str]]],
    prompt: List[Union[str, Dict[str, str]]],
    sep: List[Union[str, Dict[str, str]]],
    stop_words: List[str],
    use_history: bool
) -> None:
    template_class = Template
    templates[name] = template_class(
        prefix=prefix,
        prompt=prompt,
        sep=sep,
        stop_words=stop_words,
        use_history=use_history
    )


def get_template_and_fix_tokenizer(
    name: str,
    tokenizer: "PreTrainedTokenizer"
) -> Template:
    template = templates.get(name, None)
    assert template is not None, "Template {} does not exist.".format(name)

    if tokenizer.eos_token_id is None: # inplace method
        if len(template.stop_words):
            tokenizer.eos_token = template.stop_words[0]
        else:
            tokenizer.eos_token = "<|endoftext|>"
        logger.info("Add eos token: {}".format(tokenizer.eos_token))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    tokenizer.add_special_tokens(dict(additional_special_tokens=template.stop_words))
    return template


register_template(
    name="default",
    prefix=[
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ],
    prompt=[
        "Human: {{query}}\nAssistant: "
    ],
    sep=[
        "\n"
    ],
    stop_words=[],
    use_history=True
)


register_template(
    name="chatml",
    prefix=[
        {"token": "<|im_start|>"},
        "system\nYou are a helpful assistant."
    ],
    prompt=[
        {"token": "<|im_start|>"},
        "user\n{{query}}",
        {"token": "<|im_end|>"},
        "\n",
        {"token": "<|im_start|>"},
        "assistant\n"
    ],
    sep=[
        "\n"
    ],
    stop_words=[
        "<|im_end|>"
    ],
    use_history=True
)

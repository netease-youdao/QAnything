import os
from dotenv import load_dotenv
from .llm_for_openai_api import OpenAILLM
from .llm_for_vllm_openai_api import OpenAILLM as VllmOpenAILLM

load_dotenv()
RUNTIME_BACKEND = os.getenv("RUNTIME_BACKEND")

if RUNTIME_BACKEND == "default":
    from .llm_for_local import ZiyueLLM
else: # hf/vllm
    from .llm_for_fastchat import OpenAICustomLLM as ZiyueLLM

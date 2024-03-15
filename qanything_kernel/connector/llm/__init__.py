import platform
if platform.system() == "Linux":
    from .llm_for_fastchat import OpenAICustomLLM
elif platform.system() == "Darwin":
    from .llm_for_llamacpp import LlamaCPPCustomLLM
from .llm_for_openai_api import OpenAILLM

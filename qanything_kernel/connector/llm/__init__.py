import platform
import qanything_kernel.connector.gpuinfo.global_vars as global_vars

gpu_type = global_vars.get_gpu_type()
if gpu_type == "nvidia":
    from .llm_for_fastchat import OpenAICustomLLM
elif gpu_type == "intel":
    from .llm_for_bigdl import BigDLCustomLLM
elif gpu_type == "metal":
    from .llm_for_llamacpp import LlamaCPPCustomLLM
from .llm_for_openai_api import OpenAILLM

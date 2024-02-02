

## Table of Contents

- [QAnything Service Startup Command Usage](#QAnything-Service-Startup-Command-Usage)
- [Supported Pulic LLM using FastChat API](#Supported-Pulic-LLM-using-FastChat-API-with-Huggingface-Transformers/vllm-runtime-backend)
- [Tricks for saving GPU VRAM](#Tricks-for-saving-GPU-VRAM)
- [Comming Soon](#Comming-Soon)


## QAnything Service Startup Command Usage

```bash
Usage: bash run.sh [-c <llm_api>] [-i <device_id>] [-b <runtime_backend>] [-m <model_name>] [-t <conv_template>] [-p <tensor_parallel>] [-r <gpu_memory_utilization>]

-c <llm_api>: "Options {local, cloud} to specify the llm API mode, default is 'local'. If set to '-c cloud', please mannually set the environments {OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_MODEL_NAME, OPENAI_API_CONTEXT_LENGTH} into .env fisrt."
-i <device_id>: "Specify argument GPU device_id."
-b <runtime_backend>: "Specify argument LLM inference runtime backend, options={default, hf, vllm}"
-m <model_name>: "Specify argument the model name to load public LLM model using FastChat serve API, options={Qwen-7B-Chat, deepseek-llm-7b-chat, ...}"
-t <conv_template>: "Specify argument the conversation template according to the public LLM model when using FastChat serve API, options={qwen-7b-chat, deepseek-chat, ...}"
-p <tensor_parallel>: "Use options {1, 2} to set tensor parallel parameters for vllm backend when using FastChat serve API, default tensor_parallel=1"
-r <gpu_memory_utilization>: "Specify argument gpu_memory_utilization (0,1] for vllm backend when using FastChat serve API, default gpu_memory_utilization=0.81"
-h: "Display help usage message"
```

| Service Startup Command                                                                 | GPUs | LLM Runtime Backend      | LLM model                        |
| --------------------------------------------------------------------------------------- | -----|--------------------------| -------------------------------- |
| ```bash ./run.sh -c cloud -i 0 -b default```                                            | 1    | OpenAI API               | OpenAI API                       |
| ```bash ./run.sh -c local -i 0 -b default```                                            | 1    | FasterTransformer        | Qwen-7B-QAnything                |
| ```bash ./run.sh -c local -i 0 -b hf -m MiniChat-2-3B -t minichat```                    | 1    | Huggingface Transformers | Public LLM (e.g., MiniChat-2-3B) |
| ```bash ./run.sh -c local -i 0 -b vllm -m MiniChat-2-3B -t minichat -p 1 -r 0.81```     | 1    | vllm                     | Public LLM (e.g., MiniChat-2-3B) |
| ```bash ./run.sh -c local -i 0,1 -b default```                                          | 2    | FasterTransformer        | Qwen-7B-QAnything                |
| ```bash ./run.sh -c local -i 0,1 -b hf -m MiniChat-2-3B -t minichat```                  | 2    | Huggingface Transformers | Public LLM (e.g., MiniChat-2-3B) |
| ```bash ./run.sh -c local -i 0,1 -b vllm -m MiniChat-2-3B -t minichat -p 1 -r 0.81```   | 2    | vllm                     | Public LLM (e.g., MiniChat-2-3B) |
| ```bash ./run.sh -c local -i 0,1 -b vllm -m MiniChat-2-3B -t minichat -p 2 -r 0.81```   | 2    | vllm                     | Public LLM (e.g., MiniChat-2-3B) |

```bash
Note: You can choose the most suitable Service Startup Command based on your own device conditions.
(1) Local Embedding/Rerank will run on device gpu_id_1 when setting "-i 0,1", otherwise using gpu_id_0 as default.
(2) When setting "-c cloud" that will use local Embedding/Rerank and OpenAI LLM API, which only requires about 4GB VRAM (recommend for GPU device VRAM <= 8GB).
(3) When you use OpenAI LLM API, you will be required to enter {OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_MODEL_NAME, OPENAI_API_CONTEXT_LENGTH} immediately.
(4) "-b hf" is the most recommended way for running public LLM inference for its compatibility but with poor performance.
(5) When you choose a public Chat LLM for QAnything system, you should take care of a more suitable **PROMPT_TEMPLATE** (/path/to/QAnything/qanything_kernel/configs/model_config.py) setting considering different LLM models.
```

## Supported Pulic LLM using FastChat API with Huggingface Transformers/vllm runtime backend

| model_name                                | conv_template       | Supported Pulic LLM List                                                        |
|-------------------------------------------|---------------------|---------------------------------------------------------------------------------|
| Qwen-7B-QAnything                         | qwen-7b-qanything   | [Qwen-7B-QAnything](https://huggingface.co/netease-youdao/Qwen-7B-QAnything)    |        
| Qwen-1_8B-Chat/Qwen-7B-Chat/Qwen-14B-Chat | qwen-7b-chat        | [Qwen](https://huggingface.co/Qwen)                                             |        
| Baichuan2-7B-Chat/Baichuan2-13B-Chat      | baichuan2-chat      | [Baichuan2](https://huggingface.co/baichuan-inc)                                | 
| MiniChat-2-3B                             | minichat            | [MiniChat](https://huggingface.co/GeneZC/MiniChat-2-3B)                         |
| deepseek-llm-7b-chat                      | deepseek-chat       | [Deepseek](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat)             | 
| Yi-6B-Chat                                | Yi-34b-chat         | [Yi](https://huggingface.co/01-ai/Yi-6B-Chat)                                   | 
| chatglm3-6b                               | chatglm3            | [ChatGLM3](https://huggingface.co/THUDM/chatglm3-6b)                            | 
| ...                          ```check or add conv_template for more LLMs in "/path/to/QAnything/third_party/FastChat/fastchat/conversation.py"``` |


### 1. Run QAnything using FastChat API with **Huggingface transformers** runtime backend (recommend for GPU device with VRAM <= 16GB).
#### 1.1 Run Qwen-7B-QAnything
```bash
## Step 1. Download the public LLM model (e.g., Qwen-7B-QAnything) and save to "/path/to/QAnything/assets/custom_models"
## (Optional) Download Qwen-7B-QAnything from ModelScope: https://www.modelscope.cn/models/netease-youdao/Qwen-7B-QAnything
## (Optional) Download Qwen-7B-QAnything from Huggingface: https://huggingface.co/netease-youdao/Qwen-7B-QAnything
cd /path/to/QAnything/assets/custom_models
git clone https://huggingface.co/netease-youdao/Qwen-7B-QAnything

## Step 2. Execute the service startup command.  Here we use "-b hf" to specify the Huggingface transformers backend.
## Here we use "-b hf" to specify the transformers backend that will load model in 8 bits but do bf16 inference as default for saving VRAM.
cd /path/to/QAnything
bash ./run.sh -c local -i 0 -b hf -m Qwen-7B-QAnything -t qwen-7b-qanything

```

#### 1.2 Run a public LLM model (e.g., MiniChat-2-3B)
```bash
## Step 1. Download the public LLM model (e.g., MiniChat-2-3B) and save to "/path/to/QAnything/assets/custom_models"
cd /path/to/QAnything/assets/custom_models
git clone https://huggingface.co/GeneZC/MiniChat-2-3B

## Step 2. Execute the service startup command.  Here we use "-b hf" to specify the Huggingface transformers backend.
## Here we use "-b hf" to specify the transformers backend that will load model in 8 bits but do bf16 inference as default for saving VRAM.
cd /path/to/QAnything
bash ./run.sh -c local -i 0 -b hf -m MiniChat-2-3B -t minichat

```

### 2. Run QAnything using FastChat API with **vllm** runtime backend (recommend for GPU device with enough VRAM).
#### 2.1 Run Qwen-7B-QAnything
```bash
## Step 1. Download the public LLM model (e.g., Qwen-7B-QAnything) and save to "/path/to/QAnything/assets/custom_models"
## (Optional) Download Qwen-7B-QAnything from ModelScope: https://www.modelscope.cn/models/netease-youdao/Qwen-7B-QAnything
## (Optional) Download Qwen-7B-QAnything from Huggingface: https://huggingface.co/netease-youdao/Qwen-7B-QAnything
cd /path/to/QAnything/assets/custom_models
git clone https://huggingface.co/netease-youdao/Qwen-7B-QAnything

## Step 2. Execute the service startup command.  Here we use "-b vllm" to specify the vllm backend.
## Here we use "-b vllm" to specify the vllm backend that will do bf16 inference as default.
## Note you should adjust the gpu_memory_utilization yourself according to the model size to avoid out of memory (e.g., gpu_memory_utilization=0.81 is set default for 7B. Here, gpu_memory_utilization is set to 0.85 by "-r 0.85").
cd /path/to/QAnything
bash ./run.sh -c local -i 0 -b vllm -m Qwen-7B-QAnything -t qwen-7b-qanything -p 1 -r 0.85

```

#### 2.2 Run a public LLM model (e.g., MiniChat-2-3B)
```bash
## Step 1. Download the public LLM model (e.g., MiniChat-2-3B) and save to "/path/to/QAnything/assets/custom_models"
cd /path/to/QAnything/assets/custom_models
git clone https://huggingface.co/GeneZC/MiniChat-2-3B

## Step 2. Execute the service startup command. 
## Here we use "-b vllm" to specify the vllm backend that will do bf16 inference as default.
## Note you should adjust the gpu_memory_utilization yourself according to the model size to avoid out of memory (e.g., gpu_memory_utilization=0.81 is set default for 7B. Here, gpu_memory_utilization is set to 0.5 by "-r 0.5").
cd /path/to/QAnything
bash ./run.sh -c local -i 0 -b vllm -m MiniChat-2-3B -t minichat -p 1 -r 0.5

## (Optional) Step 2. Execute the service startup command to specify the vllm backend by "-i 0,1 -p 2". It will do faster inference by setting a tensor parallel mode on 2 GPUs.
## bash ./run.sh -c local -i 0,1 -b vllm -m MiniChat-2-3B -t minichat -p 2 -r 0.5

```

## Tricks for saving GPU VRAM
```bash
## Trick 1. (Recommend for VRAM<=12 GB or GPU Compute Capability < 7.5) Using PaddleOCR serve in CPU mode **use_gpu=False** in '/path/to/QAnything/qanything_kernel/dependent_server/ocr_serve/ocr_server.py'
# GPU Compute Capability: https://developer.nvidia.com/cuda-gpus
# Note that **use_gpu=False** must be set when using RTX-1080Ti GPU, otherwise PaddleOCR will always return **empty ocr result** when using **use_gpu=True**.
ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False, show_log=False)

## Trick 2. Try 1.8B/3B size LLM, such as Qwen-1.8B-Chat and MiniChat-2-3B.

## Trick 3. Try to limit the max length of context window by decreasing the value of **token_window** and increasing that of **offcut_token**
# /path/to/QAnything/qanything_kernel/connector/llm/llm_for_fastchat.py
# /path/to/QAnything/qanything_kernel/connector/llm/llm_for_local.py

## Trick 4. Try INT4-Weight-Only Quantization methods such as GPTQ/AWQ. You should take care of the sampling parameters considering possible loss of accuracy.

```


## Comming Soon
<details><summary>Feature Request</summary>

- Support one-api interface to add more business LLM API (https://github.com/songquanpeng/one-api).
- Support more runtime backends, such as llama.cpp (https://github.com/ggerganov/llama.cpp) and sglang (https://github.com/sgl-project/sglang).
- ...
  
</details>

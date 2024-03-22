# QAnything depends on Python 3.10+

## Installation

Requirements:

  - Python 3.10+ (suggest using conda to manage python environment)
  - System 
      - Linux: glibc 2.28+ and Cuda 12.0+ (if using GPU)
      - Windows: WSL with Ubuntu 20.04+ and GEFORCE EXPERIENCE 535.104+ (if using GPU) 
      - MacOS: M1/M2/M3 Mac with Xcode 15.0+

To install the package, run:

```bash
conda create -n qanything-python python=3.10
conda activate qanything-python
git clone -b qanything-python-v1.3.0 https://github.com/netease-youdao/QAnything.git
cd QAnything
pip install -e .
```

## Run With 4B LLM (Qwen1.5-4B-Chat) On Windows WSL or Linux

```bash
bash scripts/run_for_4B_in_Linux_or_WSL.sh
```

## Run With 7B LLM (Qwen-7B-QAnything) On Windows WSL or Linux

```bash
bash scripts/run_for_7B_in_Linux_or_WSL.sh
```

## Run With Openai API On Windows WSL or Linux，CPU Only
```bash
# fill in the API key in scripts/run_for_openai_api_with_cpu_in_Linux_or_WSL.sh
vim scripts/run_for_openai_api_with_cpu_in_Linux_or_WSL.sh
bash scripts/run_for_openai_api_with_cpu_in_Linux_or_WSL.sh
```

## Run With Openai API On Windows WSL or Linux，GPU Only
```bash
# fill in the API key in scripts/run_for_openai_api_with_gpu_in_Linux_or_WSL.sh
vim scripts/run_for_openai_api_with_gpu_in_Linux_or_WSL.sh
bash scripts/run_for_openai_api_with_gpu_in_Linux_or_WSL.sh
```

## Run With 4B LLM (Qwen1.5-4B-Chat-GGUF) On M1 Mac
```bash
bash scripts/run_for_4B_in_M1_mac.sh
```

## Run With 7B LLM (Qwen1.5-7B-Chat-GGUF) On M1 Mac
```bash
bash scripts/run_for_7B_in_M1_mac.sh
```

## Run With Openai API On M1 mac，MPS Only
```bash
# fill in the API key in scripts/run_for_openai_api_in_M1_mac.sh
vim scripts/run_for_openai_api_in_M1_mac.sh
bash scripts/run_for_openai_api_in_M1_mac.sh
```

## USE WITH WEB UI
在浏览器中打开http://0.0.0.0:8777/qanything/，即可使用UI界面
注意末尾的斜杠不可省略，否则会出现404错误

Open http://0.0.0.0:8777/qanything/ in the browser to use the UI interface
Note that the trailing slash cannot be omitted, otherwise a 404 error will occur

## API Documents
API Documentation is available at [API.md](./docs/API.md).
### Test Example
```python
python scripts/new_knowledge_base.py  # print kb_id
python scripts/upload_files.py <kb_id> scripts/weixiaobao.jpg  # print file_id
python scripts/list_files.py <kb_id>  # print files status
python scripts/stream_file.py <kb_id> # print llm res
```
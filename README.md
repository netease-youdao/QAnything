# Pure Python Environment Installation Guide
<p >
  <a href="./README.md">English</a> |
  <a href="./README_zh.md">简体中文</a>
</p>

<h1><span style="color:red;">Important things should be said three times.</span></h1>

# [2024-05-17:Latest Installation and Usage Documentation](https://github.com/netease-youdao/QAnything/blob/master/QAnything%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md) 
# [2024-05-17:Latest Installation and Usage Documentation](https://github.com/netease-youdao/QAnything/blob/master/QAnything%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md) 
# [2024-05-17:Latest Installation and Usage Documentation](https://github.com/netease-youdao/QAnything/blob/master/QAnything%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md)

## Business contact information：
### 010-82558901
![](docs/images/business.jpeg)


## Installation

Requirements:

  - Python 3.10+ (suggest using conda to manage python environment)
  - System 
      - Linux: glibc 2.28+ and Cuda 12.0+ (if using GPU)
      - Windows: WSL with Ubuntu 20.04+ and GEFORCE EXPERIENCE 535.104+ (if using GPU) 
      - MacOS: M1/M2/M3 Mac with Xcode 15.0+

<span style="color:red;">Please create a clean Python virtual environment to avoid potential conflicts(Recommend Anaconda3).</span>

To install the package, run: 
```bash
conda create -n qanything-python python=3.10
conda activate qanything-python
git clone -b qanything-python https://github.com/netease-youdao/QAnything.git
cd QAnything
pip install -e .
```

## Run With 3B LLM (MiniChat-2-3B) On Windows WSL or Linux (Require GPU with >=10GB Memory)

```bash
bash scripts/run_for_3B_in_Linux_or_WSL.sh
```

## Run With 7B LLM (Qwen-7B-QAnything) On Windows WSL or Linux (Require GPU with >=24GB Memory)

```bash
bash scripts/run_for_7B_in_Linux_or_WSL.sh
```

## Run With Openai API On Windows WSL or Linux，CPU Only

<span style="color:red;">Fill in the API key in scripts/run_for_openai_api_with_cpu_in_Linux_or_WSL.sh</span>

```bash
bash scripts/run_for_openai_api_with_cpu_in_Linux_or_WSL.sh
```

## Run With Openai API On Windows WSL or Linux，GPU Only

<span style="color:red;">Fill in the API key in scripts/run_for_openai_api_with_gpu_in_Linux_or_WSL.sh</span>

```bash
bash scripts/run_for_openai_api_with_gpu_in_Linux_or_WSL.sh
```

## Run With Openai API On M1 Mac

<span style="color:red;">Fill in the API key in scripts/run_for_openai_api_in_M1_mac.sh</span>

```bash
bash scripts/run_for_openai_api_in_M1_mac.sh
```

## Run With ollama API On M1 Mac

```bash
bash scripts/run_for_ollama_api_in_M1_mac.sh
```

## Run With 3B LLM (MiniChat-2-3B-INT8-GGUF) On M1 Mac
```bash
bash scripts/run_for_3B_in_M1_mac.sh
```

## Run in OpenCloud
OpenCloud Need Run in Docker Container, Please Install Docker First:
Docker version >= 20.10.5 and docker-compose version >= 2.23.3
```bash
docker-compose up -d
docker attach qanything-container
# Choose one of the 4 commands below to run:
bash scripts/run_for_3B_in_Linux_or_WSL.sh
bash scripts/run_for_7B_in_Linux_or_WSL.sh
bash scripts/run_for_openai_api_with_cpu_in_Linux_or_WSL.sh
bash scripts/run_for_openai_api_with_gpu_in_Linux_or_WSL.sh
```


## USE WITH WEB UI
Open http://0.0.0.0:8777/qanything/ in the browser to use the UI interface,

or open http://{your host ip}:8777/qanything/ in the browser to use the UI interface

*Note that the trailing slash cannot be omitted, otherwise a 404 error will occur*

## API Documents
API Documentation is available at [API.md](./docs/API.md).
### Test Example
```python
python scripts/new_knowledge_base.py  # print kb_id
python scripts/upload_files.py <kb_id> scripts/weixiaobao.jpg  # print file_id
python scripts/list_files.py <kb_id>  # print files status
python scripts/stream_file.py <kb_id> # print llm res
```

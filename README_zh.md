# 纯Python环境安装教程
<p >
  <a href="./README.md">English</a> |
  <a href="./README_zh.md">简体中文</a>
</p>


<h1><span style="color:red;">重要的事情说三遍！</span></h1>

# [2024-05-17:最新的安装和使用文档](https://github.com/netease-youdao/QAnything/blob/master/QAnything%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md) 
# [2024-05-17:最新的安装和使用文档](https://github.com/netease-youdao/QAnything/blob/master/QAnything%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md) 
# [2024-05-17:最新的安装和使用文档](https://github.com/netease-youdao/QAnything/blob/master/QAnything%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md)

## 商务问题联系方式：
### 010-82558901
![](docs/images/business.jpeg)


## 安装 

要求:

  - Python 3.10+ (建议使用aoaconda3来管理Python环境)
  - System 
      - Linux: glibc 2.28+ and Cuda 12.0+ (如果使用GPU)
      - Windows: WSL with Ubuntu 20.04+ and GEFORCE EXPERIENCE 535.104+ (如果使用GPU)
      - MacOS: M1/M2/M3 Mac with Xcode 15.0+

<span style="color:red;">请创建一个干净的Python虚拟环境，以避免潜在冲突（推荐使用Anaconda3）。</span>

安装软件包，请运行: 
```bash
conda create -n qanything-python python=3.10
conda activate qanything-python
git clone -b qanything-python https://github.com/netease-youdao/QAnything.git
cd QAnything
pip install -e .
```

## PDF解析++

如果你想使用更强大的pdf解析功能，请在modelscope下载相关的[解析模型](https://www.modelscope.cn/models/netease-youdao/QAnything-pdf-parser/files),并将其放置到根目录的qanything_kernel/utils/loader/pdf_to_markdown/checkpoints/下

## 在Windows WSL或Linux环境下运行3B大模型（MiniChat-2-3B）要求显存>=10GB
```bash
bash scripts/run_for_3B_in_Linux_or_WSL.sh
```

## 在Windows WSL或Linux环境下运行7B大模型（自研Qwen-7B-QAnything）要求显存>=24GB
```bash
bash scripts/run_for_7B_in_Linux_or_WSL.sh
```

## 在Windows WSL或Linux环境下运行Openai API，仅使用CPU

<span style="color:red;">在scripts/run_for_openai_api_with_cpu_in_Linux_or_WSL.sh中补充api-key等参数</span>

```bash
bash scripts/run_for_openai_api_with_cpu_in_Linux_or_WSL.sh
```

## 在Windows WSL或Linux环境下运行Openai API，使用GPU

<span style="color:red;">在scripts/run_for_openai_api_with_gpu_in_Linux_or_WSL.sh中补充api-key等参数</span>

```bash
bash scripts/run_for_openai_api_with_gpu_in_Linux_or_WSL.sh
```

## 在M1Mac环境下使用Openai API 

<span style="color:red;">在scripts/run_for_openai_api_in_M1_mac.sh中补充api-key等参数</span>

```bash
bash scripts/run_for_openai_api_in_M1_mac.sh
```

## 在M1Mac环境下使用Ollama API 

```bash
bash scripts/run_for_ollama_api_in_M1_mac.sh
```

## 在M1Mac环境下使用3B LLM（(MiniChat-2-3B-INT8-GGUF）

```bash
bash scripts/run_for_3B_in_M1_mac.sh
```

## 在OpenCloud操作系统中运行 
OpenCloud 需要在 Docker 容器中运行，请先安装 Docker：
Docker 版本 >= 20.10.5 且 docker-compose 版本 >= 2.23.3
```bash
docker-compose up -d
docker attach qanything-container
# 选择以下4个命令之一来运行：
bash scripts/run_for_3B_in_Linux_or_WSL.sh
bash scripts/run_for_7B_in_Linux_or_WSL.sh
bash scripts/run_for_openai_api_with_cpu_in_Linux_or_WSL.sh
bash scripts/run_for_openai_api_with_gpu_in_Linux_or_WSL.sh
```

## 访问前端页面 
在浏览器中打开http://0.0.0.0:8777/qanything/

或者打开http://{主机ip地址}:8777/qanything/

即可使用UI界面

注意末尾的斜杠不可省略，否则会出现404错误

## API 文档 
[API.md](./docs/API.md)

### API访问示例 
```python
python scripts/new_knowledge_base.py  # print kb_id
python scripts/upload_files.py <kb_id> scripts/weixiaobao.jpg  # print file_id
python scripts/list_files.py <kb_id>  # print files status
python scripts/stream_file.py <kb_id> # print llm res
```

# 纯Python环境安装教程
<p >
  <a href="./PurePythonEnvironmentInstallationGuide.md">English</a> |
  <a href="./纯Python环境安装教程.md">简体中文</a>
</p>

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
git clone -b qanything-python-v1.3.1 https://github.com/netease-youdao/QAnything.git
cd QAnything
pip install -e .
```

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

## 在M1Mac环境下使用4BLLM（Qwen1.5-4B-Chat-GGUF）
<span style="color:red;">Mac上建议使用Openai API, Qwen 4B/7B模型效果不佳</span>

```bash
bash scripts/run_for_4B_in_M1_mac.sh
```

## 在M1Mac环境下使用7BLLM（Qwen1.5-7B-Chat-GGUF）
<span style="color:red;">Mac上建议使用Openai API, Qwen 4B/7B模型效果不佳</span>

```bash
bash scripts/run_for_7B_in_M1_mac.sh
```

## 访问前端页面 
在浏览器中打开http://127.0.0.1:8777/qanything/

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

<div align="center">

  <a href="https://github.com/netease-youdao/QAnything">
    <!-- Please provide path to your logo here -->
    <img src="docs/images/qanything_logo.png" alt="Logo" width="911" height="175">
  </a>

# **Q**uestion and **A**nswer based on **Anything**

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_zh.md">简体中文</a>
</p>

</div>

<div align="center">

<a href="https://qanything.ai"><img src="https://img.shields.io/badge/%E5%9C%A8%E7%BA%BF%E4%BD%93%E9%AA%8C-QAnything-purple"></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://read.youdao.com#/home"><img src="https://img.shields.io/badge/%E5%9C%A8%E7%BA%BF%E4%BD%93%E9%AA%8C-有道速读-purple"></a>
&nbsp;&nbsp;&nbsp;&nbsp;

<a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://github.com/netease-youdao/QAnything/pulls"><img src="https://img.shields.io/badge/PRs-welcome-red"></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://twitter.com/YDopensource"><img src="https://img.shields.io/badge/follow-%40YDOpenSource-1DA1F2?logo=twitter&style={style}"></a>
&nbsp;&nbsp;&nbsp;&nbsp;

</div>

<details open="open">
<summary>目 录</summary>

- [什么是QAnything](#什么是QAnything)
  - [特点](#特点)
  - [架构](#架构)
- [开始之前](#开始之前)
- [最近更新](#-最近更新-)
- [开始](#开始)
  - [必要条件](#必要条件)
  - [下载安装](#下载安装)
- [常见问题](#常见问题)
- [使用](#使用)
- [微信群](#微信群)
- [支持](#支持)
- [协议](#协议)
- [Acknowledgements](#Acknowledgements)

</details>


## 什么是QAnything？
**QAnything** (**Q**uestion and **A**nswer based on **Anything**) 是致力于支持任意格式文件或数据库的本地知识库问答系统，可断网安装使用。

您的任何格式的本地文件都可以往里扔，即可获得准确、快速、靠谱的问答体验。

目前已支持格式: **PDF(pdf)**，**Word(docx)**，**PPT(pptx)**，**XLS(xlsx)**，**Markdown(md)**，**电子邮件(eml)**，**TXT(txt)**，**图片(jpg，jpeg，png)**，**CSV(csv)**，**网页链接(html)**，更多格式，敬请期待...

### 特点
- 数据安全，支持全程拔网线安装使用。
- 支持跨语种问答，中英文问答随意切换，无所谓文件是什么语种。
- 支持海量数据问答，两阶段向量排序，解决了大规模数据检索退化的问题，数据越多，效果越好。
- 高性能生产级系统，可直接部署企业应用。
- 易用性，无需繁琐的配置，一键安装部署，拿来就用。
- 支持选择多知识库问答。

### 架构
<div align="center">
<img src="docs/images/qanything_arch.png" width = "700" alt="qanything_system" align=center />
</div>

#### 为什么是两阶段检索?
知识库数据量大的场景下两阶段优势非常明显，如果只用一阶段embedding检索，随着数据量增大会出现检索退化的问题，如下图中绿线所示，二阶段rerank重排后能实现准确率稳定增长，即**数据越多，效果越好**。

<div align="center">
<img src="docs/images/two_stage_retrieval.jpg" width = "500" alt="two stage retrievaal" align=center />
</div>

QAnything使用的检索组件[BCEmbedding](https://github.com/netease-youdao/BCEmbedding)有非常强悍的双语和跨语种能力，能消除语义检索里面的中英语言之间的差异，从而实现：
- **强大的双语和跨语种语义表征能力【<a href="https://github.com/netease-youdao/BCEmbedding/tree/master?tab=readme-ov-file#semantic-representation-evaluations-in-mteb" target="_Self">基于MTEB的语义表征评测指标</a>】。**
- **基于LlamaIndex的RAG评测，表现SOTA【<a href="https://github.com/netease-youdao/BCEmbedding/tree/master?tab=readme-ov-file#rag-evaluations-in-llamaindex" target="_Self">基于LlamaIndex的RAG评测指标</a>】。**


#### 一阶段检索（embedding）
| 模型名称 | Retrieval | STS | PairClassification | Classification | Reranking | Clustering | 平均 |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|  
| bge-base-en-v1.5 | 37.14 | 55.06 | 75.45 | 59.73 | 43.05 | 37.74 | 47.20 |  
| bge-base-zh-v1.5 | 47.60 | 63.72 | 77.40 | 63.38 | 54.85 | 32.56 | 53.60 |  
| bge-large-en-v1.5 | 37.15 | 54.09 | 75.00 | 59.24 | 42.68 | 37.32 | 46.82 |  
| bge-large-zh-v1.5 | 47.54 | 64.73 | **79.14** | 64.19 | 55.88 | 33.26 | 54.21 |  
| jina-embeddings-v2-base-en | 31.58 | 54.28 | 74.84 | 58.42 | 41.16 | 34.67 | 44.29 |  
| m3e-base | 46.29 | 63.93 | 71.84 | 64.08 | 52.38 | 37.84 | 53.54 |  
| m3e-large | 34.85 | 59.74 | 67.69 | 60.07 | 48.99 | 31.62 | 46.78 |  
| ***bce-embedding-base_v1*** | **57.60** | **65.73** | 74.96 | **69.00** | **57.29** | **38.95** | ***59.43*** |  

- 更详细的评测结果详见[Embedding模型指标汇总](https://github.com/netease-youdao/BCEmbedding/blob/master/Docs/EvaluationSummary/embedding_eval_summary.md)。

#### 二阶段检索（rerank）
| 模型名称 | Reranking | 平均 |  
|:-------------------------------|:--------:|:--------:|  
| bge-reranker-base | 57.78 | 57.78 |  
| bge-reranker-large | 59.69 | 59.69 |  
| ***bce-reranker-base_v1*** | **60.06** | ***60.06*** |  

- 更详细的评测结果详见[Reranker模型指标汇总](https://github.com/netease-youdao/BCEmbedding/blob/master/Docs/EvaluationSummary/reranker_eval_summary.md)

#### 基于LlamaIndex的RAG评测（embedding and rerank）

<img src="https://github.com/netease-youdao/BCEmbedding/blob/master/Docs/assets/rag_eval_multiple_domains_summary.jpg">

***NOTE:***

- 在WithoutReranker列中，我们的bce-embedding-base_v1模型优于所有其他embedding模型。
- 在固定embedding模型的情况下，我们的bce-reranker-base_v1模型达到了最佳表现。
- **bce-embedding-base_v1和bce-reranker-base_v1的组合是SOTA。**
- 如果想单独使用embedding和rerank请参阅：[BCEmbedding](https://github.com/netease-youdao/BCEmbedding)

#### LLM

开源版本QAnything的大模型基于通义千问，并在大量专业问答数据集上进行微调；在千问的基础上大大加强了问答的能力。
如果需要商用请遵循千问的license，具体请参阅：[通义千问](https://github.com/QwenLM/Qwen)

## 开始之前
**在GitHub上加星，即可立即收到新版本的通知！**
![star_us](https://github.com/netease-youdao/QAnything/assets/29041332/fd5e5926-b9b2-4675-9f60-6cdcaca18e14)
* [🏄 在线试用QAnything](https://qanything.ai)
* [📚 在线试用有道速读](https://read.youdao.com)
* [🛠️ 想只使用BCEmbedding(embedding & rerank)](https://github.com/netease-youdao/BCEmbedding)
* [📖 常见问题](FAQ_zh.md)

## 🚀 最近更新 

- ***2024-01-29***: **支持自定义大模型，包括OpenAI API和其他开源大模型，GPU需求最低降至GTX 1050Ti，极大提升部署，调试等方面的用户体验** - 详见👉 [v1.2.0](https://github.com/netease-youdao/QAnything/releases/tag/v1.2.0)
- ***2024-01-23***: **默认开启rerank，修复在windows上启动时存在的各类问题** - 详见👉 [v1.1.1](https://github.com/netease-youdao/QAnything/releases/tag/v1.1.1)
- ***2024-01-18***: **支持一键启动，支持windows部署，提升pdf，xlsx，html解析效果** - 详见👉 [v1.1.0](https://github.com/netease-youdao/QAnything/releases/tag/v1.1.0)

## 开始

### 必要条件
#### **For Linux**
|**System**| **Required item** | **Minimum Requirement** | **Note**                                                           |
|---------------------------|-------------------|-------------------------|--------------------------------------------------------------------|
|Linux | NVIDIA GPU Memory | >= 4GB (use OpenAI API)  | 最低: GTX 1050Ti（use OpenAI API） <br> 推荐: RTX 3090                   |
|      | NVIDIA Driver Version | >= 525.105.17           |                                                                    |
|      |  Docker version    | >= 20.10.5              | [Docker install](https://docs.docker.com/engine/install/)          |
|      | docker compose  version | >= 2.23.3               | [docker compose install](https://docs.docker.com/compose/install/) |
|      | git-lfs   |                         | [git-lfs install](https://git-lfs.com/)                            |

#### **For Windows with WSL Ubuntu子系统**
| **System**                 | **Required item**        | **Minimum Requirement**   | **Note**                                                                                                                  |
|----------------------------|--------------------------|---------------------------|---------------------------------------------------------------------------------------------------------------------------|
| Windows with WSL Ubuntu子系统 | NVIDIA GPU Memory | >= 4GB (use OpenAI API)                | 最低: GTX 1050Ti（use OpenAI API） <br> 推荐: RTX 3090                                                                          |                                                                |
|                            | GEFORCE EXPERIENCE    | >= 546.33 | [GEFORCE EXPERIENCE download](https://us.download.nvidia.com/GFE/GFEClient/3.27.0.120/GeForce_Experience_v3.27.0.120.exe) |                                                                                               |
|                            |  Docker Desktop           | >=  4.26.1（131620）     | [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)                                    |
|                            | git-lfs   |                  | [git-lfs install](https://git-lfs.com/)                                                                                   |

### 下载安装
### step1: 下载本项目
```shell
git clone https://github.com/netease-youdao/QAnything.git
```
### step2: 进入项目根目录执行启动脚本
* [📖 QAnything_Startup_Usage](docs/QAnything_Startup_Usage_README.md)
* 执行 ```bash ./run.sh -h``` 获取详细的LLM服务配置方法 
  
```shell
cd QAnything
bash run.sh  # 默认在0号GPU上启动
```

<details>
<summary>（注意）如果自动下载失败，您可以从以下三个地址之一手动下载模型。</summary>

modelscope: https://modelscope.cn/models/netease-youdao/QAnything

wisemodel: https://wisemodel.cn/models/Netease_Youdao/qanything

huggingfase: https://huggingface.co/netease-youdao/QAnything

</details>

<details>
<summary>（可选）指定单GPU启动</summary>

```shell
cd QAnything
bash ./run.sh -c local -i 0 -b default # 指定0号GPU启动 GPU编号从0开始 windows机器一般只有一张卡，所以只能指定0号GPU
```
</details>

<details>
<summary>（可选）指定单GPU启动 - 推荐 Windows10/Windows11 WSL2 用户使用此方式运行 QAnything</summary>

```shell
# 注意: Windows系统请先进入**WSL2**环境
# Step 1. 下载开源 LLM 模型 (e.g., Qwen-7B-QAnything) 并保存在路径 "/path/to/QAnything/assets/custom_models"
# (可选) 从 ModelScope 下载 Qwen-7B-QAnything: https://www.modelscope.cn/models/netease-youdao/Qwen-7B-QAnything
# (可选) 从 Huggingface 下载 Qwen-7B-QAnything: https://huggingface.co/netease-youdao/Qwen-7B-QAnything
cd QAnything/assets/custom_models
git clone https://huggingface.co/netease-youdao/Qwen-7B-QAnything

# Step 2. 执行启动命令，其中"-b hf"表示指定使用 Huggingface transformers 后端运行 LLM.
cd ../../
bash ./run.sh -c local -i 0 -b hf -m Qwen-7B-QAnything -t qwen-7b-qanything
```
</details>

<details>
<summary>（可选）指定单GPU启动 - 推荐 GPU Compute Capability >= 8.6 && VRAM >= 24GB 使用此方式运行 QAnything</summary>

```shell
# 查看 GPU 算力 GPU Compute Capability: https://developer.nvidia.com/cuda-gpus
# Step 1. 下载开源 LLM 模型 (e.g., Qwen-7B-QAnything) 并保存在路径 "/path/to/QAnything/assets/custom_models"
# (可选) 从 ModelScope 下载 Qwen-7B-QAnything: https://www.modelscope.cn/models/netease-youdao/Qwen-7B-QAnything
# (可选) 从 Huggingface 下载 Qwen-7B-QAnything: https://huggingface.co/netease-youdao/Qwen-7B-QAnything
cd QAnything/assets/custom_models
git clone https://huggingface.co/netease-youdao/Qwen-7B-QAnything

# Step 2. 执行启动命令，其中"-b vllm"表示指定使用 vllm 后端运行 LLM.
cd ../../
bash ./run.sh -c local -i 0 -b vllm -m Qwen-7B-QAnything -t qwen-7b-qanything -p 1 -r 0.85
```
</details>

<details>
<summary>（可选）指定多GPU启动</summary>

```shell
cd QAnything
bash ./run.sh -c local -i 0,1 -b default  # 指定0,1号GPU启动，请确认有多张GPU可用，最多支持两张卡启动
```
</details>

### step3: 开始体验

#### 前端页面
运行成功后，即可在浏览器输入以下地址进行体验。

- 前端地址: http://`your_host`:5052/qanything/

#### API
如果想要访问API接口，请参考下面的地址:
- API address: http://`your_host`:8777/api/
- For detailed API documentation, please refer to [QAnything API 文档](docs/API.md)

#### DEBUG
如果想要查看相关日志，请查看`QAnything/logs/debug_logs`目录下的日志文件。
- **debug.log**
  - 用户请求处理日志
- **sanic_api.log**
  - 后端服务运行日志
- **llm_embed_rerank_tritonserver.log**（单卡部署）
  - LLM embedding和rerank tritonserver服务启动日志
- **llm_tritonserver.log**（多卡部署）
  - LLM tritonserver服务启动日志
- **embed_rerank_tritonserver.log**（多卡部署或使用openai接口）
  - embedding和rerank tritonserver服务启动日志
- rerank_server.log
  - rerank服务运行日志
- ocr_server.log
  - OCR服务运行日志
- npm_server.log
  - 前端服务运行日志
- llm_server_entrypoint.log
  - LLM中转服务运行日志
- fastchat_logs/*.log
  - FastChat服务运行日志

### 关闭服务
```shell
bash close.sh
```

## 断网安装
### windows断网安装
如果您想要断网安装QAnything，您可以使用以下命令启动服务。
```shell 
# 先在联网机器上下载docker镜像
docker pull quay.io/coreos/etcd:v3.5.5
docker pull minio/minio:RELEASE.2023-03-20T20-16-18Z
docker pull milvusdb/milvus:v2.3.4
docker pull mysql:latest
docker pull freeren/qanything-win:v1.2.1

# 打包镜像
docker save quay.io/coreos/etcd:v3.5.5 minio/minio:RELEASE.2023-03-20T20-16-18Z milvusdb/milvus:v2.3.4 mysql:latest freeren/qanything-win:v1.2.1 -o qanything_offline.tar

# 下载QAnything代码
wget https://github.com/netease-youdao/QAnything/archive/refs/heads/master.zip

# 把镜像qanything_offline.tar和代码QAnything-master.zip拷贝到断网机器上
cp QAnything-master.zip qanything_offline.tar /path/to/your/offline/machine

# 在断网机器上加载镜像
docker load -i qanything_offline.tar

# 解压代码，运行
unzip QAnything-master.zip
cd QAnything-master
bash run.sh
```

### Linux断网安装
如果您想要断网安装QAnything，您可以使用以下命令启动服务。
```shell 
# 先在联网机器上下载docker镜像
docker pull quay.io/coreos/etcd:v3.5.5
docker pull minio/minio:RELEASE.2023-03-20T20-16-18Z
docker pull milvusdb/milvus:v2.3.4
docker pull mysql:latest
docker pull freeren/qanything:v1.2.1

# 打包镜像
docker save quay.io/coreos/etcd:v3.5.5 minio/minio:RELEASE.2023-03-20T20-16-18Z milvusdb/milvus:v2.3.4 mysql:latest freeren/qanything:v1.2.1 -o qanything_offline.tar

# 下载QAnything代码
wget https://github.com/netease-youdao/QAnything/archive/refs/heads/master.zip

# 把镜像qanything_offline.tar和代码QAnything-master.zip拷贝到断网机器上
cp QAnything-master.zip qanything_offline.tar /path/to/your/offline/machine

# 在断网机器上加载镜像
docker load -i qanything_offline.tar

# 解压代码，运行
unzip QAnything-master.zip
cd QAnything-master
bash run.sh
```

## 常见问题
[常见问题](FAQ_zh.md)


## 使用
### 跨语种：多篇英文论文问答
[![](docs/videos/multi_paper_qa.mp4)](https://github.com/netease-youdao/QAnything/assets/141105427/8915277f-c136-42b8-9332-78f64bf5df22)
### 信息抽取
[![](docs/videos/information_extraction.mp4)](https://github.com/netease-youdao/QAnything/assets/141105427/b9e3be94-183b-4143-ac49-12fa005a8a9a)
### 文件大杂烩
[![](docs/videos/various_files_qa.mp4)](https://github.com/netease-youdao/QAnything/assets/141105427/7ede63c1-4c7f-4557-bd2c-7c51a44c8e0b)
### 网页问答
[![](docs/videos/web_qa.mp4)](https://github.com/netease-youdao/QAnything/assets/141105427/d30942f7-6dbd-4013-a4b6-82f7c2a5fbee)

### 接入API
如果需要接入API，请参阅[QAnything API 文档](docs/API.md)

## 贡献代码
我们感谢您对贡献到我们项目的兴趣。无论您是修复错误、改进现有功能还是添加全新内容，我们都欢迎您的贡献！
### 感谢以下所有贡献者
<a href="https://github.com/netease-youdao/QAnything/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=netease-youdao/QAnything" />
</a>


## 支持

### Discord <a href="https://discord.gg/5uNpPsEJz8"><img src="https://img.shields.io/discord/1197874288963895436?style=social&logo=discord"></a>
欢迎加入QAnything [Discord](https://discord.gg/5uNpPsEJz8) 社区！



### 微信群

扫描下方二维码，加入QAnything官方微信群

<img src="docs/images/Wechat.jpg" width="30%" height="auto">

### 邮箱
如果你需要私信我们团队，请通过下面的邮箱联系我们：

qanything@rd.netease.com

### GitHub issues
有任何公开的问题，欢迎提交
[Github issues](https://github.com/netease-youdao/QAnything/issues)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=netease-youdao/QAnything,netease-youdao/BCEmbedding&type=Date)](https://star-history.com/#netease-youdao/QAnything&netease-youdao/BCEmbedding&Date)

## 协议

`QAnything` 依照 [Apache 2.0 协议](./LICENSE)开源。

## Acknowledgements
- [BCEmbedding](https://github.com/netease-youdao/BCEmbedding)
- [Qwen](https://github.com/QwenLM/Qwen)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [vllm](https://github.com/vllm-project/vllm)
- [FastChat](https://github.com/lm-sys/FastChat)
- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [Langchain](https://github.com/langchain-ai/langchain)
- [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)
- [Milvus](https://github.com/milvus-io/milvus)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 
- [Sanic](https://github.com/sanic-org/sanic)

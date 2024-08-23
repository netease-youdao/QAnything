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

- [什么是QAnything](#什么是qanything)
  - [特点](#特点)
  - [架构](#架构)
- [最近更新](#-最近更新)
- [开始之前](#开始之前)
- [开始](#开始)
  - [安装方式](#安装方式)
  - [纯Python环境安装](#纯python环境安装)
  - [docker环境安装](#docker环境安装)
  - [断网安装](#断网安装)
- [常见问题](#常见问题)
- [使用](#使用)
- [路线图&反馈](#%EF%B8%8F-路线图--反馈)
- [交流&支持](#交流--支持)
- [协议](#协议)
- [Acknowledgements](#acknowledgements)

</details>

# 🚀 重要更新 
<h1><span style="color:red;">重要的事情说三遍！</span></h1>

# [2024-05-17:最新的安装和使用文档](https://github.com/netease-youdao/QAnything/blob/master/QAnything%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md) 
# [2024-05-17:最新的安装和使用文档](https://github.com/netease-youdao/QAnything/blob/master/QAnything%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md) 
# [2024-05-17:最新的安装和使用文档](https://github.com/netease-youdao/QAnything/blob/master/QAnything%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md)

## 商务问题联系方式：
### 010-82558901
![](docs/images/business.jpeg)

# 什么是QAnything？
**QAnything** (**Q**uestion and **A**nswer based on **Anything**) 是致力于支持任意格式文件或数据库的本地知识库问答系统，可断网安装使用。

您的任何格式的本地文件都可以往里扔，即可获得准确、快速、靠谱的问答体验。

目前已支持格式: **PDF(pdf)**，**Word(docx)**，**PPT(pptx)**，**XLS(xlsx)**，**Markdown(md)**，**电子邮件(eml)**，**TXT(txt)**，**图片(jpg，jpeg，png)**，**CSV(csv)**，**网页链接(html)**，更多格式，敬请期待...

## 特点
- 数据安全，支持全程拔网线安装使用。
- 支持跨语种问答，中英文问答随意切换，无所谓文件是什么语种。
- 支持海量数据问答，两阶段向量排序，解决了大规模数据检索退化的问题，数据越多，效果越好。
- 高性能生产级系统，可直接部署企业应用。
- 易用性，无需繁琐的配置，一键安装部署，拿来就用。
- 支持选择多知识库问答。

## 架构
<div align="center">
<img src="docs/images/qanything_arch.png" width = "700" alt="qanything_system" align=center />
</div>

### 为什么是两阶段检索?
知识库数据量大的场景下两阶段优势非常明显，如果只用一阶段embedding检索，随着数据量增大会出现检索退化的问题，如下图中绿线所示，二阶段rerank重排后能实现准确率稳定增长，即**数据越多，效果越好**。

<div align="center">
<img src="docs/images/two_stage_retrieval.jpg" width = "500" alt="two stage retrievaal" align=center />
</div>

QAnything使用的检索组件[BCEmbedding](https://github.com/netease-youdao/BCEmbedding)有非常强悍的双语和跨语种能力，能消除语义检索里面的中英语言之间的差异，从而实现：
- **强大的双语和跨语种语义表征能力【<a href="https://github.com/netease-youdao/BCEmbedding/tree/master?tab=readme-ov-file#semantic-representation-evaluations-in-mteb" target="_Self">基于MTEB的语义表征评测指标</a>】。**
- **基于LlamaIndex的RAG评测，表现SOTA【<a href="https://github.com/netease-youdao/BCEmbedding/tree/master?tab=readme-ov-file#rag-evaluations-in-llamaindex" target="_Self">基于LlamaIndex的RAG评测指标</a>】。**


### 一阶段检索（embedding）
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

### 二阶段检索（rerank）
| 模型名称 | Reranking | 平均 |  
|:-------------------------------|:--------:|:--------:|  
| bge-reranker-base | 57.78 | 57.78 |  
| bge-reranker-large | 59.69 | 59.69 |  
| ***bce-reranker-base_v1*** | **60.06** | ***60.06*** |  

- 更详细的评测结果详见[Reranker模型指标汇总](https://github.com/netease-youdao/BCEmbedding/blob/master/Docs/EvaluationSummary/reranker_eval_summary.md)

### 基于LlamaIndex的RAG评测（embedding and rerank）

<img src="https://github.com/netease-youdao/BCEmbedding/blob/master/Docs/assets/rag_eval_multiple_domains_summary.jpg">

***NOTE:***

- 在WithoutReranker列中，我们的bce-embedding-base_v1模型优于所有其他embedding模型。
- 在固定embedding模型的情况下，我们的bce-reranker-base_v1模型达到了最佳表现。
- **bce-embedding-base_v1和bce-reranker-base_v1的组合是SOTA。**
- 如果想单独使用embedding和rerank请参阅：[BCEmbedding](https://github.com/netease-youdao/BCEmbedding)

### LLM

开源版本QAnything的大模型基于通义千问，并在大量专业问答数据集上进行微调；在千问的基础上大大加强了问答的能力。
如果需要商用请遵循千问的license，具体请参阅：[通义千问](https://github.com/QwenLM/Qwen)

# 🚀 最近更新 
- ***2024-08-22***: **支持快速开始、前端配置参数、在线预览和编辑chunk块，优化项目架构和启动方式。** - 详见👉 [v2.0.0](https://github.com/netease-youdao/QAnything/releases/tag/v2.0.0)
- ***2024-05-20***: **支持与OpenAI API兼容的其他LLM服务，并提供优化后的PDF解析器。** - 详见👉 [v1.4.1](https://github.com/netease-youdao/QAnything/releases/tag/v1.4.1)
- ***2024-04-26***: **支持联网检索、FAQ、自定义BOT、文件溯源等。** - 详见👉 [v1.4.0](https://github.com/netease-youdao/QAnything/releases/tag/v1.4.0-python)
- ***2024-04-03***: **支持在纯Python环境中安装；支持混合检索。** - 详见👉 [v1.3.0](https://github.com/netease-youdao/QAnything/releases/tag/v1.3.0)
- ***2024-01-29***: **支持自定义大模型，包括OpenAI API和其他开源大模型，GPU需求最低降至GTX 1050Ti，极大提升部署，调试等方面的用户体验** - 详见👉 [v1.2.0](https://github.com/netease-youdao/QAnything/releases/tag/v1.2.0)
- ***2024-01-23***: **默认开启rerank，修复在windows上启动时存在的各类问题** - 详见👉 [v1.1.1](https://github.com/netease-youdao/QAnything/releases/tag/v1.1.1)
- ***2024-01-18***: **支持一键启动，支持windows部署，提升pdf，xlsx，html解析效果** - 详见👉 [v1.1.0](https://github.com/netease-youdao/QAnything/releases/tag/v1.1.0)

# 开始之前
**在GitHub上加星，即可立即收到新版本的通知！**
![star_us](https://github.com/netease-youdao/QAnything/assets/29041332/fd5e5926-b9b2-4675-9f60-6cdcaca18e14)
* [🏄 在线试用QAnything](https://qanything.ai)
* [📚 在线试用有道速读](https://read.youdao.com)
* [🛠️ 想只使用BCEmbedding(embedding & rerank)](https://github.com/netease-youdao/BCEmbedding)
* [📖 常见问题](FAQ_zh.md)
* [👂️需求反馈 | 让我听见你的声音](https://qanything.canny.io/feature-requests)


# 开始
## 安装方式
此次更新带来了使用门槛，资源占用，检索效果，问答效果，解析效果，前端效果，服务架构，使用方式等多方面的改进。同时合并了旧有的docker版和python版两个版本，改为全新的统一版本，采用docker compose单行命令一键启动。

### 最新特性表

| 特性                                | python （v1.4.1） | docker （v1.4.1） | 全新QAnything v2.0 | 说明                                                                       |
| --------------------------------- | --------------- | --------------- | ---------------- | ------------------------------------------------------------------------ |
| 详细安装文档                            | ✅               | ✅               | ✅                |                                                                          |
| API支持                             | ✅               | ✅               | ✅                |                                                                          |
| 生产环境（小型生产环境）                      | ❌               | ✅               | ✅                |                                                                          |
| 断网安装（私有化部署）                       | ❌               | ✅               | ✅                |                                                                          |
| 支持多并发                             | ❌               | ✅               | ✅                |                                                                          |
| 支持多卡推理                            | ❌               | ✅               | ❌                | v2.0版本不再提供默认本地LLM，一律通过openai接口接入，用户可自行通过ollama等工具部署本地LLM                 |
| 支持Mac（M系列芯片）                      | ✅               | ❌               | ✅                |                                                                          |
| 支持Linux                           | ✅               | ✅               | ✅                | python版本Linux下默认使用onnxruntime-gpu for cuda12，glibc<2.28时自动切换为onnxruntime |
| 支持windows （无需WSL）                 | ❌               | ❌               | ✅                | v1.4.1版本的python和docker均要求WSL环境，v2.0可直接在非WSL环境下启动                         |
| 支持纯CPU环境                          | ✅               | ❌               | ✅                | v2.0版本Mac，Linux，Win统一不再使用GPU，完全迁移至CPU                                    |
| 支持混合检索（BM25+embedding）            | ❌               | ✅               | ✅                |                                                                          |
| 支持联网检索（需外网VPN）                    | ✅               | ❌               | ✅                |                                                                          |
| 支持FAQ问答                           | ✅               | ❌               | ✅                |                                                                          |
| 支持自定义机器人（可绑定知识库，可分享）              | ✅               | ❌               | ✅                |                                                                          |
| 支持文件溯源（数据来源可直接点击打开）               | ✅               | ❌               | ✅                |                                                                          |
| 支持问答日志检索（暂只支持通过API调用）             | ✅               | ❌               | ✅                |                                                                          |
| 支持解析语音文件（依赖faster_whisper，解析速度慢）  | ✅               | ❌               | ❌                |                                                                          |
| 支持OpenCloudOS                     | ✅               | ❌               | ✅                |                                                                          |
| 支持与OpenAI接口兼容的其他开源大模型服务(包括ollama) | ✅               | ✅               | ✅                | v1.4.1版本需手动修改api_key，base_url，model等参数，v2.0版本均改为前端设置自动保存                 |
| pdf（包含表格）解析效果+++                  | ✅               | ❌               | ✅                | v1.4.1版本需手动设置，v2.0无需手动设置，pdf解析效果和性能均有提升                                  |
| 用户自定义embed，rerank配置（实验性：提升速度）     | ✅               | ❌               | ✅                | v1.4.1需手动开启，v2.0默认使用最佳配置                                                 |
| 其他文件类型解析效果+++                     | ❌               | ❌               | ✅                | v2.0版本提升url，md，xlsx，docx等解析效果                                            |
| 支持独立服务调用                          | ❌               | ❌               | ✅                | v2.0版本独立依赖服务，包括embed，rerank，ocr，pdf解析服务等，可独立调用（http）                     |
| 支持快速开始模式                          | ❌               | ❌               | ✅                | 快速开始：无需创建知识库，支持文件即传即问，支持无文件问答                                            |
| 支持仅检索模式                           | ❌               | ❌               | ✅                | 仅返回检索结果，不调用大模型进行问答                                                       |
| 支持解析结果chunks内容可视化，手动编辑            | ❌               | ❌               | ✅                | v2.0版本支持手动编辑chunks内容，实时生效                                                |
| pdf解析支持图片,支持回答带图                  | ❌               | ❌               | ✅                |                                                                          |

## 新增细节优化：

* 支持前端配置API_BASE，API_KEY，文本分片大小，输出token数量，上下文消息数量等参数
* 优化Bot角色设定的指令遵循效果，每个Bot可单独配置模型参数
* 支持创建多个对话窗口，同时保存多份历史问答记录
* 支持问答内容保存成图片
* 优化上传文件逻辑，解析文件与问答请求独立，上传文件不再影响问答
* 优化镜像大小，旧版本镜像压缩后大小为18.94GB->新版镜像压缩后大小为4.88GB，降为原有的1/4，提供完整Dockerfile
* 检索优化，chunks新增片段融合与排序，聚合单文档或双文档
* 检索阶段和问答阶段均嵌入metadata信息，提升检索和问答效果
  
  ### 各阶段数据展示：
* 知识库所有文件上传进度展示
* 知识库单个文件上传进度展示，上传各阶段耗时
* 问答信息统计，包含问答各阶段耗时，tokens消耗，模型信息等
* 用户信息统计，包含上传文件总数量，总耗时，问答历史记录等
  
  ### 问题修复
* xlsx表格支持多sheet解析
* 优化PDF表格漏识别问题
* 修复DOCX文件解析出错问题
* 优化FAQ匹配逻辑
* 支持非utf-8编码的txt文件                                          |




## docker环境安装
### 必要条件
|**System**| **Required item** | **Minimum Requirement** | **Note**                                                           |
|---------------------------|-------------------|-------------------------|--------------------------------------------------------------------|
|      | RAM Memory | >= 20GB (use OpenAI API)                             
|      |  Docker version    | >= 20.10.5              | [Docker install](https://docs.docker.com/engine/install/)          |
|  Linux/Mac     | docker compose  version | >= 2.23.3               | [docker compose install](https://docs.docker.com/compose/install/) |
|   Windows   |  Docker Desktop           | >=  4.26.1（131620）     | [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)                                    |
|      | git-lfs   |                         | [git-lfs install](https://git-lfs.com/)                            |




### [docker版本详细安装步骤，请点击此处](https://github.com/netease-youdao/QAnything/blob/master/docs/docker%E7%89%88%E6%9C%AC%E5%AE%89%E8%A3%85%E6%94%BB%E7%95%A5.md)


### step1: 下载本项目
```shell
git clone https://github.com/netease-youdao/QAnything.git
```
### step2: 进入项目根目录执行启动脚本
* [📖 QAnything_Startup_Usage](docs/QAnything_Startup_Usage_README.md)
* 执行 ```docker compose启动命令``` 获取详细的LLM服务配置方法 
  
```shell
cd QAnything
# 在 Linux 上启动
docker compose -f docker-compose-linux.yaml up
# 在 Mac 上启动
docker compose -f docker-compose-mac.yaml up
# 在 Windows 上启动
docker compose -f docker-compose-win.yaml up
```

（注意）如果启动失败，可以尝试将 `docker compose`改为 `docker-compose`。


### step3: 开始体验

#### 前端页面
运行成功后，即可在浏览器输入以下地址进行体验。

- 前端地址: http://`your_host`:8777/qanything/

#### API
如果想要访问API接口，请参考下面的地址:
- API address: http://`your_host`:8777/api/
- For detailed API documentation, please refer to [QAnything API 文档](docs/API.md)

#### DEBUG
如果想要查看相关日志，请查看`QAnything/logs/debug_logs`目录下的日志文件。
- **debug.log**
  - 用户请求处理日志
- **main_server.log**
  - 后端服务运行日志
- **rerank_server.log**
  - rerank服务运行日志
- **ocr_server.log**
  - OCR服务运行日志
- **embedding_server.log**
  - 前端服务运行日志
- **insert_files_server.log**
  - 文件上传服务运行日志
- **pdf_parser_server.log**
  - pdf解析服务运行日志

### 关闭服务
```shell
bash close.sh
```

## 离线使用
如果您想要离线使用QAnything，需要在断网机器提前部署本地的ollama模型，可以使用以下命令启动服务。
### windows离线使用
```shell 
# 先在联网机器上下载docker镜像
docker pull quay.io/coreos/etcd:v3.5.5
docker pull minio/minio:RELEASE.2023-03-20T20-16-18Z
docker pull milvusdb/milvus:v2.4.8
docker pull mysql:8.4
docker pull xixihahaliu01/qanything-win:v1.5.1  # 从 [https://github.com/netease-youdao/QAnything/blob/master/docker-compose-windows.yaml#L103] 中获取最新镜像版本号。

# 打包镜像
docker save quay.io/coreos/etcd:v3.5.5 minio/minio:RELEASE.2023-03-20T20-16-18Z milvusdb/milvus:v2.4.8 mysql:8.4 xixihahaliu01/qanything-win:v1.5.1 -o qanything_offline.tar

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

### Linux离线使用
```shell 
# 先在联网机器上下载docker镜像
docker pull quay.io/coreos/etcd:v3.5.5
docker pull minio/minio:RELEASE.2023-03-20T20-16-18Z
docker pull milvusdb/milvus:v2.4.8
docker pull mysql:8.4
docker pull xixihahaliu01/qanything-linux:v1.5.1  # 从 [https://github.com/netease-youdao/qanything/blob/master/docker-compose-linux.yaml#L104] 中获取最新镜像版本号。

# 打包镜像
docker save quay.io/coreos/etcd:v3.5.5 minio/minio:RELEASE.2023-03-20T20-16-18Z milvusdb/milvus:v2.4.8 mysql:8.4 xixihahaliu01/qanything-linux:v1.5.1 -o qanything_offline.tar

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

# 🛣️ 路线图 & 反馈
🔎 想了解QAnything的未来规划和进展，请看这里： [QAnything Roadmap](https://qanything.canny.io/)

🤬 想要给QAnything提交反馈，请看这里(可以给每个功能需求投票哦): [QAnything Feedbak](https://qanything.canny.io/feature-requests)

# 交流 & 支持

## Discord <a href="https://discord.gg/5uNpPsEJz8"><img src="https://img.shields.io/discord/1197874288963895436?style=social&logo=discord"></a>
欢迎加入QAnything [Discord](https://discord.gg/5uNpPsEJz8) 社区！



## 微信
欢迎关注微信公众号，获取最新QAnything信息

<img src="docs/images/qrcode_for_qanything.jpg" width="30%" height="auto">

欢迎扫码进入QAnything交流群

<img src="docs/images/Wechat_0509.jpg" width="30%" height="auto">

## 邮箱
如果你需要私信我们团队，请通过下面的邮箱联系我们：

qanything@rd.netease.com

## GitHub issues & discussions
有任何公开的问题，欢迎提交issues，或者在discussions区讨论
- [Github issues](https://github.com/netease-youdao/QAnything/issues)
- [Github discussions](https://github.com/netease-youdao/QAnything/discussions)
<a href="https://github.com/netease-youdao/QAnything/discussions">
  <!-- Please provide path to your logo here -->
  <img src="https://github.com/netease-youdao/QAnything/assets/29041332/ad027ec5-0bbc-4ea0-92eb-81b30c5359a1" alt="Logo" width="600">
</a>

# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=netease-youdao/QAnything,netease-youdao/BCEmbedding&type=Date)](https://star-history.com/#netease-youdao/QAnything&netease-youdao/BCEmbedding&Date)

# 协议

`QAnything` 依照 [Apache 2.0 协议](./LICENSE)开源。

# Acknowledgements
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
- [RAGFlow](https://github.com/infiniflow/ragflow)

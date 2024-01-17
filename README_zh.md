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
- [开始](#开始)
  - [必要条件](#必要条件)
  - [下载安装](#下载安装)
- [使用](#使用)
  - [接入API](#接入API)
- [微信群](#微信群)
- [支持](#支持)
- [协议](#协议)
- [Acknowledgements](#Acknowledgements)

</details>


## 什么是QAnything？
**QAnything** (**Q**uestion and **A**nswer based on **Anything**) 是致力于支持任意格式文件或数据库的本地知识库问答系统，可断网安装使用。

您的任何格式的本地文件都可以往里扔，即可获得准确、快速、靠谱的问答体验。

目前已支持格式: **PDF**，**Word(doc/docx)**，**PPT**，**Markdown**，**Eml**，**TXT**，**图片（jpg，png等）**，**网页链接**，更多格式，敬请期待...

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

## 开始
[:point_right: 在线试用QAnything](https://qanything.ai)

### 必要条件
|  **必要项**     | **最低要求**      | **备注** |
| --------------         |---------------| --------------------------------- |
| NVIDIA GPU Memory      | >= 16GB       | 推荐NVIDIA 3090|
| NVIDIA Driver 版本      | >= 525.105.17 |                           |
| CUDA 版本               | >= 12.0       |                           |
| docker compose 版本     | >= 2.12.1     | [docker compose 安装教程](https://docs.docker.com/compose/install/)|

### 下载安装
#### step1: 下载本项目
```
git clone https://github.com/netease-youdao/QAnything.git
```
#### step2: 下载模型并解压到本项目根目录下
本项目提供多种模型下载平台，选择其中一个方式下载即可。

[👉【始智AI】](https://wisemodel.cn/models/Netease_Youdao/qanything)
[👉【魔搭社区】](https://www.modelscope.cn/models/netease-youdao/QAnything)
[👉【HuggingFace】](https://huggingface.co/netease-youdao/QAnything)

<details>
<summary>下载方式1：始智AI（推荐👍）</summary>

```
cd QAnything
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://www.wisemodel.cn/Netease_Youdao/qanything.git
unzip qanything/models.zip   # in root directory of the current project
```
</details>
<details>
<summary>下载方式2：魔搭社区</summary>

```
cd QAnything
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://www.modelscope.cn/netease-youdao/QAnything.git
unzip QAnything/models.zip   # in root directory of the current project
```
</details>
<details>
<summary>下载方式3：HuggingFace</summary>

```
cd QAnything
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/netease-youdao/QAnything
unzip QAnything/models.zip   # in root directory of the current project
```
</details>

#### step3：修改配置
##### 在WINDOWNS系统下：
```
vim docker-compose-windows.yaml # change CUDA_VISIBLE_DEVICES to your gpu device id
vim front_end/.env.production # 设置准确的host，本地环境默认一般是localhost或0.0.0.0
```
##### 在Linux系统下：
```
# 判断当前wsl2是否是
vim docker-compose-linux.yaml # change CUDA_VISIBLE_DEVICES to your gpu device id
vim front_end/.env.production # 设置准确的host，本地环境默认一般是localhost或0.0.0.0
```
#### step4: 启动服务
##### 在Windows系统下
<details>
<summary>新手推荐！</summary>

```shell
# 前台启动，日志实时打印到屏幕上，ctrl+c即可停止
docker-compose -f docker-compose-windows.yaml up qanything_local
```
</details>

<details>
<summary>老手推荐！</summary>

```shell
# 后台启动，ctrl+c不会停止
docker-compose -f docker-compose-windows.yaml up -d
# 执行如下命令查看日志
docker-compose -f docker-compose-windows.yaml logs qanything_local
# 停止服务
docker-compose -f docker-compose-windows.yaml down
```
</details>

##### 在Linux系统下
<details>
<summary>新手推荐！</summary>

```shell
# 前台启动，日志实时打印到屏幕上，ctrl+c即可停止
docker-compose -f docker-compose-linux.yaml up qanything_local
```
</details>

<details>
<summary>老手推荐！</summary>

```shell
# 后台启动，ctrl+c不会停止
docker-compose -f docker-compose-linux.yaml up -d
# 执行如下命令查看日志
docker-compose -f docker-compose-linux.yaml logs qanything_local
# 停止服务
docker-compose -f docker-compose-linux.yaml down
```
</details>


安装成功后，即可在浏览器输入以下地址进行体验。

- 前端地址: http://{your_host}:5052/qanything/

- api地址: http://{your_host}:5052/api/

详细API文档请移步[QAnything API 文档](docs/API.md)

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

## 微信群

欢迎大家扫码加入官方微信交流群。

<img src="docs/images/Wechat.jpg" width="20%" height="auto">

## 支持

有任何问题，请通过以下方式联系我们:

- [Github issues](https://github.com/netease-youdao/QAnything/issues)
- [Netease Youdao](https://github.com/netease-youdao)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=netease-youdao/QAnything,netease-youdao/BCEmbedding&type=Date)](https://star-history.com/#netease-youdao/QAnything&netease-youdao/BCEmbedding&Date)

## 协议

`QAnything` 依照 [Apache 2.0 协议](./LICENSE)开源。

## Acknowledgements
- [BCEmbedding](https://github.com/netease-youdao/BCEmbedding)
- [Qwen](https://github.com/QwenLM/Qwen)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [Langchain](https://github.com/langchain-ai/langchain)
- [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)
- [Milvus](https://github.com/milvus-io/milvus)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 
- [Sanic](https://github.com/sanic-org/sanic)

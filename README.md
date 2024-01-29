<div align="center">

  <a href="https://github.com/netease-youdao/QAnything">
    <!-- Please provide path to your logo here -->
    <img src="docs/images/qanything_logo.png" alt="Logo" width="800">
  </a>

# **Q**uestion and **A**nswer based on **Anything**

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_zh.md">简体中文</a>
</p>

</div>

<div align="center">

<a href="https://qanything.ai"><img src="https://img.shields.io/badge/try%20online-qanything.ai-purple"></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://read.youdao.com#/home"><img src="https://img.shields.io/badge/try%20online-read.youdao.com-purple"></a>
&nbsp;&nbsp;&nbsp;&nbsp;

<a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://github.com/netease-youdao/QAnything/pulls"><img src="https://img.shields.io/badge/PRs-welcome-red"></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://twitter.com/YDopensource"><img src="https://img.shields.io/badge/follow-%40YDOpenSource-1DA1F2?logo=twitter&style={style}"></a>
&nbsp;&nbsp;&nbsp;&nbsp;

<a href="https://discord.gg/5uNpPsEJz8"><img src="https://img.shields.io/discord/1197874288963895436?style=social&logo=discord"></a>
&nbsp;&nbsp;&nbsp;&nbsp;



</div>

<details open="open">
<summary>Table of Contents</summary>

- [What is QAnything](#What-is-QAnything)
  - [Key features](#Key-features)
  - [Architecture](#Architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [FAQ](#FAQ)
- [Usage](#usage)
  - [API Document](#API-Document)
- [WeChat Group](#WeChat-Group)
- [Community & Support](#Community--Support)
- [License](#license)
- [Acknowledgements](#Acknowledgments)

</details>

## What is QAnything?
`QAnything`(**Q**uestion and **A**nswer based on **Anything**) is a local knowledge base question-answering system designed to support a wide range of file formats and databases, allowing for offline installation and use.

With `QAnything`, you can simply drop any locally stored file of any format and receive accurate, fast, and reliable answers.

Currently supported formats include: **PDF, Word (doc/docx), PPT, Markdown, Eml, TXT, Images (jpg, png, etc.), Web links** and more formats coming soon…


### Key features

- **Data Security**, supports installation and usage with network cable unplugged throughout the process.
- **Cross-language QA support**, freely switch between Chinese and English QA, regardless of the language of the document.
- **Supports massive data QA**, two-stage retrieval ranking, solving the degradation problem of large-scale data retrieval; the more data, the better the performance.
- **High-performance production-grade system**, directly deployable for enterprise applications.
- **User-friendly**, no need for cumbersome configurations, one-click installation and deployment, ready to use.
- **Multi knowledge base QA** Support selecting multiple knowledge bases for Q&A




### Architecture
<div align="center">
<img src="docs/images/qanything_arch.png" width = "700" alt="qanything_system" align=center />
</div>

#### Why 2 stage retrieval?
In scenarios with a large volume of knowledge base data, the advantages of a two-stage approach are very clear. If only a first-stage embedding retrieval is used, there will be a problem of retrieval degradation as the data volume increases, as indicated by the green line in the following graph. However, after the second-stage reranking, there can be a stable increase in accuracy, **the more data, the better the performance**.
<div align="center">
<img src="docs/images/two_stage_retrieval.jpg" width = "500" alt="two stage retrievaal" align=center />
</div>

QAnything uses the retrieval component [BCEmbedding](https://github.com/netease-youdao/BCEmbedding), which is distinguished for its bilingual and crosslingual proficiency. BCEmbedding excels in bridging Chinese and English linguistic gaps, which achieves
- **A high performance on <a href="https://github.com/netease-youdao/BCEmbedding/tree/master?tab=readme-ov-file#evaluate-semantic-representation-by-mteb" target="_Self">Semantic Representation Evaluations in MTEB</a>**;
- **A new benchmark in the realm of <a href="https://github.com/netease-youdao/BCEmbedding/tree/master?tab=readme-ov-file#evaluate-rag-by-llamaindex" target="_Self">RAG Evaluations in LlamaIndex</a>**.


#### 1st Retrieval（embedding）
| Model | Retrieval | STS | PairClassification | Classification | Reranking | Clustering | Avg |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|  
| bge-base-en-v1.5 | 37.14 | 55.06 | 75.45 | 59.73 | 43.05 | 37.74 | 47.20 |  
| bge-base-zh-v1.5 | 47.60 | 63.72 | 77.40 | 63.38 | 54.85 | 32.56 | 53.60 |  
| bge-large-en-v1.5 | 37.15 | 54.09 | 75.00 | 59.24 | 42.68 | 37.32 | 46.82 |  
| bge-large-zh-v1.5 | 47.54 | 64.73 | **79.14** | 64.19 | 55.88 | 33.26 | 54.21 |  
| jina-embeddings-v2-base-en | 31.58 | 54.28 | 74.84 | 58.42 | 41.16 | 34.67 | 44.29 |  
| m3e-base | 46.29 | 63.93 | 71.84 | 64.08 | 52.38 | 37.84 | 53.54 |  
| m3e-large | 34.85 | 59.74 | 67.69 | 60.07 | 48.99 | 31.62 | 46.78 |  
| ***bce-embedding-base_v1*** | **57.60** | **65.73** | 74.96 | **69.00** | **57.29** | **38.95** | ***59.43*** |  

- More evaluation details please check [Embedding Models Evaluation Summary](https://github.com/netease-youdao/BCEmbedding/blob/master/Docs/EvaluationSummary/embedding_eval_summary.md)。

#### 2nd Retrieval（rerank）
| Model | Reranking | Avg |  
|:-------------------------------|:--------:|:--------:|  
| bge-reranker-base | 57.78 | 57.78 |  
| bge-reranker-large | 59.69 | 59.69 |  
| ***bce-reranker-base_v1*** | **60.06** | ***60.06*** |  

- More evaluation details please check [Reranker Models Evaluation Summary](https://github.com/netease-youdao/BCEmbedding/blob/master/Docs/EvaluationSummary/reranker_eval_summary.md)

#### RAG Evaluations in LlamaIndex（embedding and rerank）

<img src="https://github.com/netease-youdao/BCEmbedding/blob/master/Docs/assets/rag_eval_multiple_domains_summary.jpg">

***NOTE:***

- In `WithoutReranker` setting, our `bce-embedding-base_v1` outperforms all the other embedding models.
- With fixing the embedding model, our `bce-reranker-base_v1` achieves the best performance.
- **The combination of `bce-embedding-base_v1` and `bce-reranker-base_v1` is SOTA**.
- If you want to use embedding and rerank separately, please refer to [BCEmbedding](https://github.com/netease-youdao/BCEmbedding)

#### LLM

The open source version of QAnything is based on QwenLM and has been fine-tuned on a large number of professional question-answering datasets. It greatly enhances the ability of question-answering.
If you need to use it for commercial purposes, please follow the license of QwenLM. For more details, please refer to: [QwenLM](https://github.com/QwenLM/Qwen)

## Before You Start
**Star us on GitHub, and be instantly notified for new release!**
![star_us](https://github.com/netease-youdao/QAnything/assets/29041332/fd5e5926-b9b2-4675-9f60-6cdcaca18e14)
* [🏄 Try QAnything Online](https://qanything.ai)
* [📚 Try read.youdao.com | 有道速读](https://read.youdao.com)
* [🛠️ Only use our BCEmbedding(embedding & rerank)](https://github.com/netease-youdao/BCEmbedding)
* [📖 FAQ](FAQ_zh.md)

## Getting Started

### Prerequisites
#### **For Linux**
|**System**| **Required item**        | **Minimum Requirement**   | **Note**                                                                           |
|---------------------------|--------------------------|---------------------------|-------------------------------------------------------------------------|
|Linux | Single NVIDIA GPU Memory  <br> or Double NVIDIA GPU Memory | >= 16GB <br> >= 11GB + 5G    | NVIDIA 3090 x 1 recommended <br> NVIDIA 2080TI × 2 recommended            |
|      | NVIDIA Driver Version    | >= 525.105.17             |                                                                                               |
|      | CUDA Version             | >= 12.0                   |                                                                                               |
|      |  Docker version           | >= 20.10.5                |  [Docker install](https://docs.docker.com/engine/install/) |
|      | docker compose  version  | >= 2.23.3                 | [docker compose install](https://docs.docker.com/compose/install/)  |

#### **For Winodws 11 with WSL 2**
|**System**| **Required item**        | **Minimum Requirement**   | **Note**                                                                           |
|---------------------------|--------------------------|---------------------------|-----------------------------------------------------------------------------------------------|
|Windows 11 with WSL 2| Single NVIDIA GPU Memory <br> or Double NVIDIA GPU Memory | >= 16GB  <br>  >= 11GB + 5G                | NVIDIA 3090 <br> NVIDIA 2080TI × 2                                                                      |                                                                |
|      | GEFORCE EXPERIENCE    | >= 546.33 |[GEFORCE EXPERIENCE download](https://us.download.nvidia.com/GFE/GFEClient/3.27.0.120/GeForce_Experience_v3.27.0.120.exe)             |                                                                                               |
|      |  Docker Desktop           | >=  4.26.1（131620）     | [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)     |


### Installation
### step1: pull qanything repository
```shell
git clone https://github.com/netease-youdao/QAnything.git
```
### step2: Enter the project root directory and execute the startup script.
* [📖 QAnything_Startup_Usage](docs/QAnything_Startup_Usage_README.md)
* Get detailed usage of LLM interface by ```bash ./run.sh -h```
  

If you are in the Windows11 system: Need to enter the **WSL** environment.
```shell
cd QAnything
bash run.sh  # Start on GPU 0 by default.
```

<details>
<summary>(Optional) Specify GPU startup</summary>

```shell
cd QAnything
bash ./run.sh -c local -i 0 -b default  # gpu id 0
```
</details>

<details>
<summary>(Optional) Specify multi-GPU startup </summary>

```shell
cd QAnything
bash ./run.sh -c local -i 0,1 -b default  # gpu ids: 0,1, Please confirm how many GPUs are available. Supports up to two cards for startup. 
```
</details>

### step3: start to experience
#### Front end
After successful installation, you can experience the application by entering the following addresses in your web browser.

- Front end address: http://`your_host`:5052/qanything/

#### API
If you want to visit API, please refer to the following address:
- API address: http://`your_host`:8777/api/
- For detailed API documentation, please refer to [QAnything API documentation](docs/API.md)

### Close service
If you are in the Windows11 system: Need to enter the WSL environment.
```shell
bash close.sh
```


## FAQ
[FAQ](FAQ_zh.md)


## Usage

### Cross-lingual: Multiple English paper Q&A
[![](docs/videos/multi_paper_qa.mp4)](https://github.com/netease-youdao/QAnything/assets/141105427/8915277f-c136-42b8-9332-78f64bf5df22)
### Information extraction
[![](docs/videos/information_extraction.mp4)](https://github.com/netease-youdao/QAnything/assets/141105427/b9e3be94-183b-4143-ac49-12fa005a8a9a
)
### Various files
[![](docs/videos/various_files_qa.mp4)](https://github.com/netease-youdao/QAnything/assets/141105427/7ede63c1-4c7f-4557-bd2c-7c51a44c8e0b)
### Web Q&A
[![](docs/videos/web_qa.mp4)](https://github.com/netease-youdao/QAnything/assets/141105427/d30942f7-6dbd-4013-a4b6-82f7c2a5fbee)

### API Document
If you need to access the API, please refer to the [QAnything API documentation](docs/API.md).



## Community & Support

### Discord <a href="https://discord.gg/5uNpPsEJz8"><img src="https://img.shields.io/discord/1197874288963895436?style=social&logo=discord"></a>
Welcome to the QAnything [Discord](https://discord.gg/5uNpPsEJz8) community



### WeChat Group

Welcome to scan the QR code below and join the WeChat group.

<img src="docs/images/Wechat.jpg" width="30%" height="auto">

### Email
If you need to contact our team privately, please reach out to us via the following email:

qanything@rd.netease.com

### GitHub issues
Reach out to the maintainer at one of the following places:

- [Github issues](https://github.com/netease-youdao/QAnything/issues)
- Contact options listed on [this GitHub profile](https://github.com/netease-youdao)


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=netease-youdao/QAnything,netease-youdao/BCEmbedding&type=Date)](https://star-history.com/#netease-youdao/QAnything&netease-youdao/BCEmbedding&Date)



## License

`QAnything` is licensed under [Apache 2.0 License](./LICENSE)

## Acknowledgments
`QAnything` adopts dependencies from the following:
- Thanks to our [BCEmbedding](https://github.com/netease-youdao/BCEmbedding) for the excellent embedding and rerank model. 
- Thanks to [Qwen](https://github.com/QwenLM/Qwen) for strong base language models.
- Thanks to [Triton Inference Server](https://github.com/triton-inference-server/server) for providing great open source inference serving.
- Thanks to [FastChat](https://github.com/lm-sys/FastChat) for providing a fully OpenAI-compatible API server.
- Thanks to [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) and [vllm](https://github.com/vllm-project/vllm) for highly optimized LLM inference backend.
- Thanks to [Langchain](https://github.com/langchain-ai/langchain) for the wonderful llm application framework. 
- Thanks to [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) for the inspiration provided on local knowledge base Q&A.
- Thanks to [Milvus](https://github.com/milvus-io/milvus) for the excellent semantic search library.
- Thanks to [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for its ease-to-use OCR library.
- Thanks to [Sanic](https://github.com/sanic-org/sanic) for the powerful web service framework.


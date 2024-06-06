<div align="center">

  <a href="https://github.com/netease-youdao/QAnything">
    <!-- ロゴのパスをここに提供してください -->
    <img src="docs/images/qanything_logo.png" alt="ロゴ" width="800">
  </a>

# **Q**uestion and **A**nswer based on **Anything**

<p align="center">
  <a href="./README.md">英語</a> |
  <a href="./README_zh.md">簡体字中国語</a> |
  <a href="./README_jp.md">日本語</a>
</p>

</div>

<div align="center">

<a href="https://qanything.ai"><img src="https://img.shields.io/badge/オンラインで試す-qanything.ai-purple"></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://read.youdao.com#/home"><img src="https://img.shields.io/badge/オンラインで試す-read.youdao.com-purple"></a>
&nbsp;&nbsp;&nbsp;&nbsp;

<a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://github.com/netease-youdao/QAnything/pulls"><img src="https://img.shields.io/badge/PRs-歓迎-red"></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://twitter.com/YDopensource"><img src="https://img.shields.io/badge/follow-%40YDOpenSource-1DA1F2?logo=twitter&style={style}"></a>
&nbsp;&nbsp;&nbsp;&nbsp;

<a href="https://discord.gg/5uNpPsEJz8"><img src="https://img.shields.io/discord/1197874288963895436?style=social&logo=discord"></a>
&nbsp;&nbsp;&nbsp;&nbsp;

</div>

<details open="open">
<summary>目次</summary>

- [QAnythingとは](#qanythingとは)
  - [主な特徴](#主な特徴)
  - [アーキテクチャ](#アーキテクチャ)
- [最新の更新](#-最新の更新)
- [始める前に](#始める前に)
- [始め方](#始め方)
  - [前提条件](#前提条件)
  - [インストール(純粋なPython環境)](#インストール純粋なpython環境)
  - [インストール(Docker)](#インストールdocker)
  - [オフラインでのインストール](#オフラインでのインストール)
- [FAQ](#faq)
- [使用法](#使用法)
  - [APIドキュメント](#apiドキュメント)
- [ロードマップとフィードバック](#%EF%B8%8F-ロードマップとフィードバック)
- [コミュニティとサポート](#コミュニティとサポート)
- [ライセンス](#ライセンス)
- [謝辞](#謝辞)

</details>

# 🚀 重要な更新
<h1><span style="color:red;">重要なことは三度言います。</span></h1>

# [2024-05-17:最新のインストールと使用ドキュメント](https://github.com/netease-youdao/QAnything/blob/master/QAnything%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md) 
# [2024-05-17:最新のインストールと使用ドキュメント](https://github.com/netease-youdao/QAnything/blob/master/QAnything%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md) 
# [2024-05-17:最新のインストールと使用ドキュメント](https://github.com/netease-youdao/QAnything/blob/master/QAnything%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md)

## ビジネスに関するお問い合わせ：
### 010-82558901
![](docs/images/business.jpeg)

# QAnythingとは？
`QAnything`(**Q**uestion and **A**nswer based on **Anything**) は、さまざまなファイル形式やデータベースをサポートするローカル知識ベースの質問応答システムで、オフラインでのインストールと使用が可能です。

`QAnything`を使用すると、任意の形式のローカルファイルをドロップするだけで、正確で迅速かつ信頼性の高い回答を得ることができます。

現在サポートされている形式には、**PDF(pdf)**、**Word(docx)**、**PPT(pptx)**、**XLS(xlsx)**、**Markdown(md)**、**電子メール(eml)**、**TXT(txt)**、**画像(jpg、jpeg、png)**、**CSV(csv)**、**ウェブリンク(html)**などがあります。さらに多くの形式が近日中にサポートされる予定です。

## 主な特徴

- **データセキュリティ**、インストールと使用の全過程でネットワークケーブルを抜いた状態でのサポート。
- **クロスランゲージQAサポート**、ドキュメントの言語に関係なく、中国語と英語のQAを自由に切り替えることができます。
- **大量データQAのサポート**、2段階の検索ランキングにより、大規模データ検索の劣化問題を解決。データが多ければ多いほど、パフォーマンスが向上します。
- **高性能なプロダクショングレードのシステム**、企業アプリケーションに直接デプロイ可能。
- **ユーザーフレンドリー**、面倒な設定は不要。ワンクリックでのインストールとデプロイ、すぐに使用可能。
- **複数の知識ベースQAのサポート**、複数の知識ベースを選択してQ&Aを行うことができます。

## アーキテクチャ
<div align="center">
<img src="docs/images/qanything_arch.png" width = "700" alt="qanything_system" align=center />
</div>

### なぜ2段階の検索なのか？
知識ベースのデータ量が多いシナリオでは、2段階のアプローチの利点は非常に明確です。第1段階の埋め込み検索のみを使用すると、データ量が増加するにつれて検索の劣化が発生します。以下のグラフの緑色の線で示されています。しかし、第2段階の再ランキング後、精度が安定して向上します。つまり、**データが多ければ多いほど、パフォーマンスが向上します**。
<div align="center">
<img src="docs/images/two_stage_retrieval.jpg" width = "500" alt="two stage retrievaal" align=center />
</div>

QAnythingが使用している検索コンポーネント[BCEmbedding](https://github.com/netease-youdao/BCEmbedding)は、そのバイリンガルおよびクロスリンガルの能力で際立っています。BCEmbeddingは、中国語と英語の言語間のギャップを埋めることに優れており、以下を実現しています。
- **<a href="https://github.com/netease-youdao/BCEmbedding/tree/master?tab=readme-ov-file#evaluate-semantic-representation-by-mteb" target="_Self">MTEBによるセマンティック表現評価</a>での高いパフォーマンス**;
- **<a href="https://github.com/netease-youdao/BCEmbedding/tree/master?tab=readme-ov-file#evaluate-rag-by-llamaindex" target="_Self">LlamaIndexによるRAG評価</a>での新たなベンチマーク**。

### 1段階目の検索（埋め込み）
| モデル | 検索 | STS | ペア分類 | 分類 | 再ランキング | クラスタリング | 平均 |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|  
| bge-base-en-v1.5 | 37.14 | 55.06 | 75.45 | 59.73 | 43.05 | 37.74 | 47.20 |  
| bge-base-zh-v1.5 | 47.60 | 63.72 | 77.40 | 63.38 | 54.85 | 32.56 | 53.60 |  
| bge-large-en-v1.5 | 37.15 | 54.09 | 75.00 | 59.24 | 42.68 | 37.32 | 46.82 |  
| bge-large-zh-v1.5 | 47.54 | 64.73 | **79.14** | 64.19 | 55.88 | 33.26 | 54.21 |  
| jina-embeddings-v2-base-en | 31.58 | 54.28 | 74.84 | 58.42 | 41.16 | 34.67 | 44.29 |  
| m3e-base | 46.29 | 63.93 | 71.84 | 64.08 | 52.38 | 37.84 | 53.54 |  
| m3e-large | 34.85 | 59.74 | 67.69 | 60.07 | 48.99 | 31.62 | 46.78 |  
| ***bce-embedding-base_v1*** | **57.60** | **65.73** | 74.96 | **69.00** | **57.29** | **38.95** | ***59.43*** |  

- 詳細な評価結果については、[埋め込みモデル評価のまとめ](https://github.com/netease-youdao/BCEmbedding/blob/master/Docs/EvaluationSummary/embedding_eval_summary.md)をご覧ください。

### 2段階目の検索（再ランキング）
| モデル | 再ランキング | 平均 |  
|:-------------------------------|:--------:|:--------:|  
| bge-reranker-base | 57.78 | 57.78 |  
| bge-reranker-large | 59.69 | 59.69 |  
| ***bce-reranker-base_v1*** | **60.06** | ***60.06*** |  

- 詳細な評価結果については、[再ランキングモデル評価のまとめ](https://github.com/netease-youdao/BCEmbedding/blob/master/Docs/EvaluationSummary/reranker_eval_summary.md)をご覧ください。

### LlamaIndexによるRAG評価（埋め込みと再ランキング）

<img src="https://github.com/netease-youdao/BCEmbedding/blob/master/Docs/assets/rag_eval_multiple_domains_summary.jpg">

***注記:***

- `WithoutReranker`設定では、私たちの`bce-embedding-base_v1`が他のすべての埋め込みモデルを上回ります。
- 埋め込みモデルを固定した場合、私たちの`bce-reranker-base_v1`が最高のパフォーマンスを達成します。
- **`bce-embedding-base_v1`と`bce-reranker-base_v1`の組み合わせがSOTAです**。
- 埋め込みと再ランキングを個別に使用したい場合は、[BCEmbedding](https://github.com/netease-youdao/BCEmbedding)を参照してください。

### LLM

オープンソース版のQAnythingはQwenLMに基づいており、多数の専門的な質問応答データセットで微調整されています。QwenLMのベースに基づいて、質問応答の能力を大幅に強化しています。
商用目的で使用する場合は、QwenLMのライセンスに従ってください。詳細については、[QwenLM](https://github.com/QwenLM/Qwen)を参照してください。

# 🚀 最新の更新

- ***2024-05-20***: **OpenAI APIと互換性のある他の大規模モデルサービスをサポートし、最適化された強力なPDFパーサーを提供します。** - 詳細はこちら👉 [v1.4.1](https://github.com/netease-youdao/QAnything/releases/tag/v1.4.1)
- ***2024-04-26***: **ウェブ検索、FAQ、カスタムボット、ファイルトレーサビリティプレビューなどをサポートします。** - 詳細はこちら👉 [v1.4.0](https://github.com/netease-youdao/QAnything/releases/tag/v1.4.0-python)
- ***2024-04-03***: **純粋なPython環境でのインストールをサポートします。ハイブリッド検索をサポートします。** - 詳細はこちら👉 [v1.3.0](https://github.com/netease-youdao/QAnything/releases/tag/v1.3.0)
- ***2024-01-29***: **カスタム大規模モデルのサポート、OpenAI APIおよび他のオープンソース大規模モデルを含む、最小GPU要件をGTX 1050
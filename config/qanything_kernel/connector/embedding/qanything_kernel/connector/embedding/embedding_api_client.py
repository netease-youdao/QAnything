"""API Embedding 客户端 - 支持 OpenAI、DashScope、Ollama 等

这个模块提供了通过 API 调用获取文本向量的功能。
支持多个 API 提供商，包括 OpenAI、阿里云 DashScope、本地 Ollama 等。

使用示例：
    from qanything_kernel.connector.embedding.embedding_api_client import APIEmbeddings

    # OpenAI
    embeddings = APIEmbeddings({
        "provider": "openai",
        "model_name": "text-embedding-3-small",
        "api_key": "your-api-key",
        "base_url": "https://api.openai.com/v1"
    })

    # Ollama（本地）
    embeddings = APIEmbeddings({
        "provider": "ollama",
        "model_name": "nomic-embed-text",
        "base_url": "http://localhost:11434/v1"
    })

    # 获取向量
    vectors = embeddings.embed_documents(["文本1", "文本2"])
"""
from typing import List, Dict, Any
from langchain_core.embeddings import Embeddings
from qanything_kernel.utils.custom_log import debug_logger
import os
import time


class APIEmbeddings(Embeddings):
    """API 调用的 Embedding 客户端

    支持多个 API 提供商：
    - openai: OpenAI 官方 API
    - dashscope: 阿里云 DashScope（通义千问）
    - ollama: 本地 Ollama 服务
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化 API Embedding 客户端

        Args:
            config: 配置字典，包含以下字段：
                - provider: API 提供商（openai | dashscope | ollama）
                - model_name: 模型名称
                - api_key: API 密钥（可选，支持环境变量）
                - base_url: API Base URL
                - batch_size: 批处理大小（默认 16）
                - timeout: 超时时间（秒，默认 30）
        """
        self.provider = config.get('provider', 'openai')
        self.model_name = config.get('model_name')
        self.api_key = self._resolve_env_var(config.get('api_key', ''))
        self.base_url = config.get('base_url', '')
        self.batch_size = config.get('batch_size', 16)
        self.timeout = config.get('timeout', 30)

        # 验证必填字段
        if not self.model_name:
            raise ValueError("必须指定 model_name")

        # 对于 OpenAI 和 DashScope，需要 API Key
        if self.provider in ['openai', 'dashscope'] and not self.api_key:
            raise ValueError(f"{self.provider} 需要提供 api_key")

        debug_logger.info(
            f"Initialized API Embedding: provider={self.provider}, "
            f"model={self.model_name}, base_url={self.base_url}"
        )

    def _resolve_env_var(self, value: str) -> str:
        """解析环境变量

        如果值格式为 ${ENV_VAR_NAME}，则从环境变量中读取。

        Args:
            value: 原始值

        Returns:
            解析后的值
        """
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            resolved = os.getenv(env_var, '')
            if not resolved:
                debug_logger.warning(f"环境变量 {env_var} 未设置")
            return resolved
        return value

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed 多个文档

        Args:
            texts: 文本列表

        Returns:
            向量列表，每个向量是一个浮点数列表
        """
        if not texts:
            return []

        all_embeddings = []

        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                if self.provider == "openai":
                    batch_embeddings = self._embed_openai(batch)
                elif self.provider == "dashscope":
                    batch_embeddings = self._embed_dashscope(batch)
                elif self.provider == "ollama":
                    batch_embeddings = self._embed_ollama(batch)
                else:
                    raise ValueError(f"不支持的 provider: {self.provider}")

                all_embeddings.extend(batch_embeddings)
                debug_logger.info(f"Embedded batch {i // self.batch_size + 1}, "
                                f"size: {len(batch)}")

            except Exception as e:
                debug_logger.error(f"Embedding batch {i // self.batch_size + 1} failed: {e}")
                raise

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed 单个查询

        Args:
            text: 查询文本

        Returns:
            向量（浮点数列表）
        """
        return self.embed_documents([text])[0]

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """OpenAI Embedding

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "使用 OpenAI Embedding 需要安装 openai 库：\n"
                "pip install openai"
            )

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

        response = client.embeddings.create(
            model=self.model_name,
            input=texts
        )

        # 按 index 排序（OpenAI 可能乱序返回）
        sorted_embeddings = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_embeddings]

    def _embed_dashscope(self, texts: List[str]) -> List[List[float]]:
        """阿里云 DashScope Embedding

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        try:
            import dashscope
            from dashscope import TextEmbedding
        except ImportError:
            raise ImportError(
                "使用 DashScope Embedding 需要安装 dashscope 库：\n"
                "pip install dashscope"
            )

        # 设置 API Key
        dashscope.api_key = self.api_key

        # 调用 API
        response = TextEmbedding.call(
            model=self.model_name,
            input=texts
        )

        if response.status_code != 200:
            raise Exception(f"DashScope API 调用失败: {response.code} - {response.message}")

        return [item['embedding'] for item in response.output['embeddings']]

    def _embed_ollama(self, texts: List[str]) -> List[List[float]]:
        """Ollama 本地 Embedding

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "使用 Ollama Embedding 需要安装 openai 库：\n"
                "pip install openai"
            )

        # Ollama 支持 OpenAI 兼容接口
        client = OpenAI(
            api_key="ollama",  # Ollama 不需要真实的 API Key
            base_url=self.base_url,
            timeout=self.timeout
        )

        response = client.embeddings.create(
            model=self.model_name,
            input=texts
        )

        return [item.embedding for item in response.data]

    @property
    def model_version(self) -> str:
        """获取模型版本"""
        return f"{self.provider}/{self.model_name}"

"""Embedding 工厂类 - 根据配置创建对应的 Embedding 实例

这个模块提供了一个统一的接口来创建不同类型的 Embedding 实例，
支持本地 ONNX 模型、本地 PyTorch 模型和 API 调用。

使用示例：
    from qanything_kernel.connector.embedding.embedding_factory import EmbeddingFactory

    # 使用默认配置（config/embedding.yaml）
    embeddings = EmbeddingFactory.create()

    # 使用自定义配置
    embeddings = EmbeddingFactory.create("/path/to/custom_config.yaml")

    # 获取 embeddings
    vectors = embeddings.embed_documents(["文本1", "文本2"])
    query_vector = embeddings.embed_query("查询文本")
"""
import yaml
from pathlib import Path
from typing import Optional, Union
from qanything_kernel.utils.custom_log import debug_logger


class EmbeddingFactory:
    """Embedding 工厂类

    根据配置文件创建对应的 Embedding 实例。
    支持的类型：
    - local_onnx: 本地 ONNX 模型（默认，推荐）
    - local_pytorch: 本地 PyTorch 模型
    - api: API 调用（OpenAI、DashScope、Ollama 等）
    """

    @staticmethod
    def create(config_path: Optional[Union[str, Path]] = None):
        """创建 Embedding 实例

        Args:
            config_path: 配置文件路径。如果为 None，使用默认路径 config/embedding.yaml

        Returns:
            Embedding 实例，实现了 LangChain 的 Embeddings 接口

        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置格式错误或不支持的模型类型
        """
        # 确定配置文件路径
        if config_path is None:
            # 默认路径：项目根目录/config/embedding.yaml
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "embedding.yaml"
        else:
            config_path = Path(config_path)

        # 检查配置文件是否存在
        if not config_path.exists():
            raise FileNotFoundError(f"Embedding 配置文件不存在: {config_path}")

        # 读取配置
        debug_logger.info(f"Loading embedding config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 验证配置格式
        if 'embedding' not in config:
            raise ValueError("配置文件格式错误：缺少 'embedding' 字段")

        embed_config = config['embedding']

        # 获取模型类型
        embed_type = embed_config.get('type', 'local_onnx')
        debug_logger.info(f"Creating embedding of type: {embed_type}")

        # 根据类型创建实例
        if embed_type == "local_onnx":
            return EmbeddingFactory._create_local_onnx(embed_config)
        elif embed_type == "local_pytorch":
            return EmbeddingFactory._create_local_pytorch(embed_config)
        elif embed_type == "api":
            return EmbeddingFactory._create_api(embed_config)
        else:
            raise ValueError(f"不支持的 Embedding 类型: {embed_type}。"
                           f"支持的类型：local_onnx, local_pytorch, api")

    @staticmethod
    def _create_local_onnx(config: dict):
        """创建本地 ONNX Embedding 实例"""
        try:
            from .embedding_onnx_client import ONNXEmbeddings
            local_config = config.get('local', {})
            return ONNXEmbeddings(local_config)
        except ImportError as e:
            raise ImportError(
                "创建 ONNX Embedding 失败。请确保安装了必要的依赖：\n"
                "pip install onnxruntime transformers"
            ) from e

    @staticmethod
    def _create_local_pytorch(config: dict):
        """创建本地 PyTorch Embedding 实例"""
        try:
            from .embedding_pytorch_client import PyTorchEmbeddings
            local_config = config.get('local', {})
            return PyTorchEmbeddings(local_config)
        except ImportError as e:
            raise ImportError(
                "创建 PyTorch Embedding 失败。请确保安装了必要的依赖：\n"
                "pip install torch transformers"
            ) from e

    @staticmethod
    def _create_api(config: dict):
        """创建 API Embedding 实例"""
        try:
            from .embedding_api_client import APIEmbeddings
            api_config = config.get('api', {})
            return APIEmbeddings(api_config)
        except ImportError as e:
            raise ImportError(
                "创建 API Embedding 失败。请确保安装了必要的依赖：\n"
                "pip install openai"
            ) from e

    @staticmethod
    def get_supported_types():
        """获取支持的 Embedding 类型列表"""
        return ["local_onnx", "local_pytorch", "api"]

    @staticmethod
    def get_default_config_path():
        """获取默认配置文件路径"""
        return Path(__file__).parent.parent.parent.parent / "config" / "embedding.yaml"

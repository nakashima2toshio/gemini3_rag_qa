"""
Embeddingクライアント抽象化レイヤー

OpenAI Embeddings API と Gemini Embeddings API の両方に対応する統一インターフェースを提供。

使用例:
    from helper_embedding import create_embedding_client

    # Gemini Embeddingクライアント（3072次元）
    embedding = create_embedding_client(provider="gemini")
    vector = embedding.embed_text("Hello world")
    print(f"Dimensions: {len(vector)}")  # 3072

    # バッチ処理
    vectors = embedding.embed_texts(["Hello", "World"], batch_size=100)
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import os
import logging
import time

from dotenv import load_dotenv

# SDK imports (モジュールレベルでインポート - モック対象)
from openai import OpenAI
from google import genai

load_dotenv()

logger = logging.getLogger(__name__)


# Gemini 3 アドバンテージ: 3072次元
DEFAULT_GEMINI_EMBEDDING_DIMS = 3072
DEFAULT_OPENAI_EMBEDDING_DIMS = 1536


class EmbeddingClient(ABC):
    """Embeddingクライアント抽象基底クラス"""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding次元数を返す"""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        単一テキストのEmbedding生成

        Args:
            text: 入力テキスト

        Returns:
            Embeddingベクトル（floatのリスト）
        """
        pass

    @abstractmethod
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        バッチEmbedding生成

        Args:
            texts: 入力テキストのリスト
            batch_size: バッチサイズ

        Returns:
            Embeddingベクトルのリスト
        """
        pass


class OpenAIEmbedding(EmbeddingClient):
    """OpenAI Embeddings API実装"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dims: int = DEFAULT_OPENAI_EMBEDDING_DIMS
    ):
        """
        Args:
            api_key: OpenAI APIキー（Noneの場合は環境変数から取得）
            model: 使用モデル
            dims: Embedding次元数（1536推奨）
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY が設定されていません")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed_text(self, text: str) -> List[float]:
        """単一テキストのEmbedding生成"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self._dims
        )
        return response.data[0].embedding

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """バッチEmbedding生成"""
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self._dims
            )

            # レスポンスはindex順にソートされていない場合があるため、ソート
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_data]
            all_embeddings.extend(batch_embeddings)

            # レート制限対策
            if i + batch_size < len(texts):
                time.sleep(0.1)

        return all_embeddings


class GeminiEmbedding(EmbeddingClient):
    """Gemini Embeddings API実装（3072次元: Gemini 3アドバンテージ）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-embedding-001",
        dims: int = DEFAULT_GEMINI_EMBEDDING_DIMS
    ):
        """
        Args:
            api_key: Gemini APIキー（Noneの場合は環境変数から取得）
            model: 使用モデル
            dims: Embedding次元数（3072推奨: Gemini 3最大精度）
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY が設定されていません")

        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self._dims = dims

        logger.info(f"GeminiEmbedding initialized: model={model}, dims={dims}")

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed_text(self, text: str) -> List[float]:
        """単一テキストのEmbedding生成（3072次元）"""
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config={"output_dimensionality": self._dims}
        )
        return response.embeddings[0].values

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        バッチEmbedding生成

        Note: Gemini APIは現在、1リクエストに1テキストのみ対応。
              ループ処理でバッチを模擬し、レート制限を考慮。
        """
        all_embeddings: List[List[float]] = []

        for i, text in enumerate(texts):
            embedding = self.embed_text(text)
            all_embeddings.append(embedding)

            # レート制限対策（100リクエストごとに待機）
            if (i + 1) % batch_size == 0 and i + 1 < len(texts):
                logger.debug(f"Processed {i + 1}/{len(texts)} embeddings, sleeping...")
                time.sleep(1.0)

        return all_embeddings

    def embed_texts_batch(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        バッチEmbedding生成（Gemini Batch API使用）

        Note: 大量データの場合はBatch APIを使用すると50%割引
              現時点では通常のAPIを使用（Batch API実装は将来対応）
        """
        # 現在はembed_textsと同じ実装
        return self.embed_texts(texts)


def create_embedding_client(
    provider: str = "gemini",
    **kwargs
) -> EmbeddingClient:
    """
    Embeddingクライアントのファクトリ関数

    Args:
        provider: "openai" or "gemini"
        **kwargs: クライアント初期化パラメータ

    Returns:
        EmbeddingClientインスタンス

    Example:
        # Gemini Embedding（3072次元）
        embedding = create_embedding_client("gemini")

        # OpenAI Embedding（1536次元）
        embedding = create_embedding_client("openai")

        # カスタム次元数
        embedding = create_embedding_client("gemini", dims=1536)
    """
    if provider.lower() == "openai":
        return OpenAIEmbedding(**kwargs)
    elif provider.lower() == "gemini":
        return GeminiEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'gemini'")


# デフォルトプロバイダー設定（config.ymlから読み込む予定）
DEFAULT_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")


def get_default_embedding_client(**kwargs) -> EmbeddingClient:
    """デフォルト設定でEmbeddingクライアントを取得"""
    return create_embedding_client(DEFAULT_EMBEDDING_PROVIDER, **kwargs)


# Qdrant用のヘルパー関数
def get_embedding_dimensions(provider: str = "gemini") -> int:
    """
    指定プロバイダーのデフォルトEmbedding次元数を取得

    Qdrantコレクション作成時に使用

    Args:
        provider: "openai" or "gemini"

    Returns:
        次元数
    """
    if provider.lower() == "gemini":
        return DEFAULT_GEMINI_EMBEDDING_DIMS  # 3072
    elif provider.lower() == "openai":
        return DEFAULT_OPENAI_EMBEDDING_DIMS  # 1536
    else:
        raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    # 簡易テスト
    print("EmbeddingClient テスト")
    print("=" * 40)

    try:
        # Gemini Embeddingテスト
        print("\n[Gemini Embedding Test]")
        gemini = create_embedding_client("gemini")
        print(f"Dimensions: {gemini.dimensions}")

        vector = gemini.embed_text("これはテストです")
        print(f"Vector length: {len(vector)}")
        print(f"First 5 values: {vector[:5]}")

        # 次元数検証
        if len(vector) == DEFAULT_GEMINI_EMBEDDING_DIMS:
            print(f"[OK] 3072次元の検証: PASS")
        else:
            print(f"[NG] 3072次元の検証: FAIL (actual: {len(vector)})")

    except Exception as e:
        print(f"Gemini Error: {e}")

    try:
        # OpenAI Embeddingテスト
        print("\n[OpenAI Embedding Test]")
        openai_emb = create_embedding_client("openai")
        print(f"Dimensions: {openai_emb.dimensions}")

        vector = openai_emb.embed_text("これはテストです")
        print(f"Vector length: {len(vector)}")
        print(f"First 5 values: {vector[:5]}")

    except Exception as e:
        print(f"OpenAI Error: {e}")

    print("\n" + "=" * 40)
    print(f"Gemini default dims: {get_embedding_dimensions('gemini')}")
    print(f"OpenAI default dims: {get_embedding_dimensions('openai')}")
"""
helper_embedding.py 単体テスト

テスト実行:
    pytest tests/test_helper_embedding.py -v

Note:
    - 実APIを使用するテストは @pytest.mark.integration でマーク
    - CI環境では SKIP_INTEGRATION_TESTS=1 で統合テストをスキップ
"""

import pytest
import os
from unittest.mock import Mock, patch

# テスト対象
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_embedding import (
    EmbeddingClient,
    OpenAIEmbedding,
    GeminiEmbedding,
    create_embedding_client,
    get_default_embedding_client,
    get_embedding_dimensions,
    DEFAULT_GEMINI_EMBEDDING_DIMS,
    DEFAULT_OPENAI_EMBEDDING_DIMS,
)


# ====================================
# 定数テスト
# ====================================

class TestConstants:
    """定数値のテスト"""

    def test_gemini_default_dims(self):
        """Geminiデフォルト次元数: 3072"""
        assert DEFAULT_GEMINI_EMBEDDING_DIMS == 3072

    def test_openai_default_dims(self):
        """OpenAIデフォルト次元数: 1536"""
        assert DEFAULT_OPENAI_EMBEDDING_DIMS == 1536


# ====================================
# ファクトリ関数テスト
# ====================================

class TestCreateEmbeddingClient:
    """create_embedding_client ファクトリ関数のテスト"""

    def test_create_gemini_client(self):
        """Gemini Embeddingクライアント生成"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("helper_embedding.genai"):
                client = create_embedding_client("gemini")
                assert isinstance(client, GeminiEmbedding)
                assert client.dimensions == DEFAULT_GEMINI_EMBEDDING_DIMS

    def test_create_openai_client(self):
        """OpenAI Embeddingクライアント生成"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper_embedding.OpenAI"):
                client = create_embedding_client("openai")
                assert isinstance(client, OpenAIEmbedding)
                assert client.dimensions == DEFAULT_OPENAI_EMBEDDING_DIMS

    def test_invalid_provider(self):
        """不正なプロバイダー指定でエラー"""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_embedding_client("invalid_provider")

    def test_custom_dims(self):
        """カスタム次元数指定"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("helper_embedding.genai"):
                client = create_embedding_client("gemini", dims=1536)
                assert client.dimensions == 1536


class TestGetEmbeddingDimensions:
    """get_embedding_dimensions ヘルパー関数のテスト"""

    def test_gemini_dimensions(self):
        """Gemini次元数取得"""
        assert get_embedding_dimensions("gemini") == 3072

    def test_openai_dimensions(self):
        """OpenAI次元数取得"""
        assert get_embedding_dimensions("openai") == 1536

    def test_invalid_provider(self):
        """不正なプロバイダーでエラー"""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_embedding_dimensions("invalid")


# ====================================
# OpenAIEmbedding テスト
# ====================================

class TestOpenAIEmbedding:
    """OpenAIEmbedding クラスのテスト"""

    @pytest.fixture
    def mock_openai_client(self):
        """モックOpenAI Embeddingクライアント"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper_embedding.OpenAI") as mock_class:
                mock_instance = Mock()
                mock_class.return_value = mock_instance
                client = OpenAIEmbedding()
                return client, mock_instance

    def test_init_with_env_key(self):
        """環境変数からAPIキーを取得"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-test-key"}):
            with patch("helper_embedding.OpenAI"):
                client = OpenAIEmbedding()
                assert client.api_key == "env-test-key"

    def test_init_without_key_raises_error(self):
        """APIキーなしでエラー"""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIEmbedding()

    def test_dimensions_property(self, mock_openai_client):
        """次元数プロパティ"""
        client, _ = mock_openai_client
        assert client.dimensions == 1536

    def test_embed_text(self, mock_openai_client):
        """単一テキストEmbedding"""
        client, mock_instance = mock_openai_client

        # モックレスポンス
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_embedding]
        mock_instance.embeddings.create.return_value = mock_response

        result = client.embed_text("Hello")

        assert len(result) == 1536
        mock_instance.embeddings.create.assert_called_once()

    def test_embed_texts(self, mock_openai_client):
        """バッチEmbedding"""
        client, mock_instance = mock_openai_client

        # モックレスポンス
        mock_embeddings = []
        for i in range(3):
            mock_emb = Mock()
            mock_emb.embedding = [0.1 * (i + 1)] * 1536
            mock_emb.index = i
            mock_embeddings.append(mock_emb)

        mock_response = Mock()
        mock_response.data = mock_embeddings
        mock_instance.embeddings.create.return_value = mock_response

        result = client.embed_texts(["Hello", "World", "Test"])

        assert len(result) == 3
        assert all(len(v) == 1536 for v in result)


# ====================================
# GeminiEmbedding テスト
# ====================================

class TestGeminiEmbedding:
    """GeminiEmbedding クラスのテスト"""

    @pytest.fixture
    def mock_gemini_client(self):
        """モックGemini Embeddingクライアント"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("helper_embedding.genai") as mock_genai:
                mock_instance = Mock()
                mock_genai.Client.return_value = mock_instance
                client = GeminiEmbedding()
                return client, mock_instance

    def test_init_with_env_key(self):
        """環境変数からAPIキーを取得"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-test-key"}):
            with patch("helper_embedding.genai"):
                client = GeminiEmbedding()
                assert client.api_key == "env-test-key"

    def test_init_without_key_raises_error(self):
        """APIキーなしでエラー"""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GOOGLE_API_KEY", None)
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                GeminiEmbedding()

    def test_default_dimensions(self):
        """デフォルト次元数: 3072"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("helper_embedding.genai"):
                client = GeminiEmbedding()
                assert client.dimensions == 3072

    def test_custom_dimensions(self):
        """カスタム次元数"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("helper_embedding.genai"):
                client = GeminiEmbedding(dims=1536)
                assert client.dimensions == 1536

    def test_embed_text(self, mock_gemini_client):
        """単一テキストEmbedding（3072次元）"""
        client, mock_instance = mock_gemini_client

        # モックレスポンス
        mock_embedding = Mock()
        mock_embedding.values = [0.1] * 3072
        mock_response = Mock()
        mock_response.embeddings = [mock_embedding]
        mock_instance.models.embed_content.return_value = mock_response

        result = client.embed_text("Hello")

        assert len(result) == 3072
        mock_instance.models.embed_content.assert_called_once()

        # 次元数パラメータの確認
        call_args = mock_instance.models.embed_content.call_args
        config = call_args.kwargs.get("config", {})
        assert config.get("output_dimensionality") == 3072

    def test_embed_texts(self, mock_gemini_client):
        """バッチEmbedding"""
        client, mock_instance = mock_gemini_client

        # モックレスポンス（embed_textが3回呼ばれる）
        mock_embedding = Mock()
        mock_embedding.values = [0.1] * 3072
        mock_response = Mock()
        mock_response.embeddings = [mock_embedding]
        mock_instance.models.embed_content.return_value = mock_response

        result = client.embed_texts(["Hello", "World", "Test"])

        assert len(result) == 3
        assert all(len(v) == 3072 for v in result)
        assert mock_instance.models.embed_content.call_count == 3

    def test_model_parameter(self, mock_gemini_client):
        """モデルパラメータ"""
        client, mock_instance = mock_gemini_client

        mock_embedding = Mock()
        mock_embedding.values = [0.1] * 3072
        mock_response = Mock()
        mock_response.embeddings = [mock_embedding]
        mock_instance.models.embed_content.return_value = mock_response

        client.embed_text("Hello")

        call_args = mock_instance.models.embed_content.call_args
        assert call_args.kwargs.get("model") == "gemini-embedding-001"


# ====================================
# 統合テスト（実API使用）
# ====================================

@pytest.mark.skipif(
    os.getenv("SKIP_INTEGRATION_TESTS", "1") == "1",
    reason="Integration tests skipped (set SKIP_INTEGRATION_TESTS=0 to run)"
)
class TestIntegration:
    """統合テスト（実APIを使用）"""

    @pytest.mark.integration
    def test_gemini_embed_text_3072_dims(self):
        """Gemini: 実API 3072次元Embedding生成"""
        client = create_embedding_client("gemini")
        result = client.embed_text("これはテスト文章です")

        assert len(result) == 3072
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.integration
    def test_gemini_embed_texts(self):
        """Gemini: 実API バッチEmbedding"""
        client = create_embedding_client("gemini")
        texts = ["テスト1", "テスト2", "テスト3"]
        results = client.embed_texts(texts)

        assert len(results) == 3
        assert all(len(v) == 3072 for v in results)

    @pytest.mark.integration
    def test_openai_embed_text(self):
        """OpenAI: 実API Embedding生成"""
        client = create_embedding_client("openai")
        result = client.embed_text("This is a test")

        assert len(result) == 1536
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.integration
    def test_cosine_similarity_sanity_check(self):
        """コサイン類似度の妥当性チェック"""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        client = create_embedding_client("gemini")

        # 類似テキスト
        text1 = "富士山は日本で最も高い山です"
        text2 = "日本一高い山は富士山です"
        # 非類似テキスト
        text3 = "今日の天気は晴れです"

        vec1 = client.embed_text(text1)
        vec2 = client.embed_text(text2)
        vec3 = client.embed_text(text3)

        sim_12 = cosine_similarity([vec1], [vec2])[0][0]
        sim_13 = cosine_similarity([vec1], [vec3])[0][0]

        # 類似テキストの類似度 > 非類似テキストの類似度
        assert sim_12 > sim_13, f"sim_12={sim_12}, sim_13={sim_13}"
        # 類似度は0.5以上を期待
        assert sim_12 > 0.5, f"Similar texts should have high similarity: {sim_12}"
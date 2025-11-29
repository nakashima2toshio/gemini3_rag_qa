"""
helper_llm.py 単体テスト

テスト実行:
    pytest tests/test_helper_llm.py -v

Note:
    - 実APIを使用するテストは @pytest.mark.integration でマーク
    - CI環境では SKIP_INTEGRATION_TESTS=1 で統合テストをスキップ
"""

import pytest
import os
from typing import List
from unittest.mock import Mock, patch, MagicMock

from pydantic import BaseModel

# テスト対象
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_llm import (
    LLMClient,
    OpenAIClient,
    GeminiClient,
    create_llm_client,
    get_default_llm_client,
)


# テスト用Pydanticモデル
class TestResponse(BaseModel):
    message: str
    score: int


class QAPair(BaseModel):
    question: str
    answer: str


class QAPairsResponse(BaseModel):
    qa_pairs: List[QAPair]


# ====================================
# ファクトリ関数テスト
# ====================================

class TestCreateLLMClient:
    """create_llm_client ファクトリ関数のテスト"""

    def test_create_gemini_client(self):
        """Geminiクライアント生成"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("helper_llm.genai") as mock_genai:
                client = create_llm_client("gemini")
                assert isinstance(client, GeminiClient)

    def test_create_openai_client(self):
        """OpenAIクライアント生成"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper_llm.OpenAI") as mock_openai:
                client = create_llm_client("openai")
                assert isinstance(client, OpenAIClient)

    def test_invalid_provider(self):
        """不正なプロバイダー指定でエラー"""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_client("invalid_provider")

    def test_case_insensitive_provider(self):
        """プロバイダー名の大文字小文字を無視"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("helper_llm.genai"):
                client1 = create_llm_client("GEMINI")
                client2 = create_llm_client("Gemini")
                assert isinstance(client1, GeminiClient)
                assert isinstance(client2, GeminiClient)


# ====================================
# OpenAIClient テスト
# ====================================

class TestOpenAIClient:
    """OpenAIClient クラスのテスト"""

    @pytest.fixture
    def mock_openai_client(self):
        """モックOpenAIクライアント"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper_llm.OpenAI") as mock_class:
                mock_instance = Mock()
                mock_class.return_value = mock_instance
                client = OpenAIClient()
                return client, mock_instance

    def test_init_with_env_key(self):
        """環境変数からAPIキーを取得"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-test-key"}):
            with patch("helper_llm.OpenAI"):
                client = OpenAIClient()
                assert client.api_key == "env-test-key"

    def test_init_with_explicit_key(self):
        """明示的なAPIキー指定"""
        with patch("helper_llm.OpenAI"):
            client = OpenAIClient(api_key="explicit-key")
            assert client.api_key == "explicit-key"

    def test_init_without_key_raises_error(self):
        """APIキーなしでエラー"""
        with patch.dict(os.environ, {}, clear=True):
            # OPENAI_API_KEYを削除
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIClient()

    def test_generate_content(self, mock_openai_client):
        """テキスト生成"""
        client, mock_instance = mock_openai_client
        mock_response = Mock()
        mock_response.output_text = "Hello, world!"
        mock_instance.responses.create.return_value = mock_response

        result = client.generate_content("Say hello")

        assert result == "Hello, world!"
        mock_instance.responses.create.assert_called_once()

    def test_generate_content_with_system_instruction(self, mock_openai_client):
        """システム指示付きテキスト生成"""
        client, mock_instance = mock_openai_client
        mock_response = Mock()
        mock_response.output_text = "Response"
        mock_instance.responses.create.return_value = mock_response

        client.generate_content(
            "Question",
            system_instruction="You are a helpful assistant"
        )

        call_args = mock_instance.responses.create.call_args
        messages = call_args.kwargs["input"]
        assert len(messages) == 2
        assert messages[0]["role"] == "developer"
        assert messages[1]["role"] == "user"

    def test_generate_structured(self, mock_openai_client):
        """構造化出力生成"""
        client, mock_instance = mock_openai_client
        mock_response = Mock()
        mock_response.output_parsed = TestResponse(message="test", score=100)
        mock_instance.responses.parse.return_value = mock_response

        result = client.generate_structured("Generate test", TestResponse)

        assert isinstance(result, TestResponse)
        assert result.message == "test"
        assert result.score == 100

    def test_count_tokens(self, mock_openai_client):
        """トークンカウント"""
        client, _ = mock_openai_client
        with patch("helper_llm.tiktoken") as mock_tiktoken:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
            mock_tiktoken.encoding_for_model.return_value = mock_encoding

            count = client.count_tokens("Hello world")

            assert count == 5


# ====================================
# GeminiClient テスト
# ====================================

class TestGeminiClient:
    """GeminiClient クラスのテスト"""

    @pytest.fixture
    def mock_gemini_client(self):
        """モックGeminiクライアント"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("helper_llm.genai") as mock_genai:
                mock_instance = Mock()
                mock_genai.Client.return_value = mock_instance
                client = GeminiClient()
                return client, mock_instance

    def test_init_with_env_key(self):
        """環境変数からAPIキーを取得"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-test-key"}):
            with patch("helper_llm.genai"):
                client = GeminiClient()
                assert client.api_key == "env-test-key"

    def test_init_with_explicit_key(self):
        """明示的なAPIキー指定"""
        with patch("helper_llm.genai"):
            client = GeminiClient(api_key="explicit-key")
            assert client.api_key == "explicit-key"

    def test_init_without_key_raises_error(self):
        """APIキーなしでエラー"""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GOOGLE_API_KEY", None)
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                GeminiClient()

    def test_default_model(self):
        """デフォルトモデル設定"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("helper_llm.genai"):
                client = GeminiClient()
                assert client.default_model == "gemini-2.0-flash"

    def test_custom_model(self):
        """カスタムモデル設定"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("helper_llm.genai"):
                client = GeminiClient(default_model="gemini-3-pro-preview")
                assert client.default_model == "gemini-3-pro-preview"

    def test_generate_content(self, mock_gemini_client):
        """テキスト生成"""
        client, mock_instance = mock_gemini_client
        mock_response = Mock()
        mock_response.text = "こんにちは"
        mock_instance.models.generate_content.return_value = mock_response

        result = client.generate_content("Hello")

        assert result == "こんにちは"
        mock_instance.models.generate_content.assert_called_once()

    def test_generate_content_with_thinking_level(self, mock_gemini_client):
        """思考レベル指定付きテキスト生成"""
        client, mock_instance = mock_gemini_client
        client.default_model = "gemini-3-pro-preview"
        mock_response = Mock()
        mock_response.text = "Response"
        mock_instance.models.generate_content.return_value = mock_response

        client.generate_content("Question", thinking_level="high")

        call_args = mock_instance.models.generate_content.call_args
        config = call_args.kwargs.get("config", {})
        assert config.get("thinking_level") == "high"

    def test_generate_structured(self, mock_gemini_client):
        """構造化出力生成"""
        client, mock_instance = mock_gemini_client
        mock_response = Mock()
        mock_response.text = '{"message": "test", "score": 100}'
        mock_instance.models.generate_content.return_value = mock_response

        result = client.generate_structured("Generate", TestResponse)

        assert isinstance(result, TestResponse)
        assert result.message == "test"
        assert result.score == 100

    def test_generate_structured_invalid_json(self, mock_gemini_client):
        """構造化出力: 不正なJSONでエラー"""
        client, mock_instance = mock_gemini_client
        mock_response = Mock()
        mock_response.text = "not valid json"
        mock_instance.models.generate_content.return_value = mock_response

        with pytest.raises(ValueError, match="構造化出力のパースに失敗"):
            client.generate_structured("Generate", TestResponse)

    def test_count_tokens(self, mock_gemini_client):
        """トークンカウント"""
        client, mock_instance = mock_gemini_client
        mock_response = Mock()
        mock_response.total_tokens = 10
        mock_instance.models.count_tokens.return_value = mock_response

        count = client.count_tokens("Hello world")

        assert count == 10


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
    def test_gemini_generate_content(self):
        """Gemini: 実API テキスト生成"""
        client = create_llm_client("gemini")
        result = client.generate_content("Say 'Hello' in Japanese")
        assert "こんにちは" in result or "hello" in result.lower()

    @pytest.mark.integration
    def test_gemini_generate_structured(self):
        """Gemini: 実API 構造化出力"""
        client = create_llm_client("gemini")
        result = client.generate_structured(
            "Generate a Q&A pair about Mount Fuji",
            QAPairsResponse
        )
        assert isinstance(result, QAPairsResponse)
        assert len(result.qa_pairs) > 0

    @pytest.mark.integration
    def test_gemini_count_tokens(self):
        """Gemini: 実API トークンカウント"""
        client = create_llm_client("gemini")
        count = client.count_tokens("これはテストです")
        assert count > 0
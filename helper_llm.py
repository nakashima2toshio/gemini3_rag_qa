"""
LLMクライアント抽象化レイヤー

OpenAI API と Gemini 3 API の両方に対応する統一インターフェースを提供。

使用例:
    from helper_llm import create_llm_client

    # Geminiクライアント
    llm = create_llm_client(provider="gemini")
    response = llm.generate_content("Hello")

    # 構造化出力
    from pydantic import BaseModel
    class MyResponse(BaseModel):
        answer: str

    result = llm.generate_structured("質問", MyResponse)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Type
import os
import json
import logging

from pydantic import BaseModel
from dotenv import load_dotenv

# SDK imports (モジュールレベルでインポート - モック対象)
from openai import OpenAI
from google import genai
import tiktoken

load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """LLMクライアント抽象基底クラス"""

    @abstractmethod
    def generate_content(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        テキスト生成

        Args:
            prompt: 入力プロンプト
            model: 使用モデル（Noneの場合はデフォルト）
            system_instruction: システム指示
            temperature: 温度パラメータ
            max_output_tokens: 最大出力トークン数

        Returns:
            生成されたテキスト
        """
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[BaseModel],
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs
    ) -> BaseModel:
        """
        構造化出力生成

        Args:
            prompt: 入力プロンプト
            response_schema: 出力スキーマ（Pydanticモデル）
            model: 使用モデル
            system_instruction: システム指示
            temperature: 温度パラメータ
            max_output_tokens: 最大出力トークン数

        Returns:
            パースされたPydanticモデルインスタンス
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        トークン数をカウント

        Args:
            text: カウント対象のテキスト
            model: 使用モデル

        Returns:
            トークン数
        """
        pass


class OpenAIClient(LLMClient):
    """OpenAI API実装（既存互換用）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o-mini"
    ):
        """
        Args:
            api_key: OpenAI APIキー（Noneの場合は環境変数から取得）
            default_model: デフォルトモデル
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY が設定されていません")

        self.client = OpenAI(api_key=self.api_key)
        self.default_model = default_model

    def generate_content(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """テキスト生成（OpenAI Responses API）"""
        model = model or self.default_model

        messages = []
        if system_instruction:
            messages.append({"role": "developer", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        params: dict[str, Any] = {
            "model": model,
            "input": messages,
        }
        if max_output_tokens:
            params["max_output_tokens"] = max_output_tokens
        if temperature is not None:
            params["temperature"] = temperature

        response = self.client.responses.create(**params)
        return response.output_text

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[BaseModel],
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs
    ) -> BaseModel:
        """構造化出力生成（OpenAI Structured Outputs）"""
        model = model or self.default_model

        messages = []
        if system_instruction:
            messages.append({"role": "developer", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        params: dict[str, Any] = {
            "model": model,
            "input": messages,
            "text_format": response_schema,
        }
        if max_output_tokens:
            params["max_output_tokens"] = max_output_tokens
        if temperature is not None:
            params["temperature"] = temperature

        response = self.client.responses.parse(**params)
        return response.output_parsed

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """トークン数をカウント（tiktoken使用）"""
        model = model or self.default_model
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))


class GeminiClient(LLMClient):
    """Gemini 3 API実装"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-2.0-flash",
        thinking_level: str = "low"
    ):
        """
        Args:
            api_key: Gemini APIキー（Noneの場合は環境変数から取得）
            default_model: デフォルトモデル
            thinking_level: 思考レベル ("low" or "high")
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY が設定されていません")

        self.client = genai.Client(api_key=self.api_key)
        self.default_model = default_model
        self.thinking_level = thinking_level

    def generate_content(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        thinking_level: Optional[str] = None,
        **kwargs
    ) -> str:
        """テキスト生成（Gemini generate_content）"""
        model = model or self.default_model
        thinking = thinking_level or self.thinking_level

        config: dict[str, Any] = {}
        if system_instruction:
            config["system_instruction"] = system_instruction
        if temperature is not None:
            config["temperature"] = temperature
        if max_output_tokens:
            config["max_output_tokens"] = max_output_tokens

        # Gemini 3 の思考レベル設定（gemini-3モデルのみ）
        if "gemini-3" in model:
            config["thinking_level"] = thinking

        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config if config else None
        )

        return response.text

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[BaseModel],
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        thinking_level: Optional[str] = None,
        **kwargs
    ) -> BaseModel:
        """
        構造化出力生成（Gemini JSON Schema）

        Pydanticモデルからスキーマを生成し、JSONレスポンスをパース
        """
        model = model or self.default_model
        thinking = thinking_level or self.thinking_level

        config: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        }
        if system_instruction:
            config["system_instruction"] = system_instruction
        if temperature is not None:
            config["temperature"] = temperature
        if max_output_tokens:
            config["max_output_tokens"] = max_output_tokens

        # Gemini 3 の思考レベル設定
        if "gemini-3" in model:
            config["thinking_level"] = thinking

        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )

        # JSONをパースしてPydanticモデルに変換
        try:
            data = json.loads(response.text)
            return response_schema(**data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Raw response: {response.text}")
            raise ValueError(f"構造化出力のパースに失敗: {e}")

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        トークン数をカウント（Gemini count_tokens API）

        Note: Gemini APIのcount_tokensを使用
        """
        model = model or self.default_model

        response = self.client.models.count_tokens(
            model=model,
            contents=text
        )
        return response.total_tokens


def create_llm_client(
    provider: str = "gemini",
    **kwargs
) -> LLMClient:
    """
    LLMクライアントのファクトリ関数

    Args:
        provider: "openai" or "gemini"
        **kwargs: クライアント初期化パラメータ

    Returns:
        LLMClientインスタンス

    Example:
        # Geminiクライアント
        llm = create_llm_client("gemini")

        # OpenAIクライアント
        llm = create_llm_client("openai", default_model="gpt-4o")
    """
    if provider.lower() == "openai":
        return OpenAIClient(**kwargs)
    elif provider.lower() == "gemini":
        return GeminiClient(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'gemini'")


# デフォルトプロバイダー設定（config.ymlから読み込む予定）
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")


def get_default_llm_client(**kwargs) -> LLMClient:
    """デフォルト設定でLLMクライアントを取得"""
    return create_llm_client(DEFAULT_LLM_PROVIDER, **kwargs)


if __name__ == "__main__":
    # 簡易テスト
    print("LLMClient テスト")
    print("=" * 40)

    try:
        # Geminiクライアントテスト
        print("\n[Gemini Client Test]")
        gemini = create_llm_client("gemini")
        result = gemini.generate_content("こんにちは")
        print(f"Response: {result[:100]}...")

        # トークンカウント
        tokens = gemini.count_tokens("これはテストです")
        print(f"Token count: {tokens}")

    except Exception as e:
        print(f"Gemini Error: {e}")

    try:
        # OpenAIクライアントテスト
        print("\n[OpenAI Client Test]")
        openai_client = create_llm_client("openai")
        result = openai_client.generate_content("こんにちは")
        print(f"Response: {result[:100]}...")

    except Exception as e:
        print(f"OpenAI Error: {e}")
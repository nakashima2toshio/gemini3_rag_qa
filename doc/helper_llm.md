# LLMクライアント抽象化レイヤー (helper_llm.py)

## 1. 概要
`helper_llm.py` は、異なるLLMプロバイダー（現在はOpenAIとGoogle Gemini）間のAPIの差異を吸収し、統一されたインターフェースを提供するモジュールです。これにより、アプリケーションロジックを変更することなく、使用するAIモデルやプロバイダーを容易に切り替えることが可能になります。

## 2. 主な機能
- **統一インターフェース**: テキスト生成、構造化データ生成、トークンカウントの共通メソッドを提供。
- **マルチプロバイダー対応**: `openai` (GPTシリーズ) と `gemini` (Geminiシリーズ) をサポート。
- **構造化出力 (Structured Output)**: Pydanticモデルを使用した型安全なJSON出力をサポート。
- **モデル管理**: 利用可能なモデル、価格、トークン制限などのメタデータを一元管理。

## 3. クラス構成

### 3.1 LLMClient (抽象基底クラス)
全てのLLMクライアントが継承すべきインターフェースを定義します。

- `generate_content(prompt: str, model: str = None, **kwargs) -> str`: テキスト生成。
- `generate_structured(prompt: str, response_schema: Type[BaseModel], model: str = None, **kwargs) -> BaseModel`: 指定されたPydanticスキーマに従った構造化データを生成。
- `count_tokens(text: str, model: str = None) -> int`: 入力テキストのトークン数を計算。

### 3.2 OpenAIClient
OpenAI API (`openai` パッケージ) を使用した実装です。
- `gpt-4o`, `gpt-4o-mini` などをサポート。
- 構造化出力には `beta.chat.completions.parse` を使用。

### 3.3 GeminiClient
Google Generative AI SDK (`google-generativeai` パッケージ) を使用した実装です。
- `gemini-2.0-flash`, `gemini-1.5-pro` などをサポート。
- 構造化出力には `response_mime_type: "application/json"` とスキーマプロンプトを組み合わせて使用。

## 4. ファクトリ関数

### `create_llm_client(provider: str = "gemini", **kwargs) -> LLMClient`
指定されたプロバイダーに対応するクライアントインスタンスを生成します。

- `provider`: `"gemini"` (デフォルト) または `"openai"`。
- `**kwargs`: `api_key` や `default_model` などの追加設定。

## 5. 定数・設定
モジュール内では、以下のモデル情報が辞書形式で定義されており、ヘルパー関数を通じてアクセス可能です。

- **LLM_MODELS**: 利用可能なLLMモデル名のリスト。
- **LLM_PRICING**: 各モデルの入力/出力トークン単価。
- **LLM_LIMITS**: 各モデルのコンテキストウィンドウサイズ。
- **EMBEDDING_MODELS**: 利用可能な埋め込みモデル。

## 6. 使用例

### 基本的なテキスト生成
```python
from helper_llm import create_llm_client

# Geminiクライアントの作成 (デフォルト)
client = create_llm_client(provider="gemini")

# テキスト生成
response = client.generate_content("RAGとは何ですか？簡潔に説明してください。")
print(response)
```

### 構造化データの生成
```python
from pydantic import BaseModel
from helper_llm import create_llm_client

class QAPair(BaseModel):
    question: str
    answer: str

client = create_llm_client(provider="gemini", default_model="gemini-2.0-flash")

prompt = "Pythonのリスト内包表記について、Q&Aペアを作成してください。"
qa_pair = client.generate_structured(prompt, QAPair)

print(f"Q: {qa_pair.question}")
print(f"A: {qa_pair.answer}")
```

## 7. 依存ライブラリ
- `openai`
- `google-generativeai`
- `pydantic`
- `tiktoken`
- `python-dotenv`

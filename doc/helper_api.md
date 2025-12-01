# helper_api.py - API関連コアユーティリティ

作成日: 2025-11-28 (最終更新: Gemini移行対応)

## 目次

1. [概要](#1-概要)
2. [クラス構成](#2-クラス構成)
3. [UnifiedLLMClient (Gemini 3 対応)](#3-unifiedllmclient-gemini-3-対応)
4. [ConfigManager (設定管理)](#4-configmanager-設定管理)
5. [ユーティリティ機能](#5-ユーティリティ機能)
   - [キャッシュ (MemoryCache)](#キャッシュ-memorycache)
   - [トークン管理 (TokenManager)](#トークン管理-tokenmanager)
   - [デコレータ](#デコレータ)
6. [使用例](#6-使用例)

---

## 1. 概要

`helper_api.py` は、LLM APIを利用するアプリケーションのためのコアユーティリティモジュールです。
当初は OpenAI API 専用のラッパーとして設計されましたが、現在は `helper_llm.py` を統合し、**Google Gemini API** と **OpenAI API** の両方をサポートする抽象化レイヤーを提供します。

また、設定管理、ログ出力、キャッシュ、エラーハンドリングなどの共通機能を提供し、アプリケーション開発の効率化を支援します。

## 2. クラス構成

| クラス名 | 説明 |
|---|---|
| `UnifiedLLMClient` | **(New)** Gemini と OpenAI を切り替え可能な統合クライアント |
| `ConfigManager` | `config.yml` と環境変数を管理するシングルトンクラス |
| `TokenManager` | トークン数のカウントとコスト試算（Gemini/GPT対応） |
| `MemoryCache` | APIレスポンスなどをメモリ上に一時保存するキャッシュ |
| `MessageManager` | チャット履歴の管理 |
| `ResponseProcessor` | APIレスポンスの解析と整形 |
| `OpenAIClient` | (Legacy) OpenAI API を直接利用するクライアント |

## 3. UnifiedLLMClient (Gemini 3 対応)

`UnifiedLLMClient` は、プロバイダー（Gemini / OpenAI）の差異を吸収し、統一されたインターフェースでテキスト生成や構造化出力を利用可能にします。

### 初期化

```python
from helper_api import create_llm_client

# デフォルト（環境変数 LLM_PROVIDER に従う、通常は "gemini"）
client = create_llm_client()

# プロバイダーを明示的に指定
gemini_client = create_llm_client(provider="gemini")
openai_client = create_llm_client(provider="openai")
```

### 主なメソッド

*   **`generate(prompt, model=None, ...)`**: テキスト生成を実行します。
*   **`generate_structured(prompt, response_schema, ...)`**: Pydantic モデルに基づいた JSON 出力を生成します（Gemini の `response_schema`、OpenAI の `response_format` に対応）。
*   **`count_tokens(text)`**: 入力テキストのトークン数をカウントします。

## 4. ConfigManager (設定管理)

`config.yml` ファイルの設定を読み込み、環境変数によるオーバーライドをサポートします。

```python
from helper_api import config

# 設定値の取得
api_key = config.get("api.openai_api_key")
model = config.get("models.default", "gemini-2.0-flash")
```

## 5. ユーティリティ機能

### キャッシュ (MemoryCache)

API 呼び出しの結果などをメモリ上にキャッシュし、再利用します。

```python
from helper_api import cache_result

@cache_result(ttl=3600)
def expensive_operation():
    # ...
```

### トークン管理 (TokenManager)

テキストのトークン数をカウントし、API コストを概算します。`tiktoken` を使用していますが、Gemini モデルの場合は簡易的な推定または `UnifiedLLMClient` 経由のカウントが推奨されます。

### デコレータ

*   `@error_handler`: 例外をキャッチし、ログ出力します。
*   `@timer`: 関数の実行時間を計測し、ログ出力します。

## 6. 使用例

**Gemini で構造化データ生成:**

```python
from helper_api import create_llm_client
from pydantic import BaseModel

class QAPair(BaseModel):
    question: str
    answer: str

client = create_llm_client(provider="gemini")

qa = client.generate_structured(
    prompt="日本の首都についてQ&Aを作って",
    response_schema=QAPairsResponse,
    model="gemini-2.0-flash"
)

print(qa.question) # "日本の首都はどこですか？"
print(qa.answer)   # "東京です。"
```

**OpenAI (互換モード) でテキスト生成:**

```python
client = create_llm_client(provider="openai")
response = client.generate("Hello!", model="gpt-4o-mini")
print(response)
```

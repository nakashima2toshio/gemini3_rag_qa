# helper_embedding.py - Embeddingクライアント抽象化レイヤー

作成日: 2025-11-28 (最終更新: Gemini移行対応)

## 目次

1. [概要](#1-概要)
2. [クラス構成](#2-クラス構成)
3. [EmbeddingClient (抽象基底クラス)](#3-embeddingclient-抽象基底クラス)
4. [GeminiEmbedding (Gemini API実装)](#4-geminiembedding-gemini-api実装)
5. [OpenAIEmbedding (OpenAI API実装)](#5-openaiembedding-openai-api実装)
6. [ファクトリ関数とヘルパー](#6-ファクトリ関数とヘルパー)
7. [使用例](#7-使用例)

---

## 1. 概要

`helper_embedding.py` は、テキストのベクトル化（Embedding）を行うためのクライアントを抽象化するモジュールです。
**Google Gemini API** (`gemini-embedding-001`) と **OpenAI API** (`text-embedding-3-small`) の両方に対応し、統一されたインターフェースで利用できます。

特に **Gemini 3 (Gemini 2.0)** 時代に合わせて、デフォルトで **3072次元** の高精度な Embedding を提供するように設計されています。

## 2. クラス構成

| クラス名 | 説明 | デフォルト次元数 |
|---|---|---|
| `EmbeddingClient` | 全てのEmbeddingクライアントの基底となる抽象クラス | - |
| `GeminiEmbedding` | Gemini API (`gemini-embedding-001`) を利用する実装 | **3072** |
| `OpenAIEmbedding` | OpenAI API (`text-embedding-3-small`) を利用する実装 | 1536 |

## 3. EmbeddingClient (抽象基底クラス)

すべての実装クラスは以下のメソッドを持ちます。

*   **`embed_text(text: str) -> List[float]`**: 単一のテキストをベクトル化します。
*   **`embed_texts(texts: List[str], batch_size: int) -> List[List[float]]`**: 複数のテキストをバッチ処理でベクトル化します。
*   **`dimensions` (property)**: ベクトルの次元数を返します。

## 4. GeminiEmbedding (Gemini API実装)

Google Gemini API を使用します。

*   **モデル**: `gemini-embedding-001`
*   **次元数**: 3072 (Gemini 3の標準)
*   **特徴**: 高精度、Gemini LLMとの高い親和性。バッチ処理は内部でレート制限を考慮しながら実行されます。

## 5. OpenAIEmbedding (OpenAI API実装)

OpenAI API を使用します（互換性維持のため）。

*   **モデル**: `text-embedding-3-small`
*   **次元数**: 1536
*   **特徴**: 従来のRAGシステムとの互換性。

## 6. ファクトリ関数とヘルパー

### `create_embedding_client(provider="gemini", **kwargs)`

プロバイダーを指定してクライアントを生成します。

*   `provider`: "gemini" (デフォルト) または "openai"
*   `**kwargs`: `api_key` や `model` などの追加パラメータ

### `get_embedding_dimensions(provider="gemini")`

プロバイダーごとのデフォルト次元数を取得します。Qdrantコレクション作成時などに便利です。

*   Gemini: 3072
*   OpenAI: 1536

## 7. 使用例

**Gemini で Embedding 生成:**

```python
from helper_embedding import create_embedding_client

# Geminiクライアント作成 (デフォルト: 3072次元)
embedding = create_embedding_client(provider="gemini")

# 単一テキスト
vector = embedding.embed_text("Hello world")
print(f"Dimensions: {len(vector)}")  # 3072

# バッチ処理
texts = ["Hello", "World", "Gemini"]
vectors = embedding.embed_texts(texts, batch_size=100)
```

**Qdrant 用の設定取得:**

```python
from helper_embedding import get_embedding_dimensions

dims = get_embedding_dimensions("gemini")
# Qdrantコレクション作成時に利用
# client.create_collection(..., vector_size=dims, ...)
```

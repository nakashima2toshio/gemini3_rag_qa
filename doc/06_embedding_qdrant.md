# Embedding・Qdrant登録・検索ドキュメント

本ドキュメントでは、Q/AペアデータのEmbedding（ベクトル化）、Qdrantへの登録、および検索処理について解説する。
Geminiモデルと `UnifiedLLMClient` への移行を反映した最新版である。

## 目次

- [1. 概要](#1-概要)
  - [1.1 本ドキュメントの位置づけ](#11-本ドキュメントの位置づけ)
  - [1.2 関連ファイル一覧](#12-関連ファイル一覧)
  - [1.3 データフロー図](#13-データフロー図)
- [2. Embedding（ベクトル化）](#2-embeddingベクトル化)
  - [2.1 使用モデルと設定（Gemini移行済み）](#21-使用モデルと設定gemini移行済み)
  - [2.2 embed_texts_for_qdrant() 関数の処理フロー](#22-embed_texts_for_qdrant-関数の処理フロー)
  - [2.3 バッチ処理とトークン制限](#23-バッチ処理とトークン制限)
  - [2.4 埋め込み入力の構築（question + answer）](#24-埋め込み入力の構築question--answer)
  - [2.5 空文字列・エッジケース処理](#25-空文字列エッジケース処理)
- [3. Qdrant登録](#3-qdrant登録)
  - [3.1 コレクション設計](#31-コレクション設計)
  - [3.2 ベクトルパラメータ設定](#32-ベクトルパラメータ設定)
  - [3.3 ポイント構造（PointStruct）](#33-ポイント構造pointstruct)
  - [3.4 ペイロードスキーマ](#34-ペイロードスキーマ)
  - [3.5 バッチアップサート処理](#35-バッチアップサート処理)
  - [3.6 ペイロードインデックス](#36-ペイロードインデックス)
- [4. コレクション統合](#4-コレクション統合)
  - [4.1 統合機能の概要](#41-統合機能の概要)
  - [4.2 scroll_all_points_with_vectors()](#42-scroll_all_points_with_vectors)
  - [4.3 merge_collections()](#43-merge_collections)
  - [4.4 統合時のペイロード拡張](#44-統合時のペイロード拡張)
- [5. 検索処理](#5-検索処理)
  - [5.1 クエリのベクトル化](#51-クエリのベクトル化)
  - [5.2 コサイン類似度検索](#52-コサイン類似度検索)
  - [5.3 検索結果の構造](#53-検索結果の構造)
  - [5.4 AI応答生成との連携（UnifiedLLMClient利用）](#54-ai応答生成との連携unifiedllmclient利用)
- [6. 運用・設定](#6-運用設定)
  - [6.1 Qdrant設定（QDRANT_CONFIG）](#61-qdrant設定qdrant_config)
  - [6.2 コレクション管理（CRUD）](#62-コレクション管理crud)
  - [6.3 ヘルスチェック](#63-ヘルスチェック)
  - [6.4 統計情報取得](#64-統計情報取得)
- [7. 付録](#7-付録)
  - [7.1 コレクション名とCSVファイルの対応表](#71-コレクション名とcsvファイルの対応表)
  - [7.2 コード参照一覧](#72-コード参照一覧)

---

## 1. 概要

### 1.1 本ドキュメントの位置づけ

本ドキュメントは「ベクトル化・Qdrant登録・検索」に焦点を当てる。
RAGシステム全体がGeminiモデルに移行したことに伴い、Embedding生成およびAI応答生成の両方でGemini APIが使用される。

| ドキュメント | 焦点 | 内容 |
|-------------|------|------|
| `doc/01_chunk.md` | チャンク分割技術 | SemanticCoverage、文分割 |
| `doc/04_prompt.md` | プロンプト設計 | Gemini向けプロンプト、UnifiedLLMClient |
| `doc/05_qa_pair.md` | 実行・処理フロー | 並列処理、Celery、出力、カバレージ |
| `doc/06_embedding_qdrant.md`（本書） | ベクトル化・DB登録・検索 | Embedding、Qdrant、類似度検索、コレクション統合 |

### 1.2 関連ファイル一覧

| ファイル | 役割 |
|---------|------|
| `services/qdrant_service.py` | Qdrant操作サービス層（メイン実装） |
| `helper_embedding.py` | Embedding抽象化レイヤー（GeminiEmbedding利用） |
| `helper_llm.py` | LLM抽象化レイヤー（Gemini/OpenAI共通） |
| `ui/pages/qdrant_registration_page.py` | 登録UI（CSV登録・コレクション統合） |
| `ui/pages/qdrant_search_page.py` | 検索UI |

### 1.3 データフロー図

```
[Q/Aペアデータ]
    │
    │ qa_output/*.csv (question, answer列)
    ▼
[1. データ読み込み]
    │
    │ DataFrame (question, answer)
    ▼
[2. 埋め込み入力構築]
    │
    │ List[str] ("question\nanswer" or "question")
    ▼
[3. Embedding生成]  ←── embed_texts_for_qdrant() (Gemini gemini-embedding-001)
    │
    │ List[List[float]] (3072次元ベクトル)
    ▼
[4. ポイント構築]
    │
    │ List[PointStruct] (id, vector, payload)
    ▼
[5. Qdrant登録]
    │
    │ コレクションにバッチアップサート
    ▼
[Qdrant Vector Database]
    │
    ├──[6a. 検索クエリ]  ←── embed_query_for_search() (Gemini gemini-embedding-001)
    │       │
    │       │ コサイン類似度検索
    │       ▼
    │   [検索結果] (score, question, answer, source)
    │       │
    │       │ (RAG連携)
    │       ▼
    │   [AI応答生成]  ←── UnifiedLLMClient.generate_structured() (Gemini)
    │
    └──[6b. コレクション統合]
            │
            │ 複数コレクションを1つに統合
            ▼
        [統合コレクション]
```

---

## 2. Embedding（ベクトル化）

### 2.1 使用モデルと設定（Gemini移行済み）

本システムでは、Qdrantに登録するQ/AペアのEmbeddingにGeminiの`gemini-embedding-001`モデルを使用する。
これは `helper_embedding.py` を介して利用され、デフォルトで**3072次元**のベクトルを生成する。

```python
# services/qdrant_service.py - COLLECTION_EMBEDDINGS_SEARCH
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_VECTOR_SIZE = 3072  # Gemini 3の高精度設定
```

| 項目 | 値 | 説明 |
|-----|-----|-----|
| モデル | gemini-embedding-001 | Google Gemini埋め込みモデル |
| 次元数 | 3072 | 高精度版（OpenAIの1536次元の2倍） |
| バッチ処理 | 対応 | helper_embedding.py内で処理 |

### 2.2 embed_texts_for_qdrant() 関数の処理フロー

```python
# services/qdrant_service.py:469-531
def embed_texts_for_qdrant(
    texts: List[str],
    model: str = "gemini-embedding-001",
    batch_size: int = 128
) -> List[List[float]]:
```

1.  **空文字列フィルタリング**: 空のテキストを除外。
2.  **Embedding生成**: `create_embedding_client(provider="gemini")` を使用して `embed_texts` を呼び出す。
3.  **ベクトル再配置**: 除外したインデックス位置にダミーベクトル (`[0.0] * 3072`) を挿入し、元のリストと長さを合わせる。

### 2.3 バッチ処理とトークン制限

Gemini APIのレート制限（RPM/TPM）を考慮し、`helper_embedding.py` 内で適切なウェイトタイムを挟みながらバッチ処理を行う。

### 2.4 埋め込み入力の構築（question + answer）

`build_inputs_for_embedding` 関数により、Q/Aペアをベクトル化する際の入力形式を選択できる。
デフォルトでは `question` と `answer` を改行で連結したテキスト (`"question\nanswer"`) が使用され、回答内容も含めた関連性検索が可能。

### 2.5 空文字列・エッジケース処理

空文字列や空白のみのテキストはEmbedding生成対象から除外され、対応する位置には `[0.0] * 3072` のダミーベクトルが配置される。

---

## 3. Qdrant登録

### 3.1 コレクション設計

データセットおよび生成方式ごとにQdrantコレクションを分離する。命名規則は `qa_{dataset}_{method}`。

### 3.2 ベクトルパラメータ設定

コレクション作成時 (`create_or_recreate_collection_for_qdrant` 関数) には、`gemini-embedding-001` に合わせてベクトルサイズ **3072** と距離メトリクス `models.Distance.COSINE` が設定される。

### 3.3 ポイント構造（PointStruct）

Qdrantの各データポイントは一意な `id` (64ビット正整数)、`vector` (3072次元浮動小数点リスト)、および `payload` (JSONオブジェクト) から構成される。

### 3.4 ペイロードスキーマ

`qa:v1` スキーマを基本とし、`domain`, `question`, `answer`, `source`, `created_at`, `schema` フィールドを含む。

### 3.5 バッチアップサート処理

`upsert_points_to_qdrant` 関数は `batched` ユーティリティを使用してポイントリストを128件ずつのバッチに分割し、効率的にQdrantにアップサートする。

### 3.6 ペイロードインデックス

検索効率を向上させるため、コレクション作成時に `domain` フィールドに `KEYWORD` タイプでペイロードインデックスが自動的に作成される。

---

## 4. コレクション統合

複数のQdrantコレクションを1つの新しいコレクションに統合する機能。

### 4.1 統合機能の概要

`merge_collections` 関数を使用して、指定された複数のソースコレクションから全ポイントを取得し、新しい一意のIDを付与し、ペイロードに元のコレクション情報を追加してターゲットコレクションにアップサートする。

### 4.2 scroll_all_points_with_vectors()

コレクションから全ポイントを、そのベクトルデータも含めて取得するための関数。

### 4.3 merge_collections()

ソースコレクションからのデータ取得、ID再生成、ペイロード拡張、そしてターゲットコレクションへのバッチアップサートを一連の処理として実行する。
**統合時のベクトルサイズもデフォルトで3072に設定される。**

### 4.4 統合時のペイロード拡張

統合されたポイントのペイロードには、`_source_collection` と `_original_id` フィールドが追加される。

---

## 5. 検索処理

### 5.1 クエリのベクトル化

検索クエリ (`query: str`) は、`embed_query_for_search` 関数を使用してEmbeddingベクトルに変換される。
この関数も `create_embedding_client(provider="gemini")` を使用する。

### 5.2 コサイン類似度検索

ベクトル化されたクエリを用いてQdrantの `client.search()` メソッドでコサイン類似度に基づいたTop-K検索を実行する。

### 5.3 検索結果の構造

検索結果は、`score` (類似度)、Qdrantの `id`、および `payload` (元の質問、回答、ソースなどのメタデータ) を含むリストとして返される。

### 5.4 AI応答生成との連携（UnifiedLLMClient利用）

検索された関連性の高いQ/Aペア（コンテキスト）を用いて、`UnifiedLLMClient` を介してAIによる最終的な回答を生成する（RAGパターン）。
これは `ui/pages/qdrant_search_page.py` で実装されている。

```python
from helper_llm import create_llm_client, LLMClient

# 検索結果から最適なものを選択
best_hit = hits[0]
# ...

# AI応答生成用のプロンプト構築
qa_prompt = ( ... )

# UnifiedLLMClient を使用してAI応答を生成
llm_client: LLMClient = create_llm_client(provider="gemini")
generated_answer = llm_client.generate_content(
    prompt=qa_prompt,
    model="gemini-2.0-flash",
    temperature=0.7
)
```

---

## 6. 運用・設定

### 6.1 Qdrant設定（QDRANT_CONFIG）

`services/qdrant_service.py` 内の `QDRANT_CONFIG` に、Qdrantホスト、ポート、URLなどの基本設定が定義されている。

### 6.2 コレクション管理（CRUD）

Qdrantクライアントを通じて、コレクションの作成、取得、統計情報の参照、および削除が可能。

### 6.3 ヘルスチェック

`QdrantHealthChecker` クラスは、ポートの開放状況とQdrant APIへの接続性をチェックする。

### 6.4 統計情報取得

`QdrantDataFetcher` クラスは、コレクション一覧、詳細ポイントデータ、コレクション情報を提供する。

---

## 7. 付録

### 7.1 コレクション名とCSVファイルの対応表

`services/qdrant_service.py` 内の `COLLECTION_CSV_MAPPING` に定義されている。

### 7.2 コード参照一覧

| 機能 | ファイル | 関数/クラス |
|-----|---------|------------|
| Qdrant設定 | services/qdrant_service.py | QDRANT_CONFIG |
| バッチ分割 | services/qdrant_service.py | batched() |
| ヘルスチェック | services/qdrant_service.py | QdrantHealthChecker |
| データ取得 | services/qdrant_service.py | QdrantDataFetcher |
| CSV読み込み | services/qdrant_service.py | load_csv_for_qdrant() |
| 埋め込み入力構築 | services/qdrant_service.py | build_inputs_for_embedding() |
| 埋め込み生成 | services/qdrant_service.py | embed_texts_for_qdrant() |
| コレクション作成 | services/qdrant_service.py | create_or_recreate_collection_for_qdrant() |
| 検索クエリベクトル化 | services/qdrant_service.py | embed_query_for_search() |
| 登録UI | ui/pages/qdrant_registration_page.py | show_qdrant_registration_page() |
| 検索UI | ui/pages/qdrant_search_page.py | show_qdrant_search_page() |
# a50_rag_search_local_qdrant.py - 技術仕様書

## 目次

1. [概要](#1-概要)
2. [アーキテクチャ](#2-アーキテクチャ)
3. [対応コレクション](#3-対応コレクション)
4. [UI構成](#4-ui構成)
5. [検索機能](#5-検索機能)
6. [OpenAI統合](#6-openai統合)
7. [使用方法](#7-使用方法)
8. [設定ファイル](#8-設定ファイル)
9. [トラブルシューティング](#9-トラブルシューティング)

---

## 1. 概要

### 1.1 目的

`a50_rag_search_local_qdrant.py`は、ローカルまたはリモートのQdrantベクトルデータベースに対して、自然言語クエリを使用したセマンティック検索を実行するStreamlit UIアプリケーションです。

### 1.2 起動コマンド

```bash
streamlit run a50_rag_search_local_qdrant.py --server.port=8504
```

### 1.3 主要機能

- **複数コレクション対応**: 10種類のコレクションを動的に取得・選択
- **ドメイン別検索**: qa_corpusで5ドメインをサポート
- **Named Vectors切替**: ada-002、3-small等の切替
- **動的埋め込み次元対応**: 384次元（高速）、1536次元（高精度）
- **リアルタイム類似度スコア表示**: スコア閾値の目安表示
- **OpenAI GPT-4o-mini統合**: 検索結果から日本語回答を生成
- **多言語対応**: 日英間のクロスリンガル検索

### 1.4 入出力

| 種別 | データソース | 形式 |
|------|------------|------|
| INPUT | ユーザークエリ | テキスト |
| INPUT | Qdrant Vector Database | REST API |
| OUTPUT | Streamlit WebUI | HTML/CSS |
| OUTPUT | 検索結果 + GPT回答 | DataFrame + テキスト |

---

## 2. アーキテクチャ

### 2.1 システム構成図

```
┌─────────────────────────────────────────────────────────────────┐
│              a50_rag_search_local_qdrant.py (466行)             │
├─────────────────────────────────────────────────────────────────┤
│  設定・定数                                                      │
│  ├── DEFAULTS (49-57) - デフォルト設定                          │
│  ├── COLLECTION_EMBEDDINGS (60-71) - 10コレクション定義          │
│  └── SAMPLE_QUESTIONS (137-179) - 7ドメイン質問例                │
├─────────────────────────────────────────────────────────────────┤
│  設定・ヘルパー                                                  │
│  ├── load_config (73-94) - config.yml読み込み                   │
│  └── embed_query (96-113) - クエリ埋め込み生成                  │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit UI                                                   │
│  ├── サイドバー (181-284) - 設定パネル                           │
│  │   ├── Collection セレクタ                                    │
│  │   ├── Using vector (named) セレクタ                          │
│  │   ├── Domain セレクタ (qa_corpusのみ)                        │
│  │   ├── TopK スライダー                                        │
│  │   └── サンプル質問ボタン                                      │
│  ├── メインエリア (290-405) - 検索と結果表示                     │
│  │   ├── クエリ入力フォーム                                      │
│  │   ├── 検索結果DataFrame                                      │
│  │   ├── 最高スコア結果表示                                      │
│  │   └── OpenAI応答（日本語）                                    │
│  └── エラーハンドリング (449-465) - 接続・コレクションエラー     │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 依存モジュール

```python
import os
from typing import Dict, Any, List, Optional
import pandas as pd
import streamlit as st
import yaml  # オプション

from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
```

### 2.3 データフロー

```
ユーザークエリ入力
       │
       ▼
embed_query() - OpenAI Embedding生成
       │
       ▼
QdrantClient.search() - ベクトル類似検索
       │
       ▼
検索結果DataFrame表示
       │
       ▼
OpenAI Responses API (gpt-4o-mini)
       │
       ▼
日本語回答生成・表示
```

---

## 3. 対応コレクション

### 3.1 COLLECTION_EMBEDDINGS（全10コレクション）

| コレクション名 | 埋め込みモデル | 次元数 | 説明 |
|--------------|--------------|-------|------|
| product_embeddings | text-embedding-3-small | 384 | 製品情報用（高速処理） |
| qa_corpus | text-embedding-3-small | 1536 | Q&Aコーパス（5ドメイン） |
| raw_cc_news | text-embedding-3-small | 1536 | CC News生データ |
| raw_livedoor | text-embedding-3-small | 1536 | Livedoor生データ |
| qa_cc_news_a02_llm | text-embedding-3-small | 1536 | CC News LLM生成方式 |
| qa_cc_news_a03_rule | text-embedding-3-small | 1536 | CC News ルールベース方式 |
| qa_cc_news_a10_hybrid | text-embedding-3-small | 1536 | CC News ハイブリッド方式 |
| qa_livedoor_a02_20_llm | text-embedding-3-small | 1536 | Livedoor LLM生成方式 |
| qa_livedoor_a03_rule | text-embedding-3-small | 1536 | Livedoor ルールベース方式 |
| qa_livedoor_a10_hybrid | text-embedding-3-small | 1536 | Livedoor ハイブリッド方式 |

### 3.2 ドメイン（qa_corpusのみ）

| ドメイン | 説明 | 言語 |
|---------|------|------|
| customer | カスタマーサポート・FAQ | 日本語 |
| medical | 医療QAデータ | 日本語 |
| legal | 法律・判例QA | 日本語 |
| sciq | 科学・技術QA | 日本語 |
| trivia | TriviaQA（トリビアQA） | 日本語 |

---

## 4. UI構成

### 4.1 サイドバー設定

| 設定項目 | 説明 | デフォルト |
|---------|------|----------|
| Collection | 検索対象コレクション | product_embeddings |
| Using vector (named) | 埋め込みモデル設定 | primary |
| Domain | ドメインフィルタ（qa_corpusのみ） | ALL |
| TopK | 取得する検索結果数 | 5（1-20） |
| Qdrant URL | サーバーアドレス | http://localhost:6333 |
| Debug Mode | デバッグ情報表示 | OFF |

### 4.2 サンプル質問

#### CC Newsコレクション（英語）

```python
SAMPLE_QUESTIONS["cc_news"] = [
    "Which Boston Ballet dancers starred in their Super Bowl video?",
    "What role does The Nutcracker play for ballet companies beyond being a performance piece?",
    "What insight did the speaker gain from seeing Robert's McLaren documentary?",
    "Under what circumstance can a firm still charge a fee for an SAR under GDPR?",
    "Which two technologies did Vyas use to illustrate a 5G use case?"
]
```

#### Livedoorコレクション（日本語）

```python
SAMPLE_QUESTIONS["livedoor"] = [
    "ライブドアニュースについて教えてください",
    "最新のテクノロジーニュースは？",
    "スポーツニュースで話題になっていることは？",
    "エンタメ関連のニュースを知りたい",
    "経済ニュースの最新情報は？"
]
```

### 4.3 メイン画面

| 要素 | 説明 |
|------|------|
| 機能説明 | コレクション・ドメインの説明、スコア目安 |
| クエリ入力 | テキストボックス |
| Searchボタン | 検索実行 |
| Results | 検索結果DataFrame |
| Highest Score Result | 最高スコアの詳細表示 |
| OpenAI 応答（日本語） | GPT-4o-miniによる回答 |

---

## 5. 検索機能

### 5.1 embed_query関数 (96-113)

```python
def embed_query(text: str, model: str, dims: Optional[int] = None) -> List[float]:
    """クエリテキストを埋め込みベクトルに変換

    Args:
        text: 埋め込むテキスト
        model: 使用する埋め込みモデル
        dims: ベクトルの次元数（text-embedding-3系のみ有効）

    Returns:
        埋め込みベクトル
    """
```

### 5.2 ベクトル検索

```python
hits = client.search(
    collection_name=collection,
    query_vector=qvec,
    limit=topk,
    query_filter=qfilter  # ドメインフィルタ（qa_corpusのみ）
)
```

### 5.3 フィールドマッピング

検索結果から以下のフィールドを抽出（複数のフィールド名に対応）:

| フィールド | 対応フィールド名 |
|-----------|----------------|
| score | 類似度スコア |
| domain | domain |
| question | question, text, content |
| answer | answer, response, metadata |
| source | source, file |

### 5.4 類似度スコア目安

| スコア範囲 | 意味 |
|-----------|------|
| 0.8以上 | 非常に関連性が高い（ほぼ一致） |
| 0.6-0.8 | 関連性がある（有用な結果） |
| 0.4-0.6 | 部分的に関連（参考程度） |
| 0.4未満 | 関連性が低い（フィルタリング推奨） |

---

## 6. OpenAI統合

### 6.1 日本語回答生成

```python
qa_prompt_jp = (
    "以下の検索結果（スコア・質問・回答）とユーザーの元の質問を踏まえて、"
    "日本語で簡潔かつ正確に回答してください。必要に応じて箇条書きを用いてください。\n\n"
    f"ユーザーの元の質問（query）:\n{query}\n\n"
    f"検索結果のスコア: {br_score:.4f}\n"
    f"検索結果の質問: {br_q}\n"
    f"検索結果の回答: {br_a}\n"
)

oai_client = OpenAI()
oai_resp = oai_client.responses.create(
    model="gpt-4o-mini",
    input=qa_prompt_jp
)
generated_answer = getattr(oai_resp, "output_text", None) or ""
```

### 6.2 多言語対応

- OpenAI埋め込みモデルが多言語対応
- 日英間でのクロスリンガル検索が可能
- 例: 日本語「返金は可能ですか？」と英語「Can I get a refund?」の高い類似度（0.4957）

---

## 7. 使用方法

### 7.1 前提条件

```bash
# 1. OpenAI APIキーの設定
export OPENAI_API_KEY="sk-..."

# 2. Qdrantサーバーの起動
cd docker-compose
docker-compose up -d qdrant

# 3. データ登録（未実行の場合）
python a42_qdrant_registration.py --recreate --include-answer
```

### 7.2 起動

```bash
# デフォルトポートで起動
streamlit run a50_rag_search_local_qdrant.py

# カスタムポートで起動
streamlit run a50_rag_search_local_qdrant.py --server.port=8504
```

### 7.3 基本操作

1. **コレクション選択**: サイドバーでコレクションを選択
2. **ドメイン選択**（qa_corpusのみ）: customer, medical等を選択
3. **クエリ入力**: テキストボックスにクエリを入力
4. **Search実行**: ボタンをクリック
5. **結果確認**: DataFrame + 最高スコア結果 + GPT回答を確認

### 7.4 サンプル質問の使用

サイドバーの「質問例」セクションでボタンをクリックすると、自動的にクエリ入力フィールドに設定されます。

---

## 8. 設定ファイル

### 8.1 config.yml

```yaml
rag:
  collection: "product_embeddings"  # デフォルトコレクション

embeddings:
  primary:
    provider: "openai"
    model: "text-embedding-3-small"
    dims: 1536
  ada-002:
    provider: "openai"
    model: "text-embedding-ada-002"
    dims: 1536
  3-small:
    provider: "openai"
    model: "text-embedding-3-small"
    dims: 1536

qdrant:
  url: "http://localhost:6333"
```

### 8.2 デフォルト設定

```python
DEFAULTS = {
    "rag": {"collection": "product_embeddings"},
    "embeddings": {
        "primary": {"provider": "openai", "model": "text-embedding-3-small", "dims": 1536},
        "ada-002": {"provider": "openai", "model": "text-embedding-ada-002", "dims": 1536},
        "3-small": {"provider": "openai", "model": "text-embedding-3-small", "dims": 1536},
    },
    "qdrant": {"url": "http://localhost:6333"},
}
```

---

## 9. トラブルシューティング

### 9.1 Qdrantサーバーに接続できない

**症状**:
```
❌ Qdrantサーバーに接続できません: http://localhost:6333
```

**解決方法**:
```bash
# Qdrantサーバーを起動
cd docker-compose
docker-compose up -d qdrant
```

### 9.2 コレクションが見つからない

**症状**:
```
❌ コレクション 'qa_cc_news_a02_llm' が見つかりません
```

**解決方法**:
```bash
# データを登録
python a42_qdrant_registration.py --recreate --include-answer
```

### 9.3 埋め込み生成エラー

**症状**:
```
❌ Embedding generation failed
```

**解決方法**:
- OpenAI APIキーを確認
- デバッグモードを有効にして詳細を確認
- モデル名と次元数が一致しているか確認

### 9.4 OpenAI応答生成エラー

**症状**:
```
OpenAI応答生成に失敗しました
```

**解決方法**:
- OpenAI APIキーを確認
- APIのレート制限を確認
- ネットワーク接続を確認

---

## 付録: メタデータ

| 項目 | 値 |
|------|-----|
| ファイル行数 | 466行 |
| ポート | 8504（デフォルト） |
| Qdrant URL | http://localhost:6333 |
| 埋め込みモデル | text-embedding-3-small |
| 回答生成モデル | gpt-4o-mini |
| コレクション数 | 10 |
| サンプル質問ドメイン数 | 7 |

## 関連ファイル

| ファイル | 役割 |
|---------|------|
| a40_show_qdrant_data.py | Qdrantデータ表示UI |
| a41_qdrant_truncate.py | Qdrantデータ削除 |
| a42_qdrant_registration.py | Qdrantへのデータ登録 |
| a34_rag_search_cloud_vs.py | OpenAI Vector Store検索UI |
| config.yml | 設定ファイル |
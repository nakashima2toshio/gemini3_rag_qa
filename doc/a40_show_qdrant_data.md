# a40_show_qdrant_data.py - 技術仕様書

## 目次

1. [概要](#1-概要)
2. [アーキテクチャ](#2-アーキテクチャ)
3. [QdrantHealthChecker](#3-qdranthealthchecker)
4. [QdrantDataFetcher](#4-qdrantdatafetcher)
5. [UI構成](#5-ui構成)
6. [使用方法](#6-使用方法)
7. [エクスポート機能](#7-エクスポート機能)
8. [トラブルシューティング](#8-トラブルシューティング)

---

## 1. 概要

### 1.1 目的

`a40_show_qdrant_data.py`は、Qdrantベクトルデータベースの状態を監視し、登録されているデータを視覚的に表示するためのStreamlitベースのWebアプリケーションです。

### 1.2 起動コマンド

```bash
streamlit run a40_show_qdrant_data.py --server.port=8502
```

### 1.3 主要機能

- **Qdrantサーバー接続状態チェック**: ポートチェック + API接続確認
- **コレクション一覧表示**: 全コレクションの統計情報（ベクトル数、ポイント数、ステータス）
- **データソース情報表示**: qa_output/ディレクトリーとの対応関係を自動分析
- **ポイントデータ表示**: コレクション内のデータをテーブル形式で表示
- **エクスポート機能**: CSV/JSON形式でのデータダウンロード
- **デバッグモード**: 詳細なエラー情報とサーバー設定表示

### 1.4 入出力

| 種別 | データソース | 形式 |
|------|------------|------|
| INPUT | Qdrant Vector Database (localhost:6333) | REST API |
| OUTPUT | Streamlit WebUI | HTML/CSS |
| OUTPUT | CSV/JSON エクスポート | ファイルダウンロード |

---

## 2. アーキテクチャ

### 2.1 システム構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                    a40_show_qdrant_data.py                      │
├─────────────────────────────────────────────────────────────────┤
│  QDRANT_CONFIG (48-56)                                          │
│  ├── name: "Qdrant"                                             │
│  ├── host: "localhost"                                          │
│  ├── port: 6333                                                 │
│  ├── url: "http://localhost:6333"                               │
│  └── docker_image: "qdrant/qdrant"                              │
├─────────────────────────────────────────────────────────────────┤
│  QdrantHealthChecker (61-110)                                   │
│  ├── check_port() - ポート開放チェック                           │
│  └── check_qdrant() - Qdrant接続チェック                         │
├─────────────────────────────────────────────────────────────────┤
│  QdrantDataFetcher (115-282)                                    │
│  ├── fetch_collections() - コレクション一覧取得                   │
│  ├── fetch_collection_points() - ポイントデータ取得               │
│  ├── fetch_collection_info() - コレクション詳細情報取得            │
│  └── fetch_collection_source_info() - データソース情報取得        │
├─────────────────────────────────────────────────────────────────┤
│  display_source_info() (287-328)                                │
│  └── データソース情報をStreamlit UIで表示                         │
├─────────────────────────────────────────────────────────────────┤
│  main() (333-666)                                               │
│  ├── サイドバー（接続状態、デバッグモード、自動更新）               │
│  ├── コレクション一覧表示                                        │
│  ├── データソース情報表示                                        │
│  ├── コレクション詳細データ表示                                   │
│  └── エクスポート機能（CSV/JSON）                                │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 依存モジュール

```python
import streamlit as st
import pandas as pd
import time
import logging
import socket
from qdrant_client import QdrantClient
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
```

### 2.3 サーバー設定

```python
QDRANT_CONFIG = {
    "name": "Qdrant",
    "host": "localhost",
    "port": 6333,
    "icon": "🎯",
    "url": "http://localhost:6333",
    "health_check_endpoint": "/collections",
    "docker_image": "qdrant/qdrant"
}
```

---

## 3. QdrantHealthChecker

### 3.1 概要

Qdrantサーバーへの接続状態をチェックするクラス。

### 3.2 check_port (68-79)

```python
def check_port(self, host: str, port: int, timeout: float = 2.0) -> bool:
    """ポートが開いているかチェック

    - socket接続でポート開放を確認
    - タイムアウト: 2秒（デフォルト）
    """
```

### 3.3 check_qdrant (81-110)

```python
def check_qdrant(self) -> Tuple[bool, str, Optional[Dict]]:
    """Qdrant接続チェック

    処理フロー:
    1. ポートチェック（check_port）
    2. QdrantClient接続
    3. コレクション一覧取得
    4. レスポンスタイム計測

    Returns:
        (is_connected, message, metrics)
        - is_connected: 接続成功/失敗
        - message: ステータスメッセージ
        - metrics: {collection_count, collections, response_time_ms}
    """
```

---

## 4. QdrantDataFetcher

### 4.1 概要

Qdrantからデータを取得するクラス。

### 4.2 fetch_collections (121-149)

```python
def fetch_collections(self) -> pd.DataFrame:
    """コレクション一覧を取得

    Returns:
        DataFrame with columns:
        - Collection: コレクション名
        - Vectors Count: ベクトル総数
        - Points Count: ポイント総数
        - Indexed Vectors: インデックス済みベクトル数
        - Status: コレクションの状態
    """
```

### 4.3 fetch_collection_points (151-188)

```python
def fetch_collection_points(
    self, collection_name: str, limit: int = 50
) -> pd.DataFrame:
    """コレクションの詳細データを取得

    - scrollメソッドでポイントを取得
    - payloadの各フィールドを列として追加
    - 長すぎる文字列は200文字で切り詰め
    - ベクトルデータは含まない（with_vectors=False）
    """
```

### 4.4 fetch_collection_info (190-227)

```python
def fetch_collection_info(self, collection_name: str) -> Dict[str, Any]:
    """コレクションの詳細情報を取得

    Returns:
        {
            "vectors_count": int,
            "points_count": int,
            "indexed_vectors": int,
            "status": str,
            "config": {
                "vector_size": int,
                "distance": str
            }
        }
    """
```

### 4.5 fetch_collection_source_info (229-282)

```python
def fetch_collection_source_info(
    self, collection_name: str, sample_size: int = 200
) -> Dict[str, Any]:
    """コレクションのデータソース情報を取得

    - サンプリング（デフォルト200件）から推定
    - source, generation_method, domainを集計
    - 全体のデータ数を推定

    Returns:
        {
            "total_points": int,
            "sources": {
                "source_file.csv": {
                    "sample_count": int,
                    "estimated_total": int,
                    "percentage": float,
                    "method": str,
                    "domain": str
                }
            },
            "sample_size": int
        }
    """
```

---

## 5. UI構成

### 5.1 サイドバー（左ペイン）

| 機能 | 説明 |
|------|------|
| **⚙️ Qdrant接続状態** | Qdrantサーバーへの接続状態を表示 |
| **🐛 デバッグモード** | 詳細なエラー情報とサーバー設定を表示 |
| **🔄 自動更新** | 指定間隔で自動的に接続状態を更新（5〜300秒） |
| **🔍 接続チェック実行** | 手動で接続状態を再確認 |

### 5.2 メインエリア（右ペイン）

#### 📚 コレクション一覧

| カラム | 説明 |
|--------|------|
| Collection | コレクション名 |
| Vectors Count | ベクトル総数 |
| Points Count | ポイント総数 |
| Indexed Vectors | インデックス済みベクトル数 |
| Status | コレクションの状態（green/yellow/red） |

#### 📂 データソース情報

各コレクションについて、以下の情報を表示:

| 項目 | 説明 |
|------|------|
| ソースファイル | qa_output/ディレクトリー内のCSVファイル名 |
| 推定件数 | サンプリングに基づく推定データ件数 |
| 割合 | 全体に占める割合（%） |
| 生成方法 | a02_make_qa, a03_coverage, a10_hybrid等 |
| ドメイン | cc_news, livedoor等 |

#### 🔍 コレクション詳細データ

| 機能 | 説明 |
|------|------|
| 表示件数 | 取得するポイント数を指定（1〜500件） |
| 📊 詳細情報を表示 | ベクトル設定、ステータスメトリクスを表示 |
| 🔍 ポイントデータを取得 | 実際のポイントデータをテーブル表示 |
| 📥 CSVダウンロード | ポイントデータをCSV形式で保存 |
| 📥 JSONダウンロード | ポイントデータをJSON形式で保存 |

### 5.3 セッション状態

```python
session_state = {
    'debug_mode': False,        # デバッグモード
    'auto_refresh': False,      # 自動更新
    'refresh_interval': 30      # 更新間隔（秒）
}
```

### 5.4 画面構成図

```
┌─────────────────────────────────────────────────────────────────┐
│ 🎯 Qdrant データ表示ツール                                      │
│ Qdrant Vector Database の状態監視とデータ表示                   │
├──────────────┬──────────────────────────────────────────────────┤
│ サイドバー   │ メインエリア                                     │
├──────────────┤                                                  │
│ ⚙️ 接続状態  │ 📊 Qdrant データ表示                             │
│              │                                                  │
│ □ 🐛 デバッグ │ 📚 コレクション一覧                              │
│ □ 🔄 自動更新 │ ┌────────────────────────────────────────────┐ │
│              │ │ Collection    | Vectors | Points | Status │ │
│ [🔍 接続チェ │ │ qa_cc_news_.. | 5,042   | 5,042  | green  │ │
│  ック実行]   │ │ qa_livedoor.. | 1,845   | 1,845  | green  │ │
│              │ └────────────────────────────────────────────┘ │
│ ✅ Qdrant    │                                                  │
│ ✅ Connected │ [📥 CSV] [📥 JSON]                               │
│              │                                                  │
│              │ ──────────────────────────────────────────────── │
│              │ 📂 コレクションのデータソース情報                 │
│              │                                                  │
│              │ ▼ 📦 qa_cc_news_a02_llm                          │
│              │   ソースファイル | 推定件数 | 割合 | 生成方法  │ │
│              │   a02_qa_pairs.. | 5,042    | 100% | a02_make  │ │
└──────────────┴──────────────────────────────────────────────────┘
```

---

## 6. 使用方法

### 6.1 前提条件

1. **Qdrantサーバーの起動**

```bash
# 方法1: Docker Composeを使用（推奨）
cd docker-compose
docker-compose up -d qdrant

# 方法2: 単独のDockerコマンド
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 方法3: 自動セットアップスクリプト
python server.py
```

2. **依存ライブラリのインストール**

```bash
pip install streamlit qdrant-client pandas
```

### 6.2 基本操作

1. **アプリ起動**
```bash
streamlit run a40_show_qdrant_data.py --server.port=8502
```

2. **ブラウザでアクセス**
```
http://localhost:8502
```

3. **接続チェック**
   - サイドバーの「🔍 接続チェック実行」ボタンをクリック
   - 接続状態（✅ Connected / ❌ Connection refused）を確認

4. **コレクション確認**
   - メインエリアでコレクション一覧を確認
   - 各コレクションのデータソース情報を展開

5. **詳細データ取得**
   - コレクションを選択
   - 表示件数を設定（1〜500件）
   - 「🔍 ポイントデータを取得」ボタンをクリック

---

## 7. エクスポート機能

### 7.1 コレクション一覧エクスポート

**CSV形式**:
```csv
Collection,Vectors Count,Points Count,Indexed Vectors,Status
qa_cc_news_a02_llm,5042,5042,5042,green
qa_livedoor_a02_llm,1845,1845,1845,green
```

**JSON形式**:
```json
[
  {
    "Collection": "qa_cc_news_a02_llm",
    "Vectors Count": 5042,
    "Points Count": 5042,
    "Indexed Vectors": 5042,
    "Status": "green"
  }
]
```

### 7.2 ポイントデータエクスポート

**CSV形式**:
```csv
ID,question,answer,domain,generation_method,source,created_at,schema
123456789,"質問文","回答文","cc_news","a02_make_qa","a02_qa_pairs_cc_news.csv","2025-11-01T12:00:00Z","qa:v1"
```

**JSON形式**:
```json
[
  {
    "ID": 123456789,
    "question": "質問文",
    "answer": "回答文",
    "domain": "cc_news",
    "generation_method": "a02_make_qa",
    "source": "a02_qa_pairs_cc_news.csv",
    "created_at": "2025-11-01T12:00:00Z",
    "schema": "qa:v1"
  }
]
```

### 7.3 ファイル名形式

```
qdrant_collections_YYYYMMDD_HHMMSS.csv
qdrant_collections_YYYYMMDD_HHMMSS.json
{collection_name}_points_YYYYMMDD_HHMMSS.csv
{collection_name}_points_YYYYMMDD_HHMMSS.json
```

---

## 8. トラブルシューティング

### 8.1 Qdrantサーバーに接続できない

**症状**:
```
❌ Qdrantサーバーに接続できません
Connection refused
```

**解決方法**:

**方法1: 自動セットアップ（推奨）**
```bash
python server.py
```

**方法2: 手動でDocker起動**
```bash
# ステップ1: Docker Desktopを起動
# macOS: アプリケーションフォルダから起動

# ステップ2: Qdrantを起動
cd docker-compose
docker-compose up -d qdrant

# ステップ3: 動作確認
docker-compose ps
```

**方法3: トラブルシューティング**
```bash
# ポート使用状況を確認
lsof -i :6333

# ログを確認
docker-compose logs qdrant

# 再起動
docker-compose restart qdrant
```

### 8.2 コレクションが空/見つからない

**症状**:
```
📂 データソース情報が見つかりません
```

**解決方法**:
```bash
# データを登録
python a42_qdrant_registration.py --recreate --include-answer
```

### 8.3 タイムアウトエラー

**症状**:
```
⏱️ Qdrantサーバーへの接続がタイムアウトしました
```

**解決方法**:
- Qdrantサーバーのログを確認
- ファイアウォール設定を確認
- ポート6333が使用可能か確認

### 8.4 データが正しく表示されない

**解決方法**:
```bash
# データの再登録
python a42_qdrant_registration.py --collection qa_cc_news_a02_llm --recreate

# コレクション情報の確認
python a41_qdrant_truncate.py --stats
```

---

## 付録: メタデータ

| 項目 | 値 |
|------|-----|
| ファイル行数 | 668行 |
| ポート | 8502（デフォルト） |
| Qdrantポート | 6333 |
| デフォルトサンプルサイズ | 200件 |
| デフォルト表示件数 | 50件 |
| 最大表示件数 | 500件 |
| 自動更新間隔 | 5〜300秒 |

## 関連ファイル

| ファイル | 役割 |
|---------|------|
| a41_qdrant_truncate.py | Qdrantデータ削除 |
| a42_qdrant_registration.py | Qdrantへのデータ登録 |
| a50_rag_search_local_qdrant.py | Qdrant検索UI |
| server.py | Qdrantサーバー管理 |
| docker-compose/docker-compose.yml | Docker設定 |
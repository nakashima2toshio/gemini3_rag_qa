# a41_qdrant_truncate.py - 技術仕様書

## 目次

1. [概要](#1-概要)
2. [アーキテクチャ](#2-アーキテクチャ)
3. [削除機能](#3-削除機能)
4. [統計情報表示](#4-統計情報表示)
5. [コマンドラインオプション](#5-コマンドラインオプション)
6. [使用方法](#6-使用方法)
7. [安全機能](#7-安全機能)
8. [トラブルシューティング](#8-トラブルシューティング)

---

## 1. 概要

### 1.1 目的

`a41_qdrant_truncate.py`は、Qdrantベクトルデータベースに登録されたRAGデータを安全に削除するための管理ツールです。コレクション全体の削除、特定ドメインのみの削除、統計情報の表示など、柔軟なデータ管理機能を提供します。

### 1.2 実行コマンド

```bash
# 統計情報を表示（削除なし）
python a41_qdrant_truncate.py --stats

# 全コレクションを削除（危険！）
python a41_qdrant_truncate.py --all-collections --force
```

### 1.3 主要機能

- **段階的削除**: データのみ削除 → コレクション削除 → 全コレクション削除
- **安全機能**: 確認プロンプト、ドライラン、除外リスト、カウントダウン
- **詳細統計**: ドメイン別データ数、ベクトル設定、視覚的なグラフ表示
- **バッチ処理**: 大量データの効率的な削除（デフォルト100件ずつ）
- **カラー出力**: 視認性の高いコンソール出力

### 1.4 入出力

| 種別 | データソース | 形式 |
|------|------------|------|
| INPUT | config.yml（オプション） | YAML |
| INPUT | Qdrant Vector Database | REST API |
| OUTPUT | 標準出力（コンソール） | カラーテキスト |
| OUTPUT | Qdrantベクトルデータベース | 削除操作 |

---

## 2. アーキテクチャ

### 2.1 システム構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                    a41_qdrant_truncate.py (731行)               │
├─────────────────────────────────────────────────────────────────┤
│  Colors クラス (64-73)                                          │
│  └── ANSIカラーコード定義                                        │
├─────────────────────────────────────────────────────────────────┤
│  ヘルパー関数                                                    │
│  ├── print_colored (75-77) - カラー出力                         │
│  ├── print_header (79-85) - ヘッダー出力                        │
│  └── load_config (110-131) - 設定ファイル読み込み               │
├─────────────────────────────────────────────────────────────────┤
│  統計・表示関数                                                  │
│  ├── get_collection_stats (133-193) - コレクション統計取得       │
│  ├── display_stats (195-216) - 統計情報表示                     │
│  ├── get_all_collections (369-390) - 全コレクション情報取得      │
│  └── display_all_collections_stats (392-411) - 全コレクション統計表示│
├─────────────────────────────────────────────────────────────────┤
│  確認関数                                                        │
│  ├── confirm_action (218-232) - 単一確認プロンプト               │
│  └── confirm_all_collections_deletion (413-462) - 2段階確認      │
├─────────────────────────────────────────────────────────────────┤
│  削除関数                                                        │
│  ├── delete_by_domain (234-297) - ドメイン別削除                │
│  ├── delete_all_data (299-350) - 全データ削除                   │
│  ├── drop_collection (352-367) - コレクション削除               │
│  └── delete_all_collections (464-517) - 全コレクション削除       │
├─────────────────────────────────────────────────────────────────┤
│  main (519-728) - メインエントリポイント                         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 依存モジュール

```python
import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Any
import yaml  # オプション

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
```

### 2.3 デフォルト設定

```python
DEFAULTS = {
    "rag": {
        "collection": "qa_corpus",
    },
    "qdrant": {
        "url": "http://localhost:6333"
    }
}
```

### 2.4 サポートされるドメイン

```python
SUPPORTED_DOMAINS = [
    "customer",          # カスタマーサポート
    "medical",           # 医療
    "legal",             # 法律
    "sciq",              # 科学
    "trivia",            # トリビア
    "unified",           # 統合データ
    "cc_news_llm",       # CC News LLM生成方式
    "cc_news_coverage",  # CC Newsカバレッジ改良方式
    "cc_news_hybrid"     # CC Newsハイブリッド生成方式
]
```

---

## 3. 削除機能

### 3.1 delete_by_domain (234-297)

```python
def delete_by_domain(
    client: QdrantClient,
    collection_name: str,
    domain: str,
    batch_size: int = 100,
    dry_run: bool = False
) -> int:
    """特定ドメインのデータを削除

    処理フロー:
    1. 対象ドメインのデータ数をカウント
    2. バッチサイズでポイントを検索
    3. ポイントIDを取得して削除
    4. 削除進捗を表示

    Returns:
        削除した件数
    """
```

### 3.2 delete_all_data (299-350)

```python
def delete_all_data(
    client: QdrantClient,
    collection_name: str,
    batch_size: int = 100,
    dry_run: bool = False
) -> int:
    """全データを削除（コレクションは保持）

    用途:
    - コレクション再登録の前処理
    - データのリフレッシュ
    - テスト環境のクリーンアップ
    """
```

### 3.3 drop_collection (352-367)

```python
def drop_collection(
    client: QdrantClient,
    collection_name: str,
    dry_run: bool = False
) -> bool:
    """コレクション自体を削除

    用途:
    - 不要なコレクションの完全削除
    - ベクトル設定の変更時の再作成
    """
```

### 3.4 delete_all_collections (464-517)

```python
def delete_all_collections(
    client: QdrantClient,
    excluded: List[str] = None,
    dry_run: bool = False
) -> int:
    """全コレクションを削除

    - 2段階確認（--forceでスキップ可能）
    - 除外リスト対応
    - 順次削除で進捗表示
    """
```

---

## 4. 統計情報表示

### 4.1 get_collection_stats (133-193)

```python
def get_collection_stats(
    client: QdrantClient,
    collection_name: str
) -> Optional[Dict[str, Any]]:
    """コレクションの統計情報を取得

    Returns:
        {
            "total_points": int,
            "domain_stats": {domain: count, ...},
            "vector_config": {name: {size, distance}, ...},
            "status": str
        }
    """
```

### 4.2 display_stats (195-216)

統計情報を視覚的に表示:

```
================================================================================
 📊 コレクション 'qa_cc_news_a02_llm' の統計情報
================================================================================

総ポイント数:        5,042
ステータス:           green

ドメイン別データ数:
----------------------------------------
  cc_news           5,042 ██████████████████████████████
----------------------------------------

ベクトル設定:
  primary: size=1536, distance=Distance.COSINE
```

### 4.3 display_all_collections_stats (392-411)

全コレクションの統計情報を表示:

```
================================================================================
 📊 全コレクションの統計情報
================================================================================

総コレクション数:     8
総ポイント数:        15,234

コレクション一覧:
------------------------------------------------------------
  名前                            ポイント数   ステータス
------------------------------------------------------------
  qa_cc_news_a02_llm                   5,042      green
  qa_cc_news_a03_rule                  2,358      green
  qa_livedoor_a10_hybrid               1,845      green
------------------------------------------------------------
```

---

## 5. コマンドラインオプション

### 5.1 全オプション一覧

| オプション | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `--collection` | str | qa_corpus | 対象コレクション名 |
| `--qdrant-url` | str | http://localhost:6333 | QdrantサーバーのURL |
| `--domain` | str | なし | 削除対象のドメイン |
| `--all` | flag | False | 全データを削除（コレクションは保持） |
| `--all-collections` | flag | False | 全コレクションを削除（危険！） |
| `--drop-collection` | flag | False | コレクション自体を削除 |
| `--exclude` | str[] | なし | 除外するコレクション（複数指定可） |
| `--stats` | flag | False | 統計情報のみ表示（削除なし） |
| `--dry-run` | flag | False | 削除対象を表示するが実行しない |
| `--force` | flag | False | 確認プロンプトをスキップ |
| `--batch-size` | int | 100 | 削除バッチサイズ |

### 5.2 排他的オプション

以下のオプションは**1つのみ**指定可能（複数指定不可）:
- `--domain`
- `--all`
- `--all-collections`
- `--drop-collection`
- `--stats`

### 5.3 --exclude の制限

`--exclude`は`--all-collections`と併用時のみ有効。

---

## 6. 使用方法

### 6.1 統計情報の確認

```bash
# 全コレクションの統計情報を表示
python a41_qdrant_truncate.py --stats

# 特定コレクションの統計情報を表示
python a41_qdrant_truncate.py --collection qa_cc_news_a02_llm --stats
```

### 6.2 ドライラン（削除対象の確認のみ）

```bash
# 全データの削除対象を確認（実行はしない）
python a41_qdrant_truncate.py --all --dry-run

# 特定ドメインの削除対象を確認
python a41_qdrant_truncate.py --domain cc_news_llm --dry-run

# 全コレクションの削除対象を確認
python a41_qdrant_truncate.py --all-collections --dry-run
```

### 6.3 特定ドメインのデータ削除

```bash
# 確認プロンプト付きで削除
python a41_qdrant_truncate.py --domain medical

# 確認をスキップして強制削除
python a41_qdrant_truncate.py --domain cc_news_llm --force
```

### 6.4 コレクション内の全データ削除

```bash
# 確認プロンプト付き
python a41_qdrant_truncate.py --all

# 強制削除
python a41_qdrant_truncate.py --all --force

# 特定コレクションのデータ削除
python a41_qdrant_truncate.py --collection qa_cc_news_a03_rule --all --force
```

### 6.5 コレクション自体を削除

```bash
# 確認プロンプト付き
python a41_qdrant_truncate.py --collection qa_cc_news_a02_llm --drop-collection

# 特定コレクションを強制削除
python a41_qdrant_truncate.py --collection qa_cc_news_a02_llm --drop-collection --force
```

### 6.6 全コレクションの削除（危険！）

```bash
# 全コレクションを削除（2段階確認あり）
python a41_qdrant_truncate.py --all-collections --force

# 特定コレクションを除外して削除
python a41_qdrant_truncate.py --all-collections \
  --exclude qa_corpus \
  --exclude important_data \
  --force
```

---

## 7. 安全機能

### 7.1 確認プロンプト（confirm_action）

```
⚠️  ドメイン 'cc_news' のデータを削除します
この操作は取り消せません！

実行しますか？ (yes/no):
```

### 7.2 2段階確認（confirm_all_collections_deletion）

全コレクション削除時のみ:

1. **第一確認**: yes/no プロンプト
2. **第二確認**: 3秒カウントダウン（Ctrl+Cで中止可能）

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  警告: 全コレクション削除
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

削除予定: 7 コレクション
  - qa_cc_news_a02_llm (5,042 ポイント)
  - qa_cc_news_a03_rule (2,358 ポイント)
  ...

除外: 1 コレクション
  - qa_corpus

本当に削除しますか？ (yes/no): yes

最終確認：3秒後に削除を開始します。中止するにはCtrl+Cを押してください。
3...2...1...
```

### 7.3 ドライラン

`--dry-run`オプションで削除対象のみ表示（実際の削除は実行しない）:

```
削除対象: ドメイン 'cc_news' のデータ 5,042 件
[DRY RUN] 実際の削除は実行されません。
```

### 7.4 除外リスト

`--exclude`オプションで重要なコレクションを保護:

```bash
python a41_qdrant_truncate.py --all-collections \
  --exclude qa_corpus \
  --exclude backup_data \
  --force
```

### 7.5 削除後の統計情報

削除完了後に自動的に統計情報を表示して結果を検証。

---

## 8. トラブルシューティング

### 8.1 Qdrantサーバーに接続できない

**症状**:
```
❌ Qdrant接続エラー: Connection refused
URL: http://localhost:6333
Qdrantが起動していることを確認してください。
```

**解決方法**:
```bash
# Qdrantサーバーを起動
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# または docker-compose
cd docker-compose
docker-compose up -d qdrant
```

### 8.2 コレクションが存在しない

**症状**:
```
❌ コレクション 'qa_corpus' が存在しません。
```

**解決方法**:
```bash
# 全コレクションの確認
python a41_qdrant_truncate.py --stats

# 正しいコレクション名を指定
python a41_qdrant_truncate.py --collection qa_cc_news_a02_llm --stats
```

### 8.3 アクションを指定していない

**症状**:
```
❌ アクションを指定してください（--stats, --domain, --all, --all-collections, --drop-collection）
```

**解決方法**: 必ず1つのアクションを指定する。

### 8.4 複数のアクションを指定している

**症状**:
```
❌ 複数のアクションを同時に指定することはできません
```

**解決方法**: アクションは1つのみ指定する。

### 8.5 --exclude を不正に使用

**症状**:
```
❌ --exclude は --all-collections と併用してください
```

**解決方法**: `--exclude`は`--all-collections`と一緒に使用する。

---

## 付録: メタデータ

| 項目 | 値 |
|------|-----|
| ファイル行数 | 731行 |
| デフォルトコレクション | qa_corpus |
| デフォルトQdrant URL | http://localhost:6333 |
| デフォルトバッチサイズ | 100 |
| サポートドメイン数 | 9種類 |
| 確認カウントダウン | 3秒 |

## カラーコード

| 色 | ANSIコード | 用途 |
|----|-----------|------|
| HEADER | `\033[95m` | セクション見出し |
| OKBLUE | `\033[94m` | 情報ラベル |
| OKCYAN | `\033[96m` | ドライラン表示、進捗バー |
| OKGREEN | `\033[92m` | 成功メッセージ、正常ステータス |
| WARNING | `\033[93m` | 確認プロンプト、注意事項 |
| FAIL | `\033[91m` | エラーメッセージ、危険な操作 |
| BOLD | `\033[1m` | 重要な数値、操作 |

## 関連ファイル

| ファイル | 役割 |
|---------|------|
| a40_show_qdrant_data.py | Qdrantデータ表示UI |
| a42_qdrant_registration.py | Qdrantへのデータ登録 |
| a50_rag_search_local_qdrant.py | Qdrant検索UI |
| config.yml | デフォルト設定ファイル |
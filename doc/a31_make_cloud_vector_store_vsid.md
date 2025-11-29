# a31_make_cloud_vector_store_vsid.py - 技術仕様書

## 目次

1. [概要](#1-概要)
2. [アーキテクチャ](#2-アーキテクチャ)
3. [データセット設定](#3-データセット設定)
4. [主要クラス詳細](#4-主要クラス詳細)
5. [データ処理フロー](#5-データ処理フロー)
6. [UIタブ構成](#6-uiタブ構成)
7. [使用方法](#7-使用方法)
8. [出力ファイル](#8-出力ファイル)
9. [エラーハンドリング](#9-エラーハンドリング)
10. [トラブルシューティング](#10-トラブルシューティング)

---

## 1. 概要

### 1.1 目的

`a31_make_cloud_vector_store_vsid.py`は、OpenAI Vector Storeを作成・管理するためのStreamlitアプリケーションです。Q/A生成システム（a02, a03, a10）で作成されたCSVファイルをVector Store化し、RAGシステムで利用可能な形式に変換します。

### 1.2 起動コマンド

```bash
streamlit run a31_make_cloud_vector_store_vsid.py --server.port=8502
```

### 1.3 主要機能

- **個別Vector Store作成**: 6種類のデータセットから個別にVector Storeを作成
- **統合Vector Store作成**: 複数のデータセットを1つのVector Storeに統合
- **既存Vector Store管理**: 一覧表示、ストレージ使用量監視
- **ファイル状況確認**: 入力ファイルの存在確認とステータス表示

### 1.4 対応データセット

| データセット | ファイル名 | 説明 | 生成方式 |
|------------|----------|------|---------|
| a02_cc_news | a02_qa_pairs_cc_news.csv | CC News Q&A | LLM生成（基本） |
| a02_livedoor | a02_qa_pairs_livedoor.csv | Livedoor Q&A | LLM生成（基本） |
| a03_cc_news | a03_qa_pairs_cc_news.csv | CC News Q&A | ルールベース（カバレッジ改良） |
| a03_livedoor | a03_qa_pairs_livedoor.csv | Livedoor Q&A | ルールベース（カバレッジ改良） |
| a10_cc_news | a10_qa_pairs_cc_news.csv | CC News Q&A | ハイブリッド |
| a10_livedoor | a10_qa_pairs_livedoor.csv | Livedoor Q&A | ハイブリッド |

---

## 2. アーキテクチャ

### 2.1 システム構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                a31_make_cloud_vector_store_vsid.py              │
├─────────────────────────────────────────────────────────────────┤
│  VectorStoreConfig (52-151)                                     │
│  ├── データセット設定管理（@dataclass）                           │
│  ├── get_unified_config() - 統合設定                            │
│  └── get_all_configs() - 全データセット設定                       │
├─────────────────────────────────────────────────────────────────┤
│  VectorStoreProcessor (156-349)                                 │
│  ├── load_csv_file() - CSVファイル読み込み                        │
│  ├── chunk_text() - テキストチャンク分割                          │
│  ├── clean_text() - テキストクリーニング                          │
│  └── text_to_jsonl_data() - JSONL形式変換                        │
├─────────────────────────────────────────────────────────────────┤
│  VectorStoreManager (355-799)                                   │
│  ├── create_vector_store_from_jsonl_data() - Vector Store作成    │
│  ├── process_unified_datasets() - 統合処理                       │
│  ├── process_single_dataset() - 個別処理                         │
│  └── list_vector_stores() - 一覧表示                             │
├─────────────────────────────────────────────────────────────────┤
│  VectorStoreUI (805-1045)                                       │
│  ├── setup_page/header/sidebar() - UI設定                        │
│  ├── display_dataset_selection() - データセット選択               │
│  ├── display_file_status() - ファイル状況表示                     │
│  ├── display_results() - 結果表示                                │
│  └── display_existing_stores() - 既存Store一覧                   │
├─────────────────────────────────────────────────────────────────┤
│  main() (1059-1466)                                             │
│  ├── Tab1: 個別作成                                              │
│  ├── Tab2: 統合作成                                              │
│  ├── Tab3: ファイル状況                                          │
│  └── Tab4: 既存Store一覧                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 依存モジュール

```python
from openai import OpenAI
from helper_rag import (
    AppConfig, RAGConfig, TokenManager, safe_execute,
    select_model, show_model_info,
    setup_page_config, setup_page_header, setup_sidebar_header,
    create_output_directory
)
import streamlit as st
import pandas as pd
from dataclasses import dataclass
```

---

## 3. データセット設定

### 3.1 VectorStoreConfigクラス

```python
@dataclass
class VectorStoreConfig:
    """Vector Store設定データクラス"""
    dataset_type: str           # データセット種別
    filename: str               # 入力ファイル名
    store_name: str             # Vector Store名
    description: str            # データセット説明
    chunk_size: int = 1000      # チャンクサイズ
    overlap: int = 100          # オーバーラップサイズ
    max_file_size_mb: int = 400 # 最大ファイルサイズ（MB）
    max_chunks_per_file: int = 40000  # 最大チャンク数
    csv_text_column: str = "Combined_Text"  # テキストカラム名
```

### 3.2 データセット別設定

| データセット | chunk_size | overlap | max_file_size_mb | max_chunks | text_column |
|------------|-----------|---------|-----------------|------------|-------------|
| a02_cc_news | 2000 | 100 | 30 | 4000 | question |
| a02_livedoor | 2000 | 100 | 30 | 4000 | question |
| a03_cc_news | 2000 | 100 | 30 | 4000 | question |
| a03_livedoor | 2000 | 100 | 30 | 4000 | question |
| a10_cc_news | 2000 | 100 | 30 | 4000 | question |
| a10_livedoor | 2000 | 100 | 30 | 4000 | question |

### 3.3 統合設定

```python
VectorStoreConfig.get_unified_config():
    dataset_type = "unified_all"
    store_name = "Unified Knowledge Base - All Domains"
    description = "全ドメイン統合ナレッジベース"
    chunk_size = 3000        # 中間的なサイズ
    overlap = 200
    max_file_size_mb = 100   # 統合時の制限を緩和
    max_chunks_per_file = 50000
    csv_text_column = "Combined_Text"
```

---

## 4. 主要クラス詳細

### 4.1 VectorStoreProcessor

#### load_csv_file (162-194)

```python
def load_csv_file(self, filepath: Path, text_column: str = "Combined_Text") -> List[str]:
    """CSVファイルを読み込み、指定カラムのテキストをリストとして返す

    処理:
    1. CSVファイル読み込み（UTF-8）
    2. 指定カラムの存在確認
    3. NaN値除外、文字列変換
    4. 空文字列と10文字未満のテキストを除去
    """
```

#### chunk_text (196-227)

```python
def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """長いテキストを指定サイズのチャンクに分割

    特徴:
    - 句読点（。！？.!?）を境界として分割
    - オーバーラップ処理で文脈を保持
    - 無限ループ防止機構
    """
```

#### text_to_jsonl_data (245-349)

```python
def text_to_jsonl_data(
    self,
    lines: List[str],
    dataset_type: str,
    source_dataset: str = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """テキスト行をJSONL用のデータ構造に変換

    Returns:
        (jsonl_data, stats)
        - jsonl_data: [{"id": str, "text": str, "metadata": dict}, ...]
        - stats: 統計情報辞書
    """
```

**メタデータ構造**:
```json
{
    "dataset": "a02_cc_news",
    "original_line": 0,
    "chunk_index": 0,
    "total_chunks": 1,
    "domain": "general"  // 統合モード時のみ
}
```

### 4.2 VectorStoreManager

#### create_vector_store_from_jsonl_data (370-510)

```python
def create_vector_store_from_jsonl_data(
    self,
    jsonl_data: List[Dict],
    store_name: str
) -> Optional[str]:
    """JSONL形式のデータからVector Storeを作成

    処理フロー:
    1. 一時ファイル作成（.txt拡張子）
    2. JSONL形式でデータ書き込み
    3. OpenAI Filesにアップロード（purpose="assistants"）
    4. Vector Store作成（メタデータ付与）
    5. ファイルをVector Storeにリンク
    6. 処理完了をポーリング（最大10分、5秒間隔）
    """
```

#### process_unified_datasets (512-650)

```python
def process_unified_datasets(
    self,
    selected_datasets: List[str],
    output_dir: Path = None
) -> Dict[str, Any]:
    """複数データセットを統合してVector Storeを作成

    処理:
    1. 各データセットのCSVを読み込み
    2. 統合設定でJSONL変換（source_datasetを保持）
    3. サイズ制限チェック（100MB超過時は自動削減）
    4. 統合Vector Store作成
    """
```

#### process_single_dataset (652-778)

```python
def process_single_dataset(
    self,
    dataset_type: str,
    output_dir: Path = None
) -> Dict[str, Any]:
    """単一データセットの処理

    処理:
    1. CSVファイル読み込み
    2. JSONL変換
    3. サイズ制限チェック（25MB制限）
    4. Vector Store作成

    Returns:
        {
            "success": bool,
            "vector_store_id": str,
            "store_name": str,
            "processed_lines": int,
            "total_lines": int,
            "created_chunks": int,
            "estimated_size_mb": float,
            "warnings": List[str],
            "config_used": dict
        }
    """
```

### 4.3 VectorStoreUI

#### display_dataset_selection (866-898)

データセット選択UIを2カラムレイアウトで表示。各データセットのファイル存在確認も実施。

#### display_results (935-1003)

処理結果を成功/失敗に分けて表示。警告詳細、設定情報、対処法の提案を含む。

---

## 5. データ処理フロー

```
┌─────────────────────────────────────────────────────────────────┐
│ 入力: qa_output/a02_qa_pairs_cc_news.csv                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. load_csv_file()                                              │
│    - CSV読み込み（UTF-8）                                         │
│    - questionカラムからテキスト抽出                                │
│    - 10文字未満を除去                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. text_to_jsonl_data()                                         │
│    - clean_text() でクリーニング                                  │
│    - chunk_text() でチャンク分割                                  │
│    - JSONL形式に変換                                             │
│    - サイズ・チャンク数制限チェック                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. create_vector_store_from_jsonl_data()                        │
│    - 一時ファイル作成（.txt）                                      │
│    - OpenAI Filesにアップロード                                   │
│    - Vector Store作成                                            │
│    - ファイルリンク                                               │
│    - ポーリング（最大10分）                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 出力: Vector Store ID (例: vs_68fcbd4bb2d08191b0ef5606b5971c3d)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. UIタブ構成

### 6.1 Tab1: 個別作成

- データセット選択チェックボックス（6種類）
- ファイル存在確認ステータス
- 一括処理オプション（サイドバー）
- プログレスバー付き処理実行
- 結果表示（Vector Store ID、統計情報）

### 6.2 Tab2: 統合作成

- 統合設定の表示（chunk_size, overlap, max_file_size_mb）
- クイック選択ボタン（全て選択/全て解除）
- データセット選択チェックボックス
- 推定合計サイズ表示
- データセット別統計情報
- 警告表示

### 6.3 Tab3: ファイル状況

- ファイル一覧テーブル（データセット、ファイル名、サイズ、更新日時、状態）
- ディレクトリ詳細情報（CSVファイル、TXTファイル一覧）

### 6.4 Tab4: 既存Store一覧

- Vector Store一覧テーブル（番号、名前、ID、ファイル数、ストレージ使用量、作成日時）
- 統計情報（総Store数、総ストレージ使用量、総ファイル数）
- 一覧更新ボタン

---

## 7. 使用方法

### 7.1 環境変数設定

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

### 7.2 個別作成の手順

1. 「個別作成」タブを選択
2. 作成したいデータセットにチェック
3. 「Vector Store作成開始」ボタンをクリック
4. 進行状況を確認
5. 完了後、Vector Store IDをコピー

### 7.3 統合作成の手順

1. 「統合作成」タブを選択
2. 統合したいデータセットを選択（または「全て選択」）
3. 推定サイズを確認（100MB以下推奨）
4. 「統合Vector Store作成」ボタンをクリック
5. 完了後、統合Vector Store IDをコピー

### 7.4 サイドバーオプション

- **参考モデル選択**: gpt-4o-mini, gpt-4o, gpt-4.1-mini, gpt-4.1
- **全データセット一括処理**: チェックで6データセットすべてを処理
- **API設定確認**: APIキーの設定状態を表示

---

## 8. 出力ファイル

### 8.1 結果JSONダウンロード

```json
{
  "a02_cc_news": {
    "success": true,
    "vector_store_id": "vs_68fcbd4bb2d08191b0ef5606b5971c3d",
    "store_name": "CC News Q&A - Basic Generation (a02_make_qa)",
    "processed_lines": 4000,
    "total_lines": 5389,
    "created_chunks": 4000,
    "estimated_size_mb": 14.9,
    "warnings": [],
    "config_used": {
      "chunk_size": 2000,
      "overlap": 100
    }
  }
}
```

### 8.2 Vector Store IDテキスト

```python
# CC NewsデータセットQ&A（基本生成方式）
A02_CC_NEWS_VECTOR_STORE_ID = "vs_68fcbd4bb2d08191b0ef5606b5971c3d"

# LivedoorデータセットQ&A（基本生成方式）
A02_LIVEDOOR_VECTOR_STORE_ID = "vs_68fcbd5a90988191b3e1aff1e5430089"

# 統合Vector Store
UNIFIED_VECTOR_STORE_ID = "vs_unified_12345678901234567890"
```

---

## 9. エラーハンドリング

### 9.1 404エラー対策

ファイル登録直後は404エラーが発生する可能性があるため、リトライ機構を実装:

```python
# 初回待機（3秒）
time.sleep(initial_wait)

# リトライロジック（最大10分、5秒間隔）
while waited_time < max_wait_time:
    try:
        file_status = client.vector_stores.files.retrieve(...)
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            time.sleep(wait_interval)
            waited_time += wait_interval
        else:
            raise
```

### 9.2 サイズ制限

| 処理モード | ファイルサイズ制限 | チャンク数制限 |
|-----------|-----------------|--------------|
| 個別作成 | 25MB | 4,000 |
| 統合作成 | 100MB | 50,000 |

サイズ超過時は自動削減（90%を目標）。

### 9.3 ファイル処理ステータス

| ステータス | 動作 |
|-----------|------|
| completed | 成功 → Vector Store ID返却 |
| failed | 失敗 → エラーログ出力、None返却 |
| in_progress | 処理中 → 待機継続 |
| cancelling | キャンセル中 → 待機継続 |

---

## 10. トラブルシューティング

### 10.1 404エラーが継続する場合

- 待機時間を延長（`max_wait_time`を600秒以上に）
- ネットワーク接続を確認
- OpenAI APIの状態を確認

### 10.2 ファイルサイズ超過

- `chunk_size`を増加（2000 → 3000等）
- データを分割して複数のVector Storeに分ける
- 不要な行をフィルタリング

### 10.3 APIキー未設定

```bash
# 環境変数設定
export OPENAI_API_KEY='your-api-key-here'

# または.envファイルに記載
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 10.4 入力ファイルが見つからない

```bash
# ファイル存在確認
ls -la qa_output/a02_qa_pairs_cc_news.csv

# ファイルがない場合は該当のQ/A生成スクリプトを実行
python a02_make_qa_para.py --dataset cc_news
```

---

## 付録: メタデータ

| 項目 | 値 |
|------|-----|
| バージョン | 2025.1 |
| データ形式 | jsonl_as_txt |
| ポート | 8502 |
| 最大待機時間 | 600秒（10分） |
| ポーリング間隔 | 5秒 |
| 初回待機 | 3秒 |

## 関連ファイル

| ファイル | 説明 |
|---------|------|
| a02_make_qa_para.py | 基本Q&A生成（LLM） |
| a03_rag_qa_coverage_improved.py | カバレッジ改良Q&A生成（ルールベース） |
| a10_qa_optimized_hybrid_batch.py | ハイブリッドQ&A生成 |
| a34_rag_search_cloud_vs.py | Vector Store検索（作成したIDを使用） |
| helper_rag.py | 共通ヘルパー関数 |
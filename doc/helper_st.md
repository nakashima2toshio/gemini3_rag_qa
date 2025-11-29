# helper_st.py 仕様書

作成日: 2025-11-27

## 概要

カスタマーサポートFAQデータのRAG前処理用Streamlitアプリケーション。モデル選択機能、データ前処理、トークン使用量推定、ファイル保存機能を提供する独立実装版モジュール。

## ファイル情報

- **ファイル名**: helper_st.py
- **行数**: 898行
- **主な機能**: カスタマーサポートFAQデータの前処理UI
- **フレームワーク**: Streamlit
- **起動コマンド**: `streamlit run helper_st.py --server.port=8501`

---

## アーキテクチャ

### モジュール構造

```
helper_st.py
├── インポート・ログ設定 (L1-18)
│
├── 設定管理 (L21-84)
│   └── AppConfig (L24-83)
│
├── RAG設定 (L86-111)
│   └── RAGConfig (L89-111)
│
├── トークン管理 (L114-136)
│   └── TokenManager (L117-136)
│
├── UI関数 (L139-211)
│   ├── select_model() (L142-160)
│   └── show_model_info() (L163-211)
│
├── デコレータ (L214-229)
│   └── error_handler (L217-229)
│
├── データ処理関数 (L232-444)
│   ├── clean_text() (L235-253)
│   ├── combine_columns() (L256-269)
│   ├── validate_data() (L272-305)
│   ├── load_dataset() (L308-322)
│   ├── process_rag_data() (L325-359)
│   ├── create_download_data() (L362-380)
│   ├── display_statistics() (L383-408)
│   └── estimate_token_usage() (L411-444)
│
├── ファイル保存関数 (L447-527)
│   ├── create_output_directory() (L450-471)
│   └── save_files_to_output() (L474-527)
│
├── カスタマーサポート特有処理 (L530-586)
│   ├── validate_customer_support_data_specific() (L533-562)
│   └── show_usage_instructions() (L565-586)
│
└── メイン処理 (L589-898)
    └── main() (L592-888)
```

---

## 主要クラス

### 1. AppConfig (L24-83)

アプリケーション設定（独立実装）。

#### 利用可能モデル定義 (L28-41)

```python
AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-audio-preview",
    "gpt-4o-mini-audio-preview",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o1",
    "o1-mini",
    "o3",
    "o3-mini",
    "o4",
    "o4-mini"
]

DEFAULT_MODEL = "gpt-4o-mini"
```

#### モデル料金 (L46-59)

1000トークンあたりのUSD価格：

| モデル | 入力 | 出力 |
|--------|------|------|
| gpt-4o | $0.005 | $0.015 |
| gpt-4o-mini | $0.00015 | $0.0006 |
| gpt-4o-audio-preview | $0.01 | $0.02 |
| gpt-4o-mini-audio-preview | $0.00025 | $0.001 |
| gpt-4.1 | $0.0025 | $0.01 |
| gpt-4.1-mini | $0.0001 | $0.0004 |
| o1 | $0.015 | $0.06 |
| o1-mini | $0.003 | $0.012 |
| o3 | $0.03 | $0.12 |
| o3-mini | $0.006 | $0.024 |
| o4 | $0.05 | $0.20 |
| o4-mini | $0.01 | $0.04 |

#### モデル制限 (L62-73)

| モデル | max_tokens | max_output |
|--------|-----------|------------|
| gpt-4o | 128,000 | 4,096 |
| gpt-4o-mini | 128,000 | 4,096 |
| gpt-4.1 | 128,000 | 4,096 |
| gpt-4.1-mini | 128,000 | 4,096 |
| o1 | 128,000 | 32,768 |
| o1-mini | 128,000 | 65,536 |
| o3 | 200,000 | 100,000 |
| o3-mini | 200,000 | 100,000 |
| o4 | 256,000 | 128,000 |
| o4-mini | 256,000 | 128,000 |

#### 主要メソッド

| メソッド | 行番号 | 説明 |
|---------|--------|------|
| `get_model_limits(model)` | L75-78 | モデルの制限取得 |
| `get_model_pricing(model)` | L80-83 | モデルの料金取得 |

---

### 2. RAGConfig (L89-111)

RAGデータ前処理の設定。

#### データセット設定 (L92-100)

```python
DATASET_CONFIGS = {
    "customer_support_faq": {
        "name": "カスタマーサポート・FAQ",
        "icon": "💬",
        "required_columns": ["question", "answer"],
        "description": "カスタマーサポートFAQデータセット",
        "combine_template": "{question} {answer}"
    }
}
```

#### 主要メソッド

| メソッド | 行番号 | 説明 |
|---------|--------|------|
| `get_config(dataset_type)` | L102-111 | データセット設定の取得 |

---

### 3. TokenManager (L117-136)

トークン数の管理（簡易版）。

#### count_tokens() (L120-128)

```python
@staticmethod
def count_tokens(text: str, model: str = None) -> int:
    """テキストのトークン数をカウント（簡易推定）"""
    if not text:
        return 0
    # 簡易推定: 日本語文字は0.5トークン、英数字は0.25トークン
    japanese_chars = len([c for c in text if ord(c) > 127])
    english_chars = len(text) - japanese_chars
    return int(japanese_chars * 0.5 + english_chars * 0.25)
```

#### estimate_cost() (L130-136)

```python
@staticmethod
def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """API使用コストの推定"""
    pricing = AppConfig.get_model_pricing(model)
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    return input_cost + output_cost
```

---

## UI関数

### select_model() (L142-160)

モデル選択UI。

```python
def select_model(key: str = "model_selection") -> str:
    """モデル選択UI"""
    models = AppConfig.AVAILABLE_MODELS
    default_model = AppConfig.DEFAULT_MODEL

    selected = st.sidebar.selectbox(
        "🤖 モデルを選択",
        models,
        index=default_index,
        key=key,
        help="利用するOpenAIモデルを選択してください"
    )
    return selected
```

---

### show_model_info() (L163-211)

選択されたモデルの情報を表示。

**表示内容**:
- 最大入力/出力トークン数
- 料金（1000トークンあたり）
- モデル特性（推論特化/音声対応/マルチモーダル/標準対話）
- RAG用途推奨度

**RAG用途推奨度の判定**:

| モデル | 推奨度 |
|--------|--------|
| gpt-4o-mini, gpt-4.1-mini | ✅ 最適（コスト効率良好） |
| gpt-4o, gpt-4.1 | 💡 高品質（コスト高） |
| o1, o3, o4系 | ⚠️ 推論特化（RAG用途には過剰） |
| その他 | 💬 標準的な性能 |

---

## デコレータ

### error_handler (L217-229)

```python
def error_handler(func):
    """エラーハンドリングデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            st.error(f"エラーが発生しました: {str(e)}")
            return None
    return wrapper
```

---

## データ処理関数

### clean_text() (L235-253)

テキストのクレンジング処理。

**処理内容**:
1. 改行を空白に置換（`\n`, `\r`）
2. 連続した空白を1つにまとめる
3. 先頭・末尾の空白を除去
4. 引用符の正規化（`"` → `"`, `'` → `'`）

---

### combine_columns() (L256-269)

複数列を結合して1つのテキストにする。

```python
def combine_columns(row: pd.Series, dataset_type: str = "customer_support_faq") -> str:
    """複数列を結合して1つのテキストにする"""
    config_data = RAGConfig.get_config(dataset_type)
    required_columns = config_data["required_columns"]

    cleaned_values = {}
    for col in required_columns:
        value = row.get(col, '')
        cleaned_values[col.lower()] = clean_text(str(value))

    combined = " ".join(cleaned_values.values())
    return combined.strip()
```

---

### validate_data() (L272-305)

データの検証。

**検証内容**:
- 基本統計（総行数、総列数）
- 必須列の確認
- 各列の空値確認
- 重複行の確認

---

### load_dataset() (L308-322)

データセットの読み込みと基本検証。

```python
def load_dataset(uploaded_file, dataset_type: str = None) -> Tuple[pd.DataFrame, List[str]]:
    """データセットの読み込みと基本検証"""
    df = pd.read_csv(uploaded_file)
    validation_results = validate_data(df, dataset_type)
    return df, validation_results
```

---

### process_rag_data() (L325-359)

RAGデータの前処理を実行。

**処理内容**:
1. 重複行の除去
2. 空行の除去（全列がNAの行）
3. インデックスのリセット
4. 各列のクレンジング
5. 列の結合（オプション）→ `Combined_Text`列作成

---

### create_download_data() (L362-380)

ダウンロード用データの作成。

**戻り値**:
- `csv_data`: CSV形式の文字列
- `text_data`: 結合テキストのみ（改行区切り）

---

### display_statistics() (L383-408)

処理前後の統計情報を表示。

**表示内容**:
- 元の行数 / 処理後の行数 / 除去された行数
- 結合後テキスト分析（平均/最大/最小文字数）

---

### estimate_token_usage() (L411-444)

処理済みデータのトークン使用量推定。

**表示内容**:
- 推定総トークン数
- 平均トークン/レコード
- 推定embedding費用

---

## ファイル保存関数

### create_output_directory() (L450-471)

OUTPUTディレクトリの作成。

**特徴**:
- `OUTPUT`ディレクトリ作成
- 書き込み権限テスト実行
- 権限不足時は`PermissionError`発生

---

### save_files_to_output() (L474-527)

処理済みデータをOUTPUTフォルダに保存。

**保存ファイル**:

| ファイル種類 | ファイル名 | 内容 |
|-------------|-----------|------|
| CSV | `preprocessed_{type}_{rows}rows_{timestamp}.csv` | 前処理済みデータ |
| テキスト | `{dataset_type}.txt` | 結合テキストのみ |
| メタデータ | `metadata_{type}_{timestamp}.json` | 処理情報 |

**メタデータ構造**:
```python
{
    "dataset_type": "customer_support_faq",
    "processed_rows": 100,
    "processing_timestamp": "20251127_120000",
    "created_at": "2025-11-27T12:00:00",
    "files_created": ["csv", "txt", "metadata"]
}
```

---

## カスタマーサポート特有処理

### validate_customer_support_data_specific() (L533-562)

カスタマーサポートFAQデータ特有の検証。

**検証内容**:
1. サポート関連用語の存在確認
2. 回答の長さ分析

**サポート関連用語**:
```python
support_keywords = [
    '問題', '解決', 'トラブル', 'エラー', 'サポート', 'ヘルプ', '対応',
    'problem', 'issue', 'error', 'help', 'support', 'solution', 'troubleshoot'
]
```

---

### show_usage_instructions() (L565-586)

使用方法の説明を表示。

**表示内容**:
1. モデル選択
2. CSVファイルアップロード
3. 前処理実行
4. 複数列結合
5. トークン使用量確認
6. ダウンロード

---

## メイン処理 (main関数)

### 処理フロー

```
アプリケーション起動
    ↓
ページ設定（st.set_page_config）
    ↓
サイドバー設定
    ├── モデル選択（select_model）
    ├── モデル情報表示（show_model_info）
    └── 前処理設定（チェックボックス）
    ↓
ファイルアップロード
    ↓
データ読み込み（load_dataset）
    ↓
データ検証表示（validate_data）
    ↓
前処理実行ボタン
    ├── process_rag_data()
    ├── display_statistics()
    ├── estimate_token_usage()
    └── カスタマーサポート特有分析
    ↓
ダウンロード・保存セクション
    ├── ブラウザダウンロード（st.download_button）
    └── ローカル保存（save_files_to_output）
    ↓
使用方法説明（show_usage_instructions）
```

### セッション状態管理

| キー | 説明 |
|------|------|
| `current_file_key` | ファイル識別キー |
| `original_df` | 元データ |
| `validation_results` | 検証結果 |
| `original_rows` | 元の行数 |
| `file_processed` | 処理完了フラグ |
| `processed_df` | 処理済みデータ |
| `download_data` | ダウンロード用データ |
| `download_data_key` | ダウンロードデータキー |

---

## 使用例

### アプリケーション起動

```bash
streamlit run helper_st.py --server.port=8501
```

### プログラムから利用

```python
from helper_st import (
    AppConfig, RAGConfig, TokenManager,
    clean_text, combine_columns, validate_data,
    load_dataset, process_rag_data, create_download_data,
    save_files_to_output
)

# トークン数計算
text = "これはテストテキストです"
token_count = TokenManager.count_tokens(text)

# コスト推定
cost = TokenManager.estimate_cost(
    input_tokens=1000,
    output_tokens=500,
    model="gpt-4o-mini"
)
print(f"推定コスト: ${cost:.4f}")

# データ前処理（Streamlit外で使用）
import pandas as pd

df = pd.read_csv("customer_support_faq.csv")
df_processed = process_rag_data(df, "customer_support_faq", combine_columns_option=True)

csv_data, text_data = create_download_data(df_processed)
saved_files = save_files_to_output(df_processed, "customer_support_faq", csv_data, text_data)
```

---

## 依存ライブラリ

```python
import streamlit as st
import pandas as pd
import re
import io
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from functools import wraps
```

---

## 注意事項

1. **ポート番号**: デフォルトは8501
2. **ファイル形式**: UTF-8エンコードのCSVファイル
3. **必須列**: `question`, `answer`
4. **出力先**: `OUTPUT`ディレクトリ

---

## トラブルシューティング

### 問題1: ポートが使用中

**症状**: `Address already in use`

**解決策**:
```bash
streamlit run helper_st.py --server.port=8502
```

### 問題2: 必須列が見つからない

**症状**: "必須列が不足"エラー

**解決策**:
- CSVファイルに`question`, `answer`列が存在するか確認
- 列名の大文字小文字を確認

### 問題3: OUTPUT書き込みエラー

**症状**: PermissionError

**解決策**:
```bash
# ディレクトリの権限確認
ls -la OUTPUT

# 権限付与
chmod 755 OUTPUT
```

### 問題4: セッション状態エラー

**症状**: セッション状態が不整合

**解決策**:
- ブラウザをリロード
- 別のブラウザタブで開く

---

## UI構成

### サイドバー

```
💬 カスタマーサポートFAQ
────────────────────────
🤖 モデルを選択
  [gpt-4o-mini ▼]

📊 選択モデル情報
  ├── 最大入力: 128,000
  ├── 最大出力: 4,096
  ├── 料金（1000トークン）
  │   ├── 入力: $0.00015
  │   └── 出力: $0.00060
  └── RAG用途推奨度: ✅ 最適
────────────────────────
前処理設定
  ☑️ 複数列を結合する
  ☑️ データ検証を表示

💬 サポートデータ設定
  ☑️ 書式を保護
  ☑️ 質問を正規化
```

### メインエリア

```
💬 カスタマーサポートFAQデータ前処理アプリ
────────────────────────────────────────────

📁 データファイルのアップロード
  📊 選択中のモデル情報
    🤖 選択モデル: gpt-4o-mini
    📏 最大トークン: 128,000

  [CSVファイルをドラッグ＆ドロップ]

📋 元データプレビュー
  [データフレーム表示]

🔍 データ検証
  ✅ 必須列確認済み: ['question', 'answer']
  ✅ 重複行なし

⚙️ 前処理実行
  [前処理を実行]

✅ 前処理後のデータプレビュー
  [データフレーム表示]

📊 統計情報
  元の行数: 100  処理後の行数: 98  除去された行数: 2

💾 ダウンロード・保存
  📥 ブラウザダウンロード
    [CSV形式でダウンロード] [テキスト形式でダウンロード]
  💾 ローカルファイル保存
    [OUTPUTフォルダに保存]

📖 使用方法
```

---

## RAG使用推奨モデル

### 高速・低コスト（推奨）

| モデル | 入力/出力 | 特徴 |
|--------|----------|------|
| gpt-4o-mini | $0.00015 / $0.0006 | デフォルト推奨 |
| gpt-4.1-mini | $0.0001 / $0.0004 | RAG最適 |

### バランス型（中価格）

| モデル | 入力/出力 | 特徴 |
|--------|----------|------|
| gpt-4o | $0.005 / $0.015 | 高性能バランス |
| gpt-4.1 | $0.0025 / $0.01 | 改良版 |

### 推論型（RAG使用には高価）

| モデル | 入力/出力 | 特徴 |
|--------|----------|------|
| o1-mini | $0.003 / $0.012 | 軽量推論 |
| o3-mini | $0.006 / $0.024 | 中級推論 |
| o4-mini | $0.01 / $0.04 | 高度推論 |

---

## まとめ

helper_st.pyは、カスタマーサポートFAQデータのRAG前処理を行うStreamlitアプリケーションです。

### 主要な特徴

1. **モデル選択機能**: 12種類のOpenAIモデルから選択可能
2. **データ前処理**: クレンジング、重複除去、列結合
3. **トークン推定**: 選択モデルでのトークン数とコスト推定
4. **ファイル保存**: CSV、テキスト、メタデータの自動保存
5. **セッション管理**: Streamlitセッション状態による効率的な処理

### 推奨用途

- カスタマーサポートFAQデータの前処理
- Vector Store/RAG用テキストの準備
- OpenAI embeddingモデルへの最適化
- 処理コストの事前見積もり

---

作成日: 2025-11-27
作成者: OpenAI RAG Q/A JP Development Team
# helper_rag.py 仕様書

作成日: 2024-10-29
更新日: 2025-11-27

## 概要

RAGシステムのデータ前処理とStreamlitインターフェースを支援するヘルパーモジュール。OpenAI RAGシステムにおけるデータ管理とUI構築を担当。

## ファイル情報

- **ファイル名**: helper_rag.py
- **行数**: 819行
- **主な機能**: データ前処理とUI支援
- **依存ライブラリ**: pandas、streamlit、re、json、logging

---

## アーキテクチャ

### モジュール構造

```
helper_rag.py
├── ログ設定 (L16-20)
│
├── インポート (L27-31)
│   └── config.ModelConfig（後方互換性）
│
├── AppConfig (L37-107)
│   ├── AVAILABLE_MODELS (L41-57)
│   ├── DEFAULT_MODEL (L59)
│   ├── MODEL_PRICING (L62-78)
│   ├── MODEL_LIMITS (L81-97)
│   ├── get_model_limits() (L99-102)
│   └── get_model_pricing() (L104-107)
│
├── RAGConfig (L113-191)
│   ├── DATASET_CONFIGS (L116-166)
│   ├── get_config() (L168-178)
│   ├── get_all_datasets() (L180-183)
│   └── get_dataset_by_port() (L185-191)
│
├── TokenManager (L197-220)
│   ├── count_tokens() (L200-212)
│   └── estimate_cost() (L214-220)
│
├── デコレータ (L226-238)
│   └── safe_execute
│
├── UI関数 (L244-351)
│   ├── select_model() (L244-262)
│   ├── show_model_info() (L265-313)
│   └── estimate_token_usage() (L316-351)
│
├── データ処理関数 (L357-590)
│   ├── clean_text() (L357-378)
│   ├── combine_columns() (L381-415)
│   ├── validate_data() (L418-467)
│   ├── load_dataset() (L470-480)
│   ├── process_rag_data() (L483-529)
│   ├── create_download_data() (L532-551)
│   └── display_statistics() (L554-590)
│
├── ファイル保存関数 (L596-673)
│   ├── create_output_directory() (L596-617)
│   └── save_files_to_output() (L620-673)
│
├── ページ設定関数 (L679-780)
│   ├── show_usage_instructions() (L679-745)
│   ├── setup_page_config() (L751-763)
│   ├── setup_page_header() (L766-772)
│   └── setup_sidebar_header() (L775-780)
│
└── エクスポート定義 (L786-818)
```

---

## 主要クラス

### 1. AppConfig (L37-107)

アプリケーション全体の設定管理クラス。

#### 利用可能モデル定義 (L41-57)

```python
AVAILABLE_MODELS = [
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5",
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

DEFAULT_MODEL = "gpt-5-mini"
```

#### 価格設定 (L62-78)

1000トークンあたりのUSD価格：

| モデル | 入力 | 出力 | 特徴 |
|--------|------|------|------|
| **gpt-5** | $0.01 | $0.03 | 最新フラッグシップ |
| **gpt-5-mini** | $0.0001 | $0.0004 | RAG推奨 |
| **gpt-5-nano** | $0.00005 | $0.0002 | 超軽量 |
| **gpt-4o** | $0.005 | $0.015 | 高性能バランス |
| **gpt-4o-mini** | $0.00015 | $0.0006 | RAG推奨 |
| **gpt-4o-audio-preview** | $0.01 | $0.02 | 音声対応 |
| **gpt-4o-mini-audio-preview** | $0.00025 | $0.001 | 軽量音声対応 |
| **gpt-4.1** | $0.0025 | $0.01 | 改良版 |
| **gpt-4.1-mini** | $0.0001 | $0.0004 | RAG推奨 |
| **o1** | $0.015 | $0.06 | 推論特化 |
| **o1-mini** | $0.003 | $0.012 | 軽量推論 |
| **o3** | $0.03 | $0.12 | 上級推論 |
| **o3-mini** | $0.006 | $0.024 | 中級推論 |
| **o4** | $0.05 | $0.20 | 最上級推論 |
| **o4-mini** | $0.01 | $0.04 | 高度推論 |

#### モデル制限設定 (L81-97)

| モデル | max_tokens | max_output |
|--------|-----------|------------|
| gpt-5 | 256,000 | 8,192 |
| gpt-5-mini | 128,000 | 4,096 |
| gpt-5-nano | 64,000 | 2,048 |
| gpt-4o | 128,000 | 4,096 |
| gpt-4o-mini | 128,000 | 4,096 |
| gpt-4o-audio-preview | 128,000 | 4,096 |
| gpt-4o-mini-audio-preview | 128,000 | 4,096 |
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
| `get_model_limits(model)` | L99-102 | モデルの制限取得 |
| `get_model_pricing(model)` | L104-107 | モデルの料金取得 |

---

### 2. RAGConfig (L113-191)

RAGシステムのデータセット設定管理。

#### データセット設定 (L116-166)

| データセット | アイコン | 必須列 | ポート |
|-------------|---------|--------|--------|
| **customer_support_faq** | 💬 | question, answer | 8501 |
| **medical_qa** | 🏥 | Question, Complex_CoT, Response | 8503 |
| **sciq_qa** | 🔬 | question, correct_answer | 8504 |
| **legal_qa** | ⚖️ | question, answer | 8505 |
| **trivia_qa** | 🎯 | question, answer | 8506 |

#### 設定詳細

**customer_support_faq** (L118-125)
```python
{
    "name": "カスタマーサポート・FAQ",
    "icon": "💬",
    "required_columns": ["question", "answer"],
    "description": "カスタマーサポートFAQデータセット",
    "combine_template": "{question} {answer}",
    "port": 8501
}
```

**medical_qa** (L128-135)
```python
{
    "name": "医療QAデータ",
    "icon": "🏥",
    "required_columns": ["Question", "Complex_CoT", "Response"],
    "description": "医療質問回答データセット",
    "combine_template": "{question} {complex_cot} {response}",
    "port": 8503
}
```

**sciq_qa** (L138-145)
```python
{
    "name": "科学・技術QA（SciQ）",
    "icon": "🔬",
    "required_columns": ["question", "correct_answer"],
    "description": "科学・技術質問回答データセット",
    "combine_template": "{question} {correct_answer}",
    "port": 8504
}
```

**legal_qa** (L148-155)
```python
{
    "name": "法律・判例QA",
    "icon": "⚖️",
    "required_columns": ["question", "answer"],
    "description": "法律・判例質問回答データセット",
    "combine_template": "{question} {answer}",
    "port": 8505
}
```

**trivia_qa** (L158-165)
```python
{
    "name": "雑学QA（TriviaQA）",
    "icon": "🎯",
    "required_columns": ["question", "answer"],
    "description": "雑学質問回答データセット",
    "combine_template": "{question} {answer} {entity_pages} {search_results}",
    "port": 8506
}
```

#### 主要メソッド

| メソッド | 行番号 | 説明 |
|---------|--------|------|
| `get_config(dataset_type)` | L168-178 | データセット設定の取得 |
| `get_all_datasets()` | L180-183 | 全データセットタイプのリスト取得 |
| `get_dataset_by_port(port)` | L185-191 | ポート番号からデータセット取得 |

---

### 3. TokenManager (L197-220)

トークン数の推定と管理を行うクラス。

#### count_tokens() (L200-212)

```python
@staticmethod
def count_tokens(text: str, model: str = None) -> int:
    """テキストのトークン数をカウント（簡易推定）"""
    if not text:
        return 0

    # 簡易推定: 日本語文字は0.5トークン、英数字は0.25トークン
    japanese_chars = len([c for c in text if ord(c) > 127])
    english_chars = len(text) - japanese_chars
    estimated_tokens = int(japanese_chars * 0.5 + english_chars * 0.25)

    return max(1, estimated_tokens)
```

**推定ルール**:
- 日本語文字（ord > 127）: 0.5トークン
- 英数字: 0.25トークン
- 最小値: 1トークン

#### estimate_cost() (L214-220)

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

## デコレータ

### safe_execute (L226-238)

```python
def safe_execute(func):
    """安全実行デコレータ"""
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

**特徴**:
- エラーを自動キャッチ
- ログ記録
- Streamlitエラー表示
- `None`返却

---

## UI関数

### select_model() (L244-262)

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

### show_model_info() (L265-313)

選択されたモデルの詳細情報を表示。

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

### estimate_token_usage() (L316-351)

```python
def estimate_token_usage(df_processed: pd.DataFrame, selected_model: str) -> None:
    """処理済みデータのトークン使用量推定"""
```

**表示内容**:
- 推定総トークン数
- 平均トークン/レコード
- 推定embedding費用

---

## データ処理関数

### clean_text() (L357-378)

```python
def clean_text(text: str) -> str:
    """テキストのクレンジング処理"""
```

**処理内容**:
1. 改行を空白に置換（`\n`, `\r`）
2. 連続した空白を1つにまとめる
3. 先頭・末尾の空白を除去
4. 引用符の正規化（`"` → `"`, `'` → `'`）

### combine_columns() (L381-415)

```python
def combine_columns(row: pd.Series, dataset_type: str) -> str:
    """複数列を結合して1つのテキストにする（データセット対応）"""
```

**特徴**:
- データセット設定に基づく列結合
- 医療QA（medical_qa）の特別処理
- 大文字小文字を考慮した列名マッピング

### validate_data() (L418-467)

```python
def validate_data(df: pd.DataFrame, dataset_type: str = None) -> List[str]:
    """データの検証"""
```

**検証内容**:
- 基本統計（総行数、総列数）
- 必須列の確認（部分一致も許可）
- 各列の空値確認
- 重複行の確認

### load_dataset() (L470-480)

```python
@safe_execute
def load_dataset(uploaded_file, dataset_type: str = None) -> Tuple[pd.DataFrame, List[str]]:
    """データセットの読み込みと基本検証"""
```

### process_rag_data() (L483-529)

```python
@safe_execute
def process_rag_data(df: pd.DataFrame, dataset_type: str, combine_columns_option: bool = True) -> pd.DataFrame:
    """RAGデータの前処理を実行"""
```

**処理内容**:
1. 重複行の除去
2. 空行の除去（全列がNAの行）
3. インデックスのリセット
4. 各列のクレンジング
5. 列の結合（オプション）→ `Combined_Text`列作成
6. 空の結合テキスト除去

### create_download_data() (L532-551)

```python
@safe_execute
def create_download_data(df: pd.DataFrame, include_combined: bool = True, dataset_type: str = None) -> Tuple[str, Optional[str]]:
    """ダウンロード用データの作成"""
```

**戻り値**:
- `csv_data`: CSV形式の文字列
- `text_data`: 結合テキストのみ（改行区切り）

### display_statistics() (L554-590)

```python
def display_statistics(df_original: pd.DataFrame, df_processed: pd.DataFrame, dataset_type: str = None) -> None:
    """処理前後の統計情報を表示"""
```

**表示内容**:
- 元の行数 / 処理後の行数 / 除去された行数
- 結合後テキスト分析（平均/最大/最小文字数）
- 文字数分布（25%/50%/75%点）

---

## ファイル保存関数

### create_output_directory() (L596-617)

```python
def create_output_directory() -> Path:
    """OUTPUTディレクトリの作成"""
```

**特徴**:
- `OUTPUT`ディレクトリ作成
- 書き込み権限テスト実行
- 権限不足時は`PermissionError`発生

### save_files_to_output() (L620-673)

```python
@safe_execute
def save_files_to_output(df_processed, dataset_type: str, csv_data: str, text_data: str = None) -> Dict[str, str]:
    """処理済みデータをOUTPUTフォルダに保存"""
```

**保存ファイル**:

| ファイル種類 | ファイル名 | 内容 |
|-------------|-----------|------|
| CSV | `preprocessed_{dataset_type}.csv` | 前処理済みデータ |
| テキスト | `{dataset_type}.txt` | 結合テキストのみ |
| メタデータ | `metadata_{dataset_type}.json` | 処理情報 |

**メタデータ構造**:
```python
{
    "dataset_type": "...",
    "processed_rows": 100,
    "processing_timestamp": "20251127_120000",
    "created_at": "2025-11-27T12:00:00",
    "files_created": ["csv", "txt", "metadata"],
    "processing_info": {
        "original_rows": 120,
        "removed_rows": 20
    }
}
```

---

## ページ設定関数

### show_usage_instructions() (L679-745)

```python
def show_usage_instructions(dataset_type: str) -> None:
    """使用方法の説明を表示（データセット別対応）"""
```

**表示内容**:
- 基本的な前処理手順
- RAG最適化の特徴
- 推奨モデル
- データセット特有の説明

### setup_page_config() (L751-763)

```python
def setup_page_config(dataset_type: str) -> None:
    """ページ設定の初期化"""
```

### setup_page_header() (L766-772)

```python
def setup_page_header(dataset_type: str) -> None:
    """ページヘッダーの設定"""
```

### setup_sidebar_header() (L775-780)

```python
def setup_sidebar_header(dataset_type: str) -> None:
    """サイドバーヘッダーの設定"""
```

---

## データフロー

### 基本処理フロー

```
CSVアップロード
    ↓
load_dataset()
    ↓
validate_data()
    ↓
process_rag_data()
    ├── 重複行除去
    ├── 空行除去
    ├── clean_text()
    └── combine_columns() → Combined_Text
    ↓
display_statistics()
    ↓
create_download_data()
    ↓
save_files_to_output()
```

### データセット別結合処理

| データセット | 結合フォーマット |
|-------------|-----------------|
| customer_support_faq | question + answer |
| medical_qa | Question + Complex_CoT + Response |
| sciq_qa | question + correct_answer |
| legal_qa | question + answer |
| trivia_qa | question + answer + entity_pages + search_results |

---

## 使用例

### 例1: モデル選択と情報表示

```python
import streamlit as st
from helper_rag import select_model, show_model_info

# モデル選択
selected_model = select_model(key="my_model")

# モデル情報表示
show_model_info(selected_model)
```

### 例2: データ読み込みと処理

```python
from helper_rag import load_dataset, process_rag_data, create_download_data

# データ読み込み
df, validation_results = load_dataset(uploaded_file, dataset_type="medical_qa")

# 前処理実行
df_processed = process_rag_data(df, dataset_type="medical_qa", combine_columns_option=True)

# ダウンロード用データ作成
csv_data, text_data = create_download_data(df_processed, include_combined=True)
```

### 例3: ファイル保存

```python
from helper_rag import save_files_to_output

# ファイル保存
saved_files = save_files_to_output(
    df_processed,
    dataset_type="customer_support_faq",
    csv_data=csv_data,
    text_data=text_data
)

print(f"保存ファイル: {saved_files}")
# {'csv': 'OUTPUT/preprocessed_customer_support_faq.csv',
#  'txt': 'OUTPUT/customer_support_faq.txt',
#  'metadata': 'OUTPUT/metadata_customer_support_faq.json'}
```

### 例4: トークン管理

```python
from helper_rag import TokenManager

# トークン数計算
text = "これはテストテキストです"
token_count = TokenManager.count_tokens(text)

# コスト推定
cost = TokenManager.estimate_cost(
    input_tokens=1000,
    output_tokens=500,
    model="gpt-5-mini"
)
print(f"推定コスト: ${cost:.4f}")
```

### 例5: ページ設定

```python
from helper_rag import setup_page_config, setup_page_header, setup_sidebar_header

dataset_type = "medical_qa"

# ページ設定
setup_page_config(dataset_type)
setup_page_header(dataset_type)
setup_sidebar_header(dataset_type)
```

---

## エクスポート定義

```python
__all__ = [
    # 設定クラス
    'AppConfig',
    'RAGConfig',
    'TokenManager',

    # デコレータ
    'safe_execute',

    # UI関数
    'select_model',
    'show_model_info',
    'estimate_token_usage',

    # データ処理関数
    'clean_text',
    'combine_columns',
    'validate_data',
    'load_dataset',
    'process_rag_data',
    'create_download_data',
    'display_statistics',

    # ファイル保存関数
    'create_output_directory',
    'save_files_to_output',

    # 使用方法・ページ設定関数
    'show_usage_instructions',
    'setup_page_config',
    'setup_page_header',
    'setup_sidebar_header',
]
```

---

## RAG使用推奨モデル

### 高速・低コスト（推奨）

| モデル | 入力/出力 | 特徴 |
|--------|----------|------|
| gpt-5-mini | $0.0001 / $0.0004 | デフォルト推奨 |
| gpt-4o-mini | $0.00015 / $0.0006 | RAG最適 |
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

## 制限事項

1. **トークン推定**: 簡易推定のため、正確な値が必要な場合はtiktoken使用
2. **文字エンコーディング**: UTF-8前提
3. **メモリ制限**: 大規模データセットは分割処理推奨
4. **列名の大文字小文字**: 区別なし（部分一致対応）
5. **OUTPUT権限**: 書き込み権限必須

---

## トラブルシューティング

### 問題1: 必須列が見つからない

**症状**: "必須列が不足"エラー

**解決策**:
- 列名の確認（部分一致で検索される）
- CSVファイルのエンコーディング確認（UTF-8推奨）

### 問題2: OUTPUT書き込みエラー

**症状**: PermissionError

**解決策**:
```bash
# ディレクトリの権限確認
ls -la OUTPUT

# 権限付与
chmod 755 OUTPUT
```

### 問題3: トークン数推定誤差

**症状**: 実際のトークン数と推定値の乖離

**解決策**:
- 簡易推定のため、正確な値が必要な場合はtiktoken利用
- helper_api.pyのTokenManager使用を検討

---

## 注意事項（CLAUDE.mdより）

1. **モデル名**: config.pyで定義されたモデル名をそのまま使用すること。マッピングを作成しないこと。

2. **GPT-5シリーズ、O-Series**: 全て実在するモデル。名前を変更しないこと。

---

作成日: 2024-10-29
更新日: 2025-11-27
作成者: OpenAI RAG Q/A JP Development Team
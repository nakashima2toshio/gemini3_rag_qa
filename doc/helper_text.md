# helper_text.py 仕様書

作成日: 2025-11-27

## 概要

テキスト処理のためのユーティリティモジュール。テキストのクレンジング、トークン処理、チャンク分割、テキスト分析などの共通機能を提供。

## ファイル情報

- **ファイル名**: helper_text.py
- **行数**: 559行
- **主な機能**: テキストクレンジング、トークン処理、チャンク分割
- **依存ライブラリ**: tiktoken、re、logging

## 使用箇所

- `rag_qa_pair_qdrant.py`
- `a02_make_qa_para.py`
- `helper_rag.py`
- `helper_rag_qa.py`

---

## アーキテクチャ

### モジュール構造

```
helper_text.py
├── インポート・ログ設定 (L1-21)
│
├── 定数 (L23-33)
│   ├── DEFAULT_CHUNK_SIZE = 300
│   ├── DEFAULT_CHUNK_OVERLAP = 50
│   ├── DEFAULT_MIN_CHUNK_SIZE = 50
│   └── DEFAULT_ENCODING = "cl100k_base"
│
├── テキストクレンジング関数 (L36-151)
│   ├── clean_text() (L40-80)
│   ├── normalize_japanese_text() (L83-115)
│   └── extract_sentences_japanese() (L118-151)
│
├── トークン処理関数 (L154-220)
│   ├── get_encoding() (L158-168)
│   ├── count_tokens() (L171-198)
│   └── estimate_tokens_simple() (L201-220)
│
├── チャンク分割関数 (L223-390)
│   ├── split_into_chunks() (L227-270)
│   ├── split_into_chunks_with_metadata() (L273-345)
│   └── merge_small_chunks() (L348-390)
│
├── テキスト分析関数 (L393-476)
│   ├── analyze_text_complexity() (L397-440)
│   └── extract_key_concepts() (L443-476)
│
├── ユーティリティ関数 (L479-550)
│   ├── truncate_text() (L483-510)
│   └── get_text_stats() (L513-550)
│
└── 後方互換性エイリアス (L553-559)
    ├── count_tokens_tiktoken = count_tokens
    └── split_text_into_chunks = split_into_chunks
```

---

## 定数

### デフォルト設定 (L27-33)

| 定数 | 値 | 説明 |
|------|-----|------|
| `DEFAULT_CHUNK_SIZE` | 300 | チャンクサイズ（トークン数） |
| `DEFAULT_CHUNK_OVERLAP` | 50 | オーバーラップトークン数 |
| `DEFAULT_MIN_CHUNK_SIZE` | 50 | 最小チャンクサイズ |
| `DEFAULT_ENCODING` | "cl100k_base" | デフォルトエンコーディング |

---

## テキストクレンジング関数

### clean_text() (L40-80)

テキストのクレンジング処理。

```python
def clean_text(text: str) -> str:
    """
    テキストのクレンジング処理

    Args:
        text: クレンジング対象のテキスト

    Returns:
        クレンジング済みのテキスト
    """
```

**処理内容**:
1. None/NA/空値チェック
2. 文字列変換
3. 改行を空白に置換（`\n`, `\r`）
4. 連続した空白を1つにまとめる
5. 先頭・末尾の空白を除去
6. 引用符の正規化（`"` → `"`, `'` → `'`）

**特徴**:
- pandas NAチェック対応
- イテラブルオブジェクトのチェック

---

### normalize_japanese_text() (L83-115)

日本語テキストの正規化。

```python
def normalize_japanese_text(text: str) -> str:
    """
    日本語テキストの正規化

    Args:
        text: 正規化対象のテキスト

    Returns:
        正規化済みのテキスト
    """
```

**処理内容**:
1. 全角英数字を半角に変換（`Ａ` → `A`）
2. 全角スペースを半角に変換
3. 連続する句読点の正規化（`。。` → `。`）

**文字コード変換**:
| 変換前 | 変換後 |
|--------|--------|
| `Ａ-Ｚ` (0xFF21-0xFF3A) | `A-Z` (0x0041-0x005A) |
| `ａ-ｚ` (0xFF41-0xFF5A) | `a-z` (0x0061-0x007A) |
| `０-９` (0xFF10-0xFF19) | `0-9` (0x0030-0x0039) |
| `　` (0x3000) | ` ` (0x0020) |

---

### extract_sentences_japanese() (L118-151)

日本語テキストから文を抽出。

```python
def extract_sentences_japanese(text: str) -> List[str]:
    """
    日本語テキストから文を抽出

    Args:
        text: 対象テキスト

    Returns:
        文のリスト
    """
```

**文末パターン**: `。！？!?`

**処理フロー**:
1. 文末パターンで分割
2. 文末記号を前の文に結合
3. 空文を除外

---

## トークン処理関数

### get_encoding() (L158-168)

tiktokenエンコーディングを取得。

```python
def get_encoding(encoding_name: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    """
    tiktokenエンコーディングを取得

    Args:
        encoding_name: エンコーディング名

    Returns:
        tiktokenエンコーディング
    """
```

---

### count_tokens() (L171-198)

テキストのトークン数をカウント。

```python
def count_tokens(text: str, model: str = None) -> int:
    """
    テキストのトークン数をカウント

    Args:
        text: 対象テキスト
        model: モデル名（指定がなければデフォルトエンコーディング使用）

    Returns:
        トークン数
    """
```

**処理フロー**:
1. モデル指定あり → `tiktoken.encoding_for_model(model)`
2. モデル指定なし → `tiktoken.get_encoding(DEFAULT_ENCODING)`
3. エラー時 → `estimate_tokens_simple()`にフォールバック

---

### estimate_tokens_simple() (L201-220)

簡易的なトークン数推定。

```python
def estimate_tokens_simple(text: str) -> int:
    """
    簡易的なトークン数推定

    日本語文字は約0.5トークン、英数字は約0.25トークンとして推定

    Args:
        text: 対象テキスト

    Returns:
        推定トークン数
    """
```

**推定ルール**:
| 文字種 | 係数 |
|--------|------|
| 日本語（ord > 127） | 0.5 |
| 英数字 | 0.25 |

**最小値**: 1トークン

---

## チャンク分割関数

### split_into_chunks() (L227-270)

テキストをチャンクに分割。

```python
def split_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    encoding_name: str = DEFAULT_ENCODING
) -> List[str]:
    """
    テキストをチャンクに分割

    Args:
        text: 分割するテキスト
        chunk_size: チャンクサイズ（トークン数）
        overlap: オーバーラップ（トークン数）
        encoding_name: エンコーディング名

    Returns:
        チャンクのリスト
    """
```

**処理フロー**:
```
テキスト
    ↓
トークン化（tiktoken）
    ↓
チャンクサイズ以下？
    ├── YES → そのまま返す
    └── NO → 分割処理
                ↓
            オーバーラップ付きで分割
                ↓
            チャンクリスト
```

---

### split_into_chunks_with_metadata() (L273-345)

テキストをメタデータ付きチャンクに分割。

```python
def split_into_chunks_with_metadata(
    text: str,
    doc_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    encoding_name: str = DEFAULT_ENCODING
) -> List[dict]:
    """
    テキストをメタデータ付きチャンクに分割

    Args:
        text: 分割するテキスト
        doc_id: ドキュメントID
        chunk_size: チャンクサイズ（トークン数）
        overlap: オーバーラップ（トークン数）
        encoding_name: エンコーディング名

    Returns:
        チャンクデータのリスト
    """
```

**戻り値の構造**:
```python
{
    "id": "doc_001_chunk_0",
    "text": "チャンクテキスト",
    "tokens": 150,
    "doc_id": "doc_001",
    "chunk_idx": 0,
    "position": "start"  # "start", "middle", "end", "full"
}
```

**positionの値**:
| 値 | 説明 |
|----|------|
| `full` | テキスト全体が1チャンク |
| `start` | 最初のチャンク |
| `middle` | 中間のチャンク |
| `end` | 最後のチャンク |

---

### merge_small_chunks() (L348-390)

小さいチャンクを統合。

```python
def merge_small_chunks(
    chunks: List[dict],
    min_tokens: int = DEFAULT_MIN_CHUNK_SIZE,
    max_tokens: int = DEFAULT_CHUNK_SIZE * 2
) -> List[dict]:
    """
    小さいチャンクを統合

    Args:
        chunks: チャンクデータのリスト
        min_tokens: 統合対象の最小トークン数
        max_tokens: 統合後の最大トークン数

    Returns:
        統合後のチャンクリスト
    """
```

**統合条件**:
1. 現在のチャンクが`min_tokens`未満
2. 統合後が`max_tokens`以下
3. 同じドキュメントのチャンク

**統合後のID形式**: `{doc_id}_merged_{chunk_idx1}_{chunk_idx2}`

---

## テキスト分析関数

### analyze_text_complexity() (L397-440)

テキストの複雑度を分析。

```python
def analyze_text_complexity(text: str) -> dict:
    """
    テキストの複雑度を分析

    Args:
        text: 分析対象テキスト

    Returns:
        複雑度分析結果
    """
```

**戻り値の構造**:
```python
{
    "complexity_level": "low",  # "low", "medium", "high"
    "sentence_count": 5,
    "avg_sentence_length": 25.5,
    "token_count": 128,
    "technical_terms": []
}
```

**複雑度判定基準**:
| 複雑度 | 条件 |
|--------|------|
| high | avg_sentence_length > 50 または token_count > 500 |
| medium | avg_sentence_length > 25 または token_count > 200 |
| low | それ以外 |

---

### extract_key_concepts() (L443-476)

テキストからキーコンセプトを抽出（簡易版）。

```python
def extract_key_concepts(text: str, max_concepts: int = 5) -> List[str]:
    """
    テキストからキーコンセプトを抽出（簡易版）

    Args:
        text: 対象テキスト
        max_concepts: 最大抽出数

    Returns:
        キーコンセプトのリスト
    """
```

**抽出パターン**: `[一-龯々ぁ-んァ-ン]{2,}` （漢字・ひらがな・カタカナの2文字以上の連続）

---

## ユーティリティ関数

### truncate_text() (L483-510)

テキストを指定トークン数で切り詰め。

```python
def truncate_text(text: str, max_tokens: int = 1000, add_ellipsis: bool = True) -> str:
    """
    テキストを指定トークン数で切り詰め

    Args:
        text: 対象テキスト
        max_tokens: 最大トークン数
        add_ellipsis: 省略記号を追加するか

    Returns:
        切り詰められたテキスト
    """
```

**特徴**:
- トークンベースの切り詰め
- オプションで省略記号（`...`）追加

---

### get_text_stats() (L513-550)

テキストの統計情報を取得。

```python
def get_text_stats(text: str) -> dict:
    """
    テキストの統計情報を取得

    Args:
        text: 対象テキスト

    Returns:
        統計情報
    """
```

**戻り値の構造**:
```python
{
    "char_count": 500,
    "token_count": 128,
    "word_count": 50,
    "sentence_count": 5,
    "avg_word_length": 10.0
}
```

---

## 後方互換性エイリアス (L553-559)

旧名称でのインポートをサポート。

```python
count_tokens_tiktoken = count_tokens
split_text_into_chunks = split_into_chunks
```

---

## 使用例

### 例1: テキストクレンジング

```python
from helper_text import clean_text, normalize_japanese_text

# 基本的なクレンジング
raw_text = "  Hello\n\nWorld  "
cleaned = clean_text(raw_text)
print(cleaned)  # "Hello World"

# 日本語正規化
japanese_text = "Ｈｅｌｌｏ　Ｗｏｒｌｄ"
normalized = normalize_japanese_text(japanese_text)
print(normalized)  # "Hello World"
```

### 例2: トークン処理

```python
from helper_text import count_tokens, estimate_tokens_simple

text = "これはテストテキストです"

# tiktokenによるカウント
token_count = count_tokens(text)
print(f"トークン数: {token_count}")

# 簡易推定
estimated = estimate_tokens_simple(text)
print(f"推定トークン数: {estimated}")
```

### 例3: チャンク分割

```python
from helper_text import split_into_chunks, split_into_chunks_with_metadata

long_text = "..." # 長いテキスト

# シンプルな分割
chunks = split_into_chunks(long_text, chunk_size=200, overlap=30)
print(f"チャンク数: {len(chunks)}")

# メタデータ付き分割
chunks_with_meta = split_into_chunks_with_metadata(
    long_text,
    doc_id="doc_001",
    chunk_size=200,
    overlap=30
)

for chunk in chunks_with_meta:
    print(f"ID: {chunk['id']}, トークン: {chunk['tokens']}, 位置: {chunk['position']}")
```

### 例4: 小さいチャンクの統合

```python
from helper_text import split_into_chunks_with_metadata, merge_small_chunks

chunks = split_into_chunks_with_metadata(text, doc_id="doc_001")
merged = merge_small_chunks(chunks, min_tokens=50, max_tokens=400)

print(f"統合前: {len(chunks)}チャンク")
print(f"統合後: {len(merged)}チャンク")
```

### 例5: テキスト分析

```python
from helper_text import analyze_text_complexity, get_text_stats

text = "日本語のテキストを分析します。複雑度を判定します。"

# 複雑度分析
complexity = analyze_text_complexity(text)
print(f"複雑度: {complexity['complexity_level']}")
print(f"文数: {complexity['sentence_count']}")

# 統計情報
stats = get_text_stats(text)
print(f"文字数: {stats['char_count']}")
print(f"トークン数: {stats['token_count']}")
```

### 例6: テキスト切り詰め

```python
from helper_text import truncate_text

long_text = "非常に長いテキスト..." * 100

# 100トークンに切り詰め
truncated = truncate_text(long_text, max_tokens=100, add_ellipsis=True)
print(truncated)
```

---

## 依存ライブラリ

```python
import re
import logging
from typing import List
import tiktoken
```

---

## 注意事項

1. **tiktokenの必要性**: トークン処理には`tiktoken`が必須
2. **エンコーディング**: デフォルトは`cl100k_base`（GPT-4、GPT-3.5-turbo用）
3. **日本語処理**: 形態素解析器（MeCab等）は未使用（簡易版）
4. **pandas依存**: `clean_text()`でのNAチェックにpandas使用（オプショナル）

---

## トラブルシューティング

### 問題1: tiktoken import error

**症状**: `ModuleNotFoundError: No module named 'tiktoken'`

**解決策**:
```bash
pip install tiktoken
```

### 問題2: エンコーディングエラー

**症状**: `KeyError: model not found`

**解決策**: モデル名が正しいか確認、またはデフォルトエンコーディング使用

### 問題3: 簡易推定の精度

**症状**: 実際のトークン数と推定値に乖離

**解決策**: `count_tokens()`を使用（tiktokenベース）

---

## エクスポート一覧

```python
# テキストクレンジング
clean_text
normalize_japanese_text
extract_sentences_japanese

# トークン処理
get_encoding
count_tokens
estimate_tokens_simple

# チャンク分割
split_into_chunks
split_into_chunks_with_metadata
merge_small_chunks

# テキスト分析
analyze_text_complexity
extract_key_concepts

# ユーティリティ
truncate_text
get_text_stats

# 後方互換性エイリアス
count_tokens_tiktoken  # = count_tokens
split_text_into_chunks  # = split_into_chunks

# 定数
DEFAULT_CHUNK_SIZE
DEFAULT_CHUNK_OVERLAP
DEFAULT_MIN_CHUNK_SIZE
DEFAULT_ENCODING
```

---

## まとめ

helper_text.pyは、テキスト処理のための包括的なユーティリティモジュールです。

### 主要な特徴

1. **テキストクレンジング**: 改行除去、空白正規化、引用符正規化
2. **日本語対応**: 全角半角変換、句読点正規化、文抽出
3. **トークン処理**: tiktokenベースの正確なカウント、簡易推定
4. **チャンク分割**: オーバーラップ対応、メタデータ付与、統合機能
5. **テキスト分析**: 複雑度判定、キーコンセプト抽出

### 推奨用途

- RAGシステムのテキスト前処理
- ベクトルストア用のチャンク分割
- トークン数の事前確認
- テキストの品質分析

---

作成日: 2025-11-27
作成者: OpenAI RAG Q/A JP Development Team
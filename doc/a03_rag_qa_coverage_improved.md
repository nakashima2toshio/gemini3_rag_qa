# a03_rag_qa_coverage_improved.py - 高カバレッジ・多様性Q/A生成システム

作成日: 2025-11-08 (最終更新: Gemini移行対応)

## 目次

1. [概要](#1-概要)
2. [アーキテクチャ](#2-アーキテクチャ)
3. [キーワード抽出](#3-キーワード抽出)
4. [階層化質問タイプ](#4-階層化質問タイプ)
5. [チャンク複雑度分析](#5-チャンク複雑度分析)
6. [ドメイン適適応戦略](#6-ドメイン適応戦略)
7. [Q/A生成戦略](#7-qa生成戦略)
8. [カバレージ分析](#8-カバレージ分析)
9. [コマンドラインオプション](#9-コマンドラインオプション)
10. [実行方法](#10-実行方法)
11. [出力ファイル](#11-出力ファイル)
12. [トラブルシューティング](#12-トラブルシューティング)

---

## 1. 概要

### 1.1 目的

`a03_rag_qa_coverage_improved.py`は、**95%以上のカバレッジ**と**質問タイプの多様性**を最優先するQ/A生成システムです。
「基礎」「理解」「応用」といった階層化された質問タイプ定義に基づき、テンプレートベースの手法とLLM (`gemini-2.0-flash`) をハイブリッドに組み合わせることで、ドキュメントの情報を余すことなくQ/A化します。

> **【使い分け】**
> *   **効率・速度重視**: [a10_qa_optimized_hybrid_batch.py](a10_qa_optimized_hybrid_batch.md) を使用（通常はこちらで十分な品質が得られます）。
> *   **品質・網羅性重視**: 本スクリプト (`a03`) を使用。教育用コンテンツ作成や、情報の取りこぼしが許されない高精度な検索システム構築に適しています。

### 1.2 起動コマンド

```bash
python a03_rag_qa_coverage_improved.py \
  --input OUTPUT/preprocessed_cc_news.csv \
  --dataset cc_news \
  --analyze-coverage \
  --coverage-threshold 0.60 \
  --qa-per-chunk 10 \
  --max-chunks 2000 \
  --model gemini-2.0-flash
```

### 1.3 主要機能

- **セマンティックチャンク分割**: 段落優先のセマンティック分割（`helper_rag_qa.py`使用、`gemini-embedding-001`利用）
- **階層化質問タイプ**: 3階層11タイプの質問タイプ定義
- **チャンク複雑度分析**: 専門用語密度、平均文長（`UnifiedLLMClient`による正確なトークンカウント）、統計情報検出
- **ドメイン適応戦略**: データセット別の最適化戦略
- **包括的Q/A生成**: ルール/テンプレート/LLM (`gemini-2.0-flash`) のハイブリッド生成
- **バッチ処理による埋め込み生成**: Gemini APIのバッチ処理で高速化
- **改良版カバレッジ分析**: 3段階の分布評価（高・中・低）、`gemini-embedding-001`を使用

### 1.4 対応データセット

| データセット | キー | 言語 | 説明 |
|------------|------|------|------|
| CC-News | `cc_news` | 英語 | 英語ニュース記事 |
| CC100日本語 | `japanese_text` | 日本語 | Webテキストコーパス |
| Wikipedia日本語版 | `wikipedia_ja` | 日本語 | 百科事典的知識 |
| Livedoorニュース | `livedoor` | 日本語 | ニュースコーパス |

### 1.5 a02_make_qa_para.pyとの比較

| 項目 | a02（LLM版） | a03（ハイブリッド版） |
|------|-------------|---------------------|
| **主な目的** | LLMで高品質Q/A生成 | カバレッジ最大化と多様性確保 |
| **Q/A生成手法** | LLM (`gemini-2.0-flash`) | ルール + テンプレート + LLM (Gemini) |
| **コスト** | 中程度 | 低〜中（テンプレート主体で調整可能） |
| **生成速度** | 中速 | 高速（ルールベース部分）〜中速 |
| **カバレッジ** | 90-95% | **95%+** |
| **Q/A品質** | 非常に高い | 高い（多様なタイプを網羅） |

---

## 2. アーキテクチャ

### 2.1 システム構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                a03_rag_qa_coverage_improved.py                  │
├─────────────────────────────────────────────────────────────────┤
│  [1] データ読み込み                                              │
│      load_input_data()                                          │
│                              │                                  │
│                              ▼                                  │
│  [2] セマンティックチャンク分割                                   │
│      SemanticCoverage.create_semantic_chunks()                  │
│      (gemini-embedding-001利用)                                 │
│                              │                                  │
│                              ▼                                  │
│  [3] Q/A生成                                                    │
│      ├── generate_llm_qa() (gemini-2.0-flash利用)                │
│      └── generate_comprehensive_qa_for_chunk()（テンプレート）    │
│                              │                                  │
│                              ▼                                  │
│  [4] カバレッジ分析                                              │
│      calculate_improved_coverage()                              │
│      (gemini-embedding-001利用)                                 │
│                              │                                  │
│                              ▼                                  │
│  [5] 結果保存                                                    │
│      save_results() → qa_output/a03/                           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 依存モジュール

```python
from helper_rag_qa import SemanticCoverage
from helper_llm import create_llm_client, LLMClient
from models import QAPairsResponse

import pandas as pd
import numpy as np
from collections import Counter
```

### 2.3 データセット設定

```python
DATASET_CONFIGS = {
    "cc_news": {
        "name": "CC-News英語ニュース",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "en"
    },
    "japanese_text": {
        "name": "日本語Webテキスト",
        "text_column": "Combined_Text",
        "title_column": None,
        "lang": "ja"
    },
    "wikipedia_ja": {
        "name": "Wikipedia日本語版",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja"
    },
    "livedoor": {
        "name": "ライブドアニュース",
        "text_column": "Combined_Text",
        "title_column": "title",
        "category_column": "category",
        "lang": "ja"
    }
}
```

---

## 3. キーワード抽出

### 3.1 KeywordExtractorクラス

MeCabと正規表現を統合したキーワード抽出クラスです。

```python
class KeywordExtractor:
    """
    MeCabが利用可能な場合は複合名詞抽出を優先し、
    利用不可の場合は正規表現版に自動フォールバック
    """
    # ... (省略) 
```

### 3.2 ストップワード

```python
self.stopwords = {
    'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん',
    'ます', 'です', 'ある', 'いる', 'する', 'なる', 'できる',
    'いう', '的', 'な', 'に', 'を', 'は', 'が', 'で', 'と',
    'の', 'から', 'まで', '等', 'など', 'よる', 'おく', 'くる'
}
```

### 3.3 正規表現パターン

```python
# カタカナ語、漢字複合語、英数字を抽出
pattern = r'[ァ-ヴー]{2,}|[一-龥]{2,}|[A-Za-z]{2,}[A-Za-z0-9]*'
```

### 3.4 シングルトンインスタンス

```python
def get_keyword_extractor() -> KeywordExtractor:
    """KeywordExtractorのシングルトンインスタンスを取得"""
    # ...
```

---

## 4. 階層化質問タイプ

### 4.1 3階層11タイプの定義

`basic`, `understanding`, `application` の3階層で、合計11種類の質問タイプを定義しています。

### 4.2 階層別の特徴

| 階層 | タイプ数 | 目的 | 難易度 |
|------|---------|------|--------|
| basic | 3 | 基本的な事実確認 | 低 |
| understanding | 4 | 概念の理解・関係性 | 中 |
| application | 4 | 応用・実践・評価 | 高 |

---

## 5. チャンク複雑度分析

### 5.1 analyze_chunk_complexity関数

`UnifiedLLMClient` (Gemini API) を使用して正確なトークンカウントを行い、チャンクの複雑度を分析します。

```python
def analyze_chunk_complexity(chunk_text: str, lang: str = "auto") -> Dict:
    """
    チャンクの複雑度を分析して、適切なQ/A生成戦略を決定
    
    UnifiedLLMClient.count_tokens を使用してトークン数を取得
    """
    # ...
```

### 5.2 複雑度レベル判定

| レベル | スコア | 条件 |
|--------|--------|------|
| high | >= 5 | 概念密度 > 5% または 平均文長 > 30 |
| medium | >= 3 | 概念密度 > 2% または 平均文長 > 20 |
| low | < 3 | その他 |

---

## 6. ドメイン適応戦略

データセット（ドメイン）ごとに、生成する質問タイプの重点や避けるべきタイプを定義しています。

---

## 7. Q/A生成戦略

### 7.1 LLMベースQ/A生成（generate_llm_qa）

`UnifiedLLMClient.generate_structured` を使用して、Geminiモデル (`gemini-2.0-flash`等) から高品質なQ/Aペアを生成します。Pydanticモデル (`QAPairsResponse`) を利用して、出力の構造を保証します。

```python
def generate_llm_qa(chunk_text: str, chunk_idx: int, model: str, qa_per_chunk: int = 2) -> List[Dict]:
    """
    LLMを使用して高品質なQ/Aペアを生成
    UnifiedLLMClient.generate_structured を利用
    """
    # ...
```

### 7.2 包括的Q/A生成（generate_comprehensive_qa_for_chunk）

テンプレートベースで5つの異なる戦略（包括的、詳細事実、文脈、キーワード、テーマ）を用いてQ/Aを生成します。

---

## 8. カバレッジ分析

### 8.1 改良版カバレッジ計算

`gemini-embedding-001` を使用して、チャンクと生成されたQ/Aペアの埋め込みベクトルを生成し、コサイン類似度でカバレッジを計算します。

### 8.2 バッチ処理による埋め込み生成

Gemini APIの制限を考慮し、バッチサイズを調整して埋め込みを生成します。

```python
MAX_BATCH_SIZE = 100 # Gemini APIの安全なバッチサイズ

# ... バッチ処理ロジック ...
```

### 8.3 カバレッジ結果の構造と分布評価

カバレッジ率、閾値ごとの分布（高・中・低）を含む詳細な分析結果を提供します。

---

## 9. コマンドラインオプション

### 9.1 全オプション一覧

| オプション | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `--input` | str | 必須 | 入力ファイルパス |
| `--dataset` | str | 必須 | データセットタイプ |
| `--max-docs` | int | None | 処理する最大文書数 |
| `--methods` | str[] | ['rule', 'template'] | 使用する手法 (llmを追加可能) |
| `--model` | str | **`gemini-2.0-flash`** | 使用するGeminiモデル |
| `--output` | str | qa_output | 出力ディレクトリ |
| `--analyze-coverage` | flag | False | カバレッジ分析を実行 |
| `--coverage-threshold` | float | 0.65 | カバレッジ判定閾値 |
| `--qa-per-chunk` | int | 4 | チャンクあたりのQ/A数 |
| `--max-chunks` | int | 300 | 処理する最大チャンク数 |
| `--demo` | flag | False | デモモード |

### 9.2 手法オプション

```bash
--methods rule template    # デフォルト（ルール+テンプレート）
--methods rule template llm  # LLM (Gemini) も追加（高品質化）
```

---

## 10. 実行方法

### 10.1 テスト実行（小規模）

```bash
python a03_rag_qa_coverage_improved.py \
  --input OUTPUT/preprocessed_cc_news.csv \
  --dataset cc_news \
  --max-docs 150 \
  --qa-per-chunk 4 \
  --max-chunks 609 \
  --analyze-coverage \
  --coverage-threshold 0.60 \
  --model gemini-2.0-flash
```

### 10.2 推奨実行（中規模）

```bash
python a03_rag_qa_coverage_improved.py \
  --input OUTPUT/preprocessed_cc_news.csv \
  --dataset cc_news \
  --qa-per-chunk 10 \
  --max-chunks 2000 \
  --analyze-coverage \
  --coverage-threshold 0.60 \
  --methods rule template llm
```

### 10.3 実行時間の見積もり

| 設定 | 文書数 | チャンク数 | Q/A数 | 実行時間 | コスト |
|------|--------|----------|-------|---------|--------|
| テスト | 150 | 609 | 7,308 | 2分 | 低 |
| 推奨 | 自動 | 2,000 | 20,000 | 8-10分 | 低〜中 |
| 全文書 | 7,499 | 18,000 | 144,000 | 60-90分 | 中 |

※ Gemini 2.0 Flash は非常に安価かつ高速なため、OpenAIモデルと比較してコストパフォーマンスが良いです。

---

## 11. 出力ファイル

### 11.1 出力ディレクトリ構造

```
qa_output/a03/
├── qa_pairs_{dataset}_{timestamp}.json    # Q/Aペア（JSON）
├── qa_pairs_{dataset}_{timestamp}.csv     # Q/Aペア（CSV全カラム）
├── coverage_{dataset}_{timestamp}.json    # カバレッジ分析結果
└── summary_{dataset}_{timestamp}.json     # サマリー情報

qa_output/
└── a03_qa_pairs_{dataset}.csv             # 統一フォーマット（question/answerのみ）
```

---

## 12. トラブルシューティング

### 12.1 APIキーエラー

**症状**: `GOOGLE_API_KEYが設定されていません`

**解決策**:
```bash
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```

### 12.2 ファイルが見つからない

**症状**: `FileNotFoundError: 入力ファイルが見つかりません`

**解決策**:
```bash
ls OUTPUT/preprocessed_cc_news.csv
```

### 12.3 カバレッジが低い

**症状**: カバレッジ率が70%未満

**解決策**:
```bash
# Q/A数を増やす
python a03_rag_qa_coverage_improved.py --qa-per-chunk 10

# LLM手法を追加して多様性を高める
python a03_rag_qa_coverage_improved.py --methods rule template llm

# 閾値を下げる（必要に応じて）
python a03_rag_qa_coverage_improved.py --coverage-threshold 0.55
```

### 12.4 MeCabが利用できない

**症状**: `⚠️ MeCabが利用できません（正規表現モード）`

**解決策**: MeCabをインストールしてください（`doc/01_install.md`参照）。

---

## 付録: 品質改善機能一覧

### A.1 実装済み機能

| 機能 | 説明 |
|------|------|
| 階層化質問タイプ | 3階層11タイプの質問分類 |
| チャンク複雑度分析 | 専門用語密度・文長による分析 (Gemini tokenizer対応) |
| ドメイン適応戦略 | データセット別の最適化 |
| 品質スコアリング | Q/A品質の自動評価 |
| バッチ埋め込み生成 | Gemini API制限を考慮したバッチ処理 |
| 3段階カバレッジ分布 | 高・中・低の詳細評価 |
| LLMハイブリッド生成 | Gemini 2.0 Flashによる高品質Q/A生成の統合 |

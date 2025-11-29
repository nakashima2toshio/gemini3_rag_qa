# a03_rag_qa_coverage_improved.py - セマンティックカバレッジ分析とQ/A生成システム

## 目次

1. [概要](#1-概要)
2. [アーキテクチャ](#2-アーキテクチャ)
3. [キーワード抽出](#3-キーワード抽出)
4. [階層化質問タイプ](#4-階層化質問タイプ)
5. [チャンク複雑度分析](#5-チャンク複雑度分析)
6. [ドメイン適応戦略](#6-ドメイン適応戦略)
7. [Q/A生成戦略](#7-qa生成戦略)
8. [カバレージ分析](#8-カバレージ分析)
9. [コマンドラインオプション](#9-コマンドラインオプション)
10. [実行方法](#10-実行方法)
11. [出力ファイル](#11-出力ファイル)
12. [トラブルシューティング](#12-トラブルシューティング)

---

## 1. 概要

### 1.1 目的

`a03_rag_qa_coverage_improved.py`は、セマンティックチャンク分割と包括的Q/A生成戦略により、**80%以上のカバレッジ**を目指すQ/A生成システムです。テンプレートベースの手法により、LLMコストを最小限に抑えながら高品質なQ/Aペアを大量生成します。

### 1.2 起動コマンド

```bash
python a03_rag_qa_coverage_improved.py \
  --input OUTPUT/preprocessed_cc_news.csv \
  --dataset cc_news \
  --analyze-coverage \
  --coverage-threshold 0.60 \
  --qa-per-chunk 10 \
  --max-chunks 2000
```

### 1.3 主要機能

- **セマンティックチャンク分割**: 段落優先のセマンティック分割（helper_rag_qa.py使用）
- **階層化質問タイプ**: 3階層11タイプの質問タイプ定義
- **チャンク複雑度分析**: 専門用語密度、平均文長、統計情報検出
- **ドメイン適応戦略**: データセット別の最適化戦略
- **包括的Q/A生成**: 5つの異なる質問タイプで高カバレッジを実現
- **バッチ処理による埋め込み生成**: OpenAI APIのバッチ処理で高速化
- **改良版カバレッジ分析**: 3段階の分布評価（高・中・低）

### 1.4 対応データセット

| データセット | キー | 言語 | 説明 |
|------------|------|------|------|
| CC-News | `cc_news` | 英語 | 英語ニュース記事 |
| CC100日本語 | `japanese_text` | 日本語 | Webテキストコーパス |
| Wikipedia日本語版 | `wikipedia_ja` | 日本語 | 百科事典的知識 |
| Livedoorニュース | `livedoor` | 日本語 | ニュースコーパス |

### 1.5 a02_make_qa_para.pyとの比較

| 項目 | a02（LLM版） | a03（テンプレート版） |
|------|-------------|---------------------|
| **主な目的** | LLMでQ/A生成 | テンプレートベースで高カバレッジ |
| **Q/A生成手法** | LLMによる高品質生成 | ルールベース+テンプレート |
| **コスト** | 中程度（LLM呼び出し） | 極めて低い（埋め込みのみ） |
| **生成速度** | 中速（API待機あり） | 高速（ルールベース） |
| **カバレッジ** | 90-95% | **95%+** |
| **Q/A品質** | 非常に高い | 高い（構造化） |

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
│                              │                                  │
│                              ▼                                  │
│  [3] Q/A生成                                                    │
│      ├── generate_advanced_qa_for_chunk()（高度版）              │
│      └── generate_comprehensive_qa_for_chunk()（包括版）         │
│                              │                                  │
│                              ▼                                  │
│  [4] カバレッジ分析                                              │
│      calculate_improved_coverage()                              │
│                              │                                  │
│                              ▼                                  │
│  [5] 結果保存                                                    │
│      save_results() → qa_output/a03/                           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 依存モジュール

```python
from helper_rag_qa import SemanticCoverage

import tiktoken
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

    def __init__(self, prefer_mecab: bool = True):
        """MeCab優先設定"""

    def extract(self, text: str, top_n: int = 5) -> List[str]:
        """キーワード抽出（自動フォールバック対応）"""

    def _extract_with_mecab(self, text: str, top_n: int) -> List[str]:
        """MeCabによる複合名詞抽出"""

    def _extract_with_regex(self, text: str, top_n: int) -> List[str]:
        """正規表現によるキーワード抽出"""

    def _filter_and_count(self, words: List[str], top_n: int) -> List[str]:
        """頻度ベースのフィルタリング"""
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
    global _keyword_extractor
    if _keyword_extractor is None:
        _keyword_extractor = KeywordExtractor()
    return _keyword_extractor
```

---

## 4. 階層化質問タイプ

### 4.1 3階層11タイプの定義

```python
QUESTION_TYPES_HIERARCHY = {
    "basic": {
        "definition": {"ja": "定義型（〜とは何ですか？）", "en": "Definition (What is...?)"},
        "identification": {"ja": "識別型（〜の例を挙げてください）", "en": "Identification (Give examples of...)"},
        "enumeration": {"ja": "列挙型（〜の種類/要素は？）", "en": "Enumeration (List the types/elements of...)"}
    },
    "understanding": {
        "cause_effect": {"ja": "因果関係型（〜の結果/影響は？）", "en": "Cause-Effect (What is the result/impact of...?)"},
        "process": {"ja": "プロセス型（〜はどのように行われますか？）", "en": "Process (How is... performed?)"},
        "mechanism": {"ja": "メカニズム型（〜の仕組みは？）", "en": "Mechanism (How does... work?)"},
        "comparison": {"ja": "比較型（〜と〜の違いは？）", "en": "Comparison (What's the difference between...?)"}
    },
    "application": {
        "synthesis": {"ja": "統合型（〜を組み合わせるとどうなりますか？）", "en": "Synthesis (What happens when combining...?)"},
        "evaluation": {"ja": "評価型（〜の長所と短所は？）", "en": "Evaluation (What are the pros and cons of...?)"},
        "prediction": {"ja": "予測型（〜の場合どうなりますか？）", "en": "Prediction (What would happen if...?)"},
        "practical": {"ja": "実践型（〜はどのように活用されますか？）", "en": "Practical (How is... applied in practice?)"}
    }
}
```

### 4.2 階層別の特徴

| 階層 | タイプ数 | 目的 | 難易度 |
|------|---------|------|--------|
| basic | 3 | 基本的な事実確認 | 低 |
| understanding | 4 | 概念の理解・関係性 | 中 |
| application | 4 | 応用・実践・評価 | 高 |

---

## 5. チャンク複雑度分析

### 5.1 analyze_chunk_complexity関数

```python
def analyze_chunk_complexity(chunk_text: str, lang: str = "auto") -> Dict:
    """
    チャンクの複雑度を分析して、適切なQ/A生成戦略を決定

    Returns:
        {
            "complexity_level": "high" | "medium" | "low",
            "complexity_score": int,
            "technical_terms": List[str],  # 上位10個
            "avg_sentence_length": float,
            "concept_density": float,
            "sentence_count": int,
            "token_count": int,
            "has_statistics": bool,
            "numeric_data": List[str],  # 上位5個
            "lang": str
        }
    """
```

### 5.2 複雑度レベル判定

| レベル | スコア | 条件 |
|--------|--------|------|
| high | >= 5 | 概念密度 > 5% または 平均文長 > 30 |
| medium | >= 3 | 概念密度 > 2% または 平均文長 > 20 |
| low | < 3 | その他 |

### 5.3 スコア計算ロジック

```python
complexity_score = 0

# 概念密度による加点
if concept_density > 5:
    complexity_score += 3
elif concept_density > 2:
    complexity_score += 2
else:
    complexity_score += 1

# 平均文長による加点
if avg_sentence_length > 30:
    complexity_score += 2
elif avg_sentence_length > 20:
    complexity_score += 1

# 統計情報による加点
if has_statistics:
    complexity_score += 1
```

### 5.4 専門用語検出パターン

**日本語**:
```python
technical_pattern = r'[ァ-ヴー]{4,}|[一-龥]{4,}'
```

**英語**:
```python
technical_pattern = r'[A-Z][a-z]+(?:[A-Z][a-z]+)+|\b\w{10,}\b'
```

---

## 6. ドメイン適応戦略

### 6.1 データセット別戦略

```python
DOMAIN_SPECIFIC_STRATEGIES = {
    "cc_news": {
        "focus_types": ["cause_effect", "process", "comparison", "evaluation"],
        "avoid_types": ["definition"],
        "emphasis": "時事性と社会的影響"
    },
    "wikipedia_ja": {
        "focus_types": ["definition", "mechanism", "enumeration", "comparison"],
        "avoid_types": ["prediction"],
        "emphasis": "正確な定義と体系的な知識"
    },
    "livedoor": {
        "focus_types": ["cause_effect", "evaluation", "practical", "comparison"],
        "avoid_types": ["mechanism"],
        "emphasis": "読者の関心と実用性"
    },
    "japanese_text": {
        "focus_types": ["definition", "process", "practical", "comparison"],
        "avoid_types": [],
        "emphasis": "一般的な理解と応用"
    }
}
```

### 6.2 複雑度に基づくQ/A分布

| 複雑度 | basic | understanding | application |
|--------|-------|---------------|-------------|
| high | 1 | 3 | 1 |
| medium | 2 | 2 | 1 |
| low | 3 | 1 | 0 |

---

## 7. Q/A生成戦略

### 7.1 高度なQ/A生成（generate_advanced_qa_for_chunk）

複雑度分析とドメイン適応を組み合わせた高度なQ/A生成:

```python
def generate_advanced_qa_for_chunk(
    chunk_text: str,
    chunk_idx: int,
    qa_per_chunk: int = 5,
    lang: str = "auto",
    dataset_type: str = "custom"
) -> List[Dict]:
    """
    高度なQ/A生成システム（品質スコアリング機能付き）

    処理フロー:
    1. チャンク複雑度分析
    2. ドメイン適応戦略の取得
    3. コンテキスト強化型Q/A生成
    4. マルチホップ推論型Q/A生成（中〜高複雑度）
    5. 階層化質問タイプに基づくQ/A生成
    6. 品質スコアリングとソート
    """
```

### 7.2 包括的Q/A生成（generate_comprehensive_qa_for_chunk）

5つの戦略による包括的なQ/A生成:

#### 1. comprehensive型（包括的質問）

**英語**:
```json
{
  "question": "What information is discussed in this section?",
  "answer": "チャンク全体のテキスト（最大500文字）",
  "type": "comprehensive",
  "coverage_strategy": "full_chunk"
}
```

**日本語**:
```json
{
  "question": "このセクションにはどのような情報が含まれていますか？",
  "answer": "チャンク全体のテキスト",
  "type": "comprehensive"
}
```

#### 2. factual_detailed型（詳細な事実確認）

**英語**:
```json
{
  "question": "What specific information is provided about {concept}?",
  "answer": "該当文 + 次の文（文脈付与）",
  "type": "factual_detailed"
}
```

**日本語**:
```json
{
  "question": "「{文の先頭30文字}」について詳しく説明してください。",
  "answer": "該当文 + 次の文",
  "type": "factual_detailed"
}
```

#### 3. contextual型（文脈関連型）

```json
{
  "question": "How does {current_concept} relate to {previous_concept}?",
  "answer": "前の文 + 現在の文",
  "type": "contextual"
}
```

#### 4. keyword_based型（キーワード抽出型）

**英語**:
```json
{
  "question": "What is mentioned about {keyword}?",
  "answer": "該当文",
  "type": "keyword_based"
}
```

**日本語（MeCab使用）**:
```json
{
  "question": "「{keyword}」について何が述べられていますか？",
  "answer": "該当文",
  "type": "keyword_based",
  "keyword": "抽出されたキーワード"
}
```

#### 5. thematic型（テーマ型）

**英語**:
```json
{
  "question": "What is the main theme related to {theme_concept}?",
  "answer": "チャンク全体のテキスト（最大400文字）",
  "type": "thematic"
}
```

**日本語**:
```json
{
  "question": "「{theme_keyword}」に関する主要テーマは何ですか？",
  "answer": "チャンク全体のテキスト",
  "type": "thematic"
}
```

---

## 8. カバレージ分析

### 8.1 改良版カバレッジ計算

```python
def calculate_improved_coverage(
    chunks: List[Dict],
    qa_pairs: List[Dict],
    analyzer: SemanticCoverage,
    threshold: float = 0.65
) -> Tuple[Dict, List[float]]:
    """
    改善されたカバレッジ計算（バッチ処理版）

    Returns:
        (coverage_results, max_similarities)
    """
```

### 8.2 バッチ処理による埋め込み生成

```python
MAX_BATCH_SIZE = 2048

if len(qa_texts) <= MAX_BATCH_SIZE:
    # 一度にすべて処理可能
    qa_chunks = [{"text": text} for text in qa_texts]
    qa_embeddings = analyzer.generate_embeddings(qa_chunks)
else:
    # バッチサイズを超える場合は分割処理
    for i in range(0, len(qa_texts), MAX_BATCH_SIZE):
        batch = qa_texts[i:i+MAX_BATCH_SIZE]
        batch_chunks = [{"text": text} for text in batch]
        batch_embeddings = analyzer.generate_embeddings(batch_chunks)
        qa_embeddings.extend(batch_embeddings)
```

### 8.3 カバレッジ結果の構造

```python
coverage_results = {
    "coverage_rate": float,           # カバレッジ率
    "covered_chunks": int,            # カバー済みチャンク数
    "total_chunks": int,              # 総チャンク数
    "threshold": float,               # 使用した閾値
    "avg_max_similarity": float,      # 平均最大類似度
    "min_max_similarity": float,      # 最小の最大類似度
    "max_max_similarity": float,      # 最大の最大類似度
    "uncovered_chunks": List[int],    # 未カバーチャンクのインデックス
    "coverage_distribution": {
        "high_coverage": int,         # >= 0.7
        "medium_coverage": int,       # 0.5 - 0.7
        "low_coverage": int           # < 0.5
    }
}
```

### 8.4 カバレッジ分布評価

| レベル | 類似度範囲 | 説明 |
|--------|----------|------|
| 高カバレッジ | >= 0.7 | 十分にカバーされている |
| 中カバレッジ | 0.5 - 0.7 | ある程度カバーされている |
| 低カバレッジ | < 0.5 | カバー不足 |

---

## 9. コマンドラインオプション

### 9.1 全オプション一覧

| オプション | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `--input` | str | 必須 | 入力ファイルパス |
| `--dataset` | str | 必須 | データセットタイプ |
| `--max-docs` | int | None | 処理する最大文書数 |
| `--methods` | str[] | ['rule', 'template'] | 使用する手法 |
| `--model` | str | gpt-4o-mini | 使用するモデル |
| `--output` | str | qa_output | 出力ディレクトリ |
| `--analyze-coverage` | flag | False | カバレッジ分析を実行 |
| `--coverage-threshold` | float | 0.65 | カバレッジ判定閾値 |
| `--qa-per-chunk` | int | 4 | チャンクあたりのQ/A数 |
| `--max-chunks` | int | 300 | 処理する最大チャンク数 |
| `--demo` | flag | False | デモモード |

### 9.2 手法オプション

```bash
--methods rule template    # デフォルト（ルール+テンプレート）
--methods rule template llm  # LLMも追加（コスト増）
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
  --coverage-threshold 0.60
```

**実行時間**: 約2分
**生成Q/A数**: 約7,300個
**カバレッジ**: 99.7%

### 10.2 推奨実行（中規模）

```bash
python a03_rag_qa_coverage_improved.py \
  --input OUTPUT/preprocessed_cc_news.csv \
  --dataset cc_news \
  --qa-per-chunk 10 \
  --max-chunks 2000 \
  --analyze-coverage \
  --coverage-threshold 0.60
```

**実行時間**: 約8-10分
**生成Q/A数**: 約20,000個
**カバレッジ**: 95%+

### 10.3 本番実行（全文書）

```bash
python a03_rag_qa_coverage_improved.py \
  --input OUTPUT/preprocessed_cc_news.csv \
  --dataset cc_news \
  --qa-per-chunk 10 \
  --max-chunks 18000 \
  --analyze-coverage \
  --coverage-threshold 0.60
```

**実行時間**: 約60-90分
**生成Q/A数**: 約144,000個
**カバレッジ**: 95%+

### 10.4 実行時間の見積もり

| 設定 | 文書数 | チャンク数 | Q/A数 | 実行時間 | コスト |
|------|--------|----------|-------|---------|--------|
| テスト | 150 | 609 | 7,308 | 2分 | $0.001 |
| 推奨 | 自動 | 2,000 | 20,000 | 8-10分 | $0.005 |
| 中規模 | 1,000 | 2,400 | 24,000 | 10-12分 | $0.006 |
| 全文書 | 7,499 | 18,000 | 144,000 | 60-90分 | $0.025 |

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

### 11.2 Q/Aペア形式（JSON）

```json
[
  {
    "question": "What information is discussed in this section?",
    "answer": "AI technology has significantly advanced...",
    "type": "comprehensive",
    "chunk_idx": 0,
    "coverage_strategy": "full_chunk"
  },
  {
    "question": "「自然言語処理」について何が述べられていますか？",
    "answer": "AI技術の発展により...",
    "type": "keyword_based",
    "chunk_idx": 2,
    "keyword": "自然言語処理"
  }
]
```

### 11.3 カバレッジ結果形式（JSON）

```json
{
  "coverage_rate": 0.903,
  "covered_chunks": 1526,
  "total_chunks": 1689,
  "threshold": 0.6,
  "avg_max_similarity": 0.745,
  "min_max_similarity": 0.312,
  "max_max_similarity": 0.987,
  "uncovered_chunks": [45, 123, 456],
  "coverage_distribution": {
    "high_coverage": 1173,
    "medium_coverage": 484,
    "low_coverage": 32
  }
}
```

### 11.4 サマリー形式（JSON）

```json
{
  "dataset_type": "cc_news",
  "generated_at": "20251108_010658",
  "total_qa_pairs": 4278,
  "coverage_rate": 0.903,
  "coverage_details": {
    "high_coverage": 1173,
    "medium_coverage": 484,
    "low_coverage": 32
  },
  "files": {
    "qa_json": "qa_output/a03/qa_pairs_cc_news_20251108_010658.json",
    "qa_csv": "qa_output/a03/qa_pairs_cc_news_20251108_010658.csv",
    "coverage": "qa_output/a03/coverage_cc_news_20251108_010658.json",
    "summary": "qa_output/a03/summary_cc_news_20251108_010658.json"
  }
}
```

---

## 12. トラブルシューティング

### 12.1 APIキーエラー

**症状**: `OPENAI_API_KEYが設定されていません`

**解決策**:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
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

# 閾値を下げる
python a03_rag_qa_coverage_improved.py --coverage-threshold 0.55

# チャンク数を増やす
python a03_rag_qa_coverage_improved.py --max-chunks 2000
```

### 12.4 MeCabが利用できない

**症状**: `⚠️ MeCabが利用できません（正規表現モード）`

**影響**: キーワード抽出が正規表現ベースになる（セマンティック分割には影響なし）

**解決策（オプション）**:
```bash
# macOS
brew install mecab mecab-ipadic
pip install mecab-python3

# Ubuntu/Debian
sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8
pip install mecab-python3
```

### 12.5 メモリ不足

**症状**: 大量データ処理時のメモリエラー

**解決策**:
```bash
# チャンク数を制限
python a03_rag_qa_coverage_improved.py --max-chunks 1000 --max-docs 500
```

---

## 付録: 品質改善機能一覧

### A.1 実装済み機能

| 機能 | 説明 |
|------|------|
| 階層化質問タイプ | 3階層11タイプの質問分類 |
| チャンク複雑度分析 | 専門用語密度・文長による分析 |
| ドメイン適応戦略 | データセット別の最適化 |
| 品質スコアリング | Q/A品質の自動評価 |
| バッチ埋め込み生成 | 大量データの高速処理 |
| 3段階カバレッジ分布 | 高・中・低の詳細評価 |

### A.2 品質統計出力例

```
📊 Q/A品質統計:
  - 基礎レベル: 1426件（定義・識別・列挙型）
  - 理解レベル: 2137件（因果関係・プロセス・メカニズム・比較型）
  - 応用レベル: 715件（統合・評価・予測・実践型）
  - 平均品質スコア: 0.82
  - 平均多様性スコア: 0.88
```
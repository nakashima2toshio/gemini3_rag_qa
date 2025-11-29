# helper_rag_qa.py 技術仕様書

最終更新日: 2025-11-27

## 概要

RAG Q&A生成のための包括的ユーティリティモジュール。キーワード抽出、Q/A生成、セマンティックカバレッジ分析など、17個の専門クラスを提供。多言語対応（英語・日本語）により、入力言語に応じた適切なQ&A生成が可能。

## ファイル情報

- **ファイル名**: helper_rag_qa.py
- **行数**: 約3200行
- **主要機能**: キーワード抽出、Q/A生成、カバレッジ分析、多言語対応
- **クラス数**: 17クラス
- **対応言語**: 英語（en）、日本語（ja）

---

## アーキテクチャ

### モジュール構造

```
helper_rag_qa.py
├── インポート・設定 (L1-27)
│
├── キーワード抽出関連 (L28-1478)
│   ├── BestKeywordSelector (L61-226)
│   ├── SmartKeywordSelector (L228-442)
│   ├── get_best_keywords() (L446-461)
│   ├── get_smart_keywords() (L463-477)
│   ├── QACountOptimizer (L479-685)
│   └── QAOptimizedExtractor (L687-1478)
│
├── セマンティックカバレッジ関連 (L1480-1971)
│   ├── SemanticCoverage (L1484-1935)
│   └── QAGenerationConsiderations (L1937-1971)
│
├── データモデル (L1974-1985)
│   ├── QAPair (L1974-1980)
│   └── QAPairsList (L1983-1985)
│
├── Q/A生成クラス (L1988-2890)
│   ├── LLMBasedQAGenerator (L1988-2070)
│   ├── ChainOfThoughtQAGenerator (L2072-2130)
│   ├── RuleBasedQAGenerator (L2132-2230)
│   ├── TemplateBasedQAGenerator (L2232-2320)
│   ├── HybridQAGenerator (L2322-2420)
│   ├── AdvancedQAGenerationTechniques (L2422-2490)
│   ├── QAGenerationOptimizer (L2492-2560)
│   └── OptimizedHybridQAGenerator (L2562-2890)
│
└── バッチ処理クラス (L2892-3200)
    └── BatchHybridQAGenerator (L2892-3200)
```

---

## クラス一覧

### キーワード抽出関連（4クラス）

| クラス | 行番号 | 説明 |
|--------|--------|------|
| BestKeywordSelector | L61-226 | 3手法比較選択 |
| SmartKeywordSelector | L228-442 | 自動最適化 |
| QACountOptimizer | L479-685 | 最適Q/A数決定 |
| QAOptimizedExtractor | L687-1478 | Q/A特化抽出 |

### セマンティックカバレッジ関連（2クラス）

| クラス | 行番号 | 説明 |
|--------|--------|------|
| SemanticCoverage | L1484-1935 | 網羅性測定 |
| QAGenerationConsiderations | L1937-1971 | 生成前チェック |

### データモデル（2クラス）

| クラス | 行番号 | 説明 |
|--------|--------|------|
| QAPair | L1974-1980 | Q/Aペアモデル |
| QAPairsList | L1983-1985 | Q/Aリスト |

### Q/A生成関連（9クラス）

| クラス | 行番号 | 説明 |
|--------|--------|------|
| LLMBasedQAGenerator | L1988-2070 | LLM生成 |
| ChainOfThoughtQAGenerator | L2072-2130 | CoT生成 |
| RuleBasedQAGenerator | L2132-2230 | ルール生成 |
| TemplateBasedQAGenerator | L2232-2320 | テンプレート生成 |
| HybridQAGenerator | L2322-2420 | ハイブリッド生成 |
| AdvancedQAGenerationTechniques | L2422-2490 | 高度技術 |
| QAGenerationOptimizer | L2492-2560 | 最適化 |
| OptimizedHybridQAGenerator | L2562-2890 | 最適化ハイブリッド |
| BatchHybridQAGenerator | L2892-3200 | バッチ処理 |

---

## 主要クラス詳細

## 1. キーワード抽出関連クラス

### 1.1 BestKeywordSelector (L61-226)

3つの手法（MeCab/正規表現/統合版）から最良のキーワードを選択。

#### 評価重み付け (L72-78)

```python
self.weights = {
    'coverage': 0.25,      # カバレージ率
    'diversity': 0.15,     # 多様性
    'technicality': 0.25,  # 専門性
    'coherence': 0.20,     # 一貫性
    'length_balance': 0.15 # 長さのバランス
}
```

#### 主要メソッド

| メソッド | 行番号 | 説明 |
|---------|--------|------|
| `evaluate_keywords()` | L80-144 | キーワードセットの品質評価（5指標） |
| `calculate_total_score()` | L146-158 | 総合スコア計算 |
| `extract_best()` | L160-206 | 3手法から最良の結果を選択 |

#### 評価指標

1. **カバレージ率**: キーワードがテキストに存在する割合
2. **多様性**: 文字数の分散（標準偏差2-4文字が理想）
3. **専門性**: カタカナ・英語・漢字複合語の割合
4. **一貫性**: キーワード間の関連性
5. **長さバランス**: 2-8文字の割合

---

### 1.2 SmartKeywordSelector (L228-442)

テキスト特性に応じた最適なキーワード抽出。

#### モード別デフォルトtop_n (L235-241)

| モード | top_n | 用途 |
|--------|-------|------|
| summary | 5 | 要約・概要把握 |
| standard | 10 | 標準的な分析 |
| detailed | 15 | 詳細分析 |
| exhaustive | 20 | 網羅的抽出 |
| tag | 3 | タグ付け |

#### calculate_auto_top_n() (L243-279)

テキスト長に基づく自動決定:

| 文字数 | top_n |
|--------|-------|
| < 100 | 3 |
| < 300 | 5 |
| < 500 | 7（専門用語密度で+2） |
| < 1000 | 10（文数で+2） |
| < 2000 | 15 |
| >= 2000 | 20 + log調整（最大30） |

---

### 1.3 QACountOptimizer (L479-685)

最適なQ/A生成数を動的に決定。

#### 決定モード

| モード | 説明 | Q/A数目安 |
|--------|------|-----------|
| auto | 文書長ベース自動決定 | 3-30 |
| evaluation | 網羅性重視 | 文数の7.5% |
| learning | 主要概念カバー | 10-20 |
| search_test | 多様な質問パターン | 20-30 |
| faq | ユーザーニーズベース | 5-15 |

---

### 1.4 QAOptimizedExtractor (L687-1478)

Q/Aペア生成に最適化されたキーワード抽出。

#### 主要メソッド

| メソッド | 説明 |
|---------|------|
| `extract_for_qa_generation()` | Q/A生成特化のキーワード抽出 |
| `extract_with_relationships()` | エンティティ間の関係性抽出 |
| `classify_difficulty()` | 難易度分類（easy/medium/hard） |
| `generate_context_hints()` | キーワード周辺の文脈情報抽出 |
| `detect_duplicate_qa()` | 重複Q/Aペアの検出 |
| `generate_qa_pairs()` | 抽出結果からQ/Aペア生成 |

#### 関係性タイプ

| タイプ | 説明 |
|--------|------|
| is_a | 定義関係 |
| causes | 因果関係 |
| uses | 使用関係 |
| transforms | 変換関係 |
| in_context | 文脈関係 |
| by_means_of | 手段関係 |
| temporal | 時系列関係 |
| co_occurs | 共起関係 |

---

## 2. セマンティックカバレッジ関連クラス

### 2.1 SemanticCoverage (L1484-1935)

意味的な網羅性を測定。

#### 初期化 (L1487-1500)

```python
def __init__(self, embedding_model="text-embedding-3-small"):
    self.embedding_model = embedding_model
    # APIキーの確認
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and api_key != 'your-openai-api-key-here':
        self.client = OpenAI()
        self.has_api_key = True
    else:
        self.client = None
        self.has_api_key = False
    self.tokenizer = tiktoken.get_encoding("cl100k_base")
    # MeCab利用可否チェック
    self.mecab_available = self._check_mecab_availability()
```

#### 主要メソッド

| メソッド | 行番号 | 説明 |
|---------|--------|------|
| `_check_mecab_availability()` | L1502-1511 | MeCab利用可能性チェック |
| `create_semantic_chunks()` | L1513-1606 | セマンティックチャンク分割 |
| `_split_into_paragraphs()` | L1608-1621 | 段落分割 |
| `_chunk_by_paragraphs()` | L1623-1683 | 段落ベースチャンク化 |
| `_force_split_sentence()` | L1685-1711 | 強制分割（上限超過時） |
| `_split_into_sentences()` | L1713-1731 | 文分割（言語自動判定） |
| `_split_sentences_mecab()` | L1733-1765 | MeCab文分割（日本語） |
| `_adjust_chunks_for_topic_continuity()` | L1767-1807 | トピック連続性調整 |
| `generate_embeddings()` | L1809-1851 | チャンク埋め込み生成 |
| `generate_embedding()` | L1853-1868 | 単一テキスト埋め込み |
| `generate_embeddings_batch()` | L1870-1910 | バッチ埋め込み生成 |
| `cosine_similarity()` | L1912-1934 | コサイン類似度計算 |

#### create_semantic_chunks() 処理フロー

```
入力テキスト
    ↓
段落分割（prefer_paragraphs=True）
    ↓
各段落のトークン数チェック
    ├── <= max_tokens → そのままチャンク化
    └── > max_tokens → 文単位に分割
                            ├── <= max_tokens → 文グループ化
                            └── > max_tokens → 強制分割
    ↓
トピック連続性調整（短いチャンクをマージ）
    ↓
チャンクリスト出力
```

#### チャンクタイプ

| タイプ | 説明 |
|--------|------|
| paragraph | 段落単位（最も意味的） |
| sentence_group | 文グループ（段落分割後） |
| forced_split | 強制分割（最終手段） |
| merged | マージ済み |

---

### 2.2 QAGenerationConsiderations (L1937-1971)

Q/A生成前のチェックリスト。

#### analyze_document_characteristics()

```python
{
    "document_type": "技術文書/物語/レポート等",
    "complexity_level": "専門性のレベル",
    "factual_density": "事実情報の密度",
    "structure": "構造化の度合い",
    "language": "言語と文体",
    "domain": "ドメイン特定",
    "length": "文書長",
    "ambiguity_level": "曖昧さの度合い"
}
```

---

## 3. データモデル

### 3.1 QAPair (L1974-1980)

```python
class QAPair(BaseModel):
    question: str
    answer: str
    question_type: str
    difficulty: str
    source_span: str
```

### 3.2 QAPairsList (L1983-1985)

```python
class QAPairsList(BaseModel):
    qa_pairs: List[QAPair]
```

---

## 4. Q/A生成クラス

### 4.1 LLMBasedQAGenerator (L1988-2070)

LLM（GPT-5-mini）を使用したQ/A生成。

#### 主要メソッド

| メソッド | 説明 |
|---------|------|
| `generate_basic_qa()` | 基本的なQ/A生成（Pydantic構造化出力） |
| `generate_diverse_qa()` | 6種類のQ/A生成 |

#### Q/Aタイプ

| タイプ | 説明 |
|--------|------|
| factual | 事実確認 |
| causal | 因果関係 |
| comparative | 比較 |
| inferential | 推論 |
| summary | 要約 |
| application | 応用 |

---

### 4.2 ChainOfThoughtQAGenerator (L2072-2130)

思考の連鎖（CoT）を使った高品質Q/A生成。

#### 5ステップ処理

1. テキストの主要なトピックと概念を抽出
2. 各トピックについて重要な情報を特定
3. その情報を問う質問を設計
4. テキストから正確な回答を抽出
5. 質問と回答の妥当性を検証

---

### 4.3 RuleBasedQAGenerator (L2132-2230)

ルールベースのQ/A生成。

#### パターンマッチング

- 「〜とは〜である」→ 定義質問
- 「〜は〜と呼ばれる」→ 名称質問
- spaCy固有表現認識 → エンティティ質問

---

### 4.4 TemplateBasedQAGenerator (L2232-2320)

テンプレートを使用したQ/A生成。

#### テンプレート種類

| タイプ | 説明 |
|--------|------|
| definition | 定義質問 |
| function | 役割質問 |
| method | 方法質問 |
| characteristic | 特徴質問 |
| comparison | 比較質問 |
| reason | 理由質問 |

---

### 4.5 HybridQAGenerator (L2322-2420)

複数手法の組み合わせ。

#### 処理フロー

1. ルールベース抽出
2. テンプレートベース生成
3. LLM生成（オプション）
4. 重複除去
5. 品質検証

---

### 4.6 OptimizedHybridQAGenerator (L2562-2890)

最適化されたハイブリッドQ/A生成。

#### 初期化

```python
def __init__(
    self,
    model: str = "gpt-5-mini",
    embedding_model: str = "text-embedding-3-small"
):
    # temperature非対応モデルのリスト
    self.no_temperature_models = {
        'gpt-5-mini', 'o1-mini', 'o1-preview',
        'o3-mini', 'o3', 'o4-mini', 'o4'
    }
```

#### 主要メソッド

| メソッド | 説明 |
|---------|------|
| `generate_optimized_hybrid_qa()` | 最適化ハイブリッドQ/A生成 |
| `_calculate_semantic_coverage()` | セマンティックカバレッジ計算 |
| `_create_semantic_chunks()` | セマンティックチャンク分割 |

#### カバレージ計算の多段階閾値

| レベル | 閾値 | 説明 |
|--------|------|------|
| strict | 0.85 | 厳格（専門用語完全一致） |
| standard | 0.80 | 標準（現行） |
| lenient | 0.75 | 緩和（関連性あり） |

---

### 4.7 BatchHybridQAGenerator (L2892-3200)

バッチ処理に最適化。

#### 初期化

```python
def __init__(
    self,
    model: str = "gpt-5-mini",
    embedding_model: str = "text-embedding-3-small",
    batch_size: int = 10,
    embedding_batch_size: int = 100,
    quality_mode: bool = False,
    target_coverage: float = 0.95
):
```

#### 品質重視モード（quality_mode=True）

- batch_sizeを最大5に制限
- target_coverage=0.95を目標
- 階層的Q/A生成を使用
- カバレッジフィードバックループ

#### 主要メソッド

| メソッド | 説明 |
|---------|------|
| `calculate_optimal_qa_count()` | 最適Q/A数の動的決定 |
| `generate_hierarchical_qa()` | 階層的Q/A生成（3層） |
| `generate_with_coverage_feedback()` | カバレッジフィードバックループ |
| `generate_batch_hybrid_qa()` | バッチQ/A生成 |

#### 階層的Q/A生成（3層構造）

| 層 | 説明 | Q/A数 |
|----|------|-------|
| 第1層 | 文書全体の包括的質問 | 1-2個 |
| 第2層 | パラグラフレベルの詳細質問 | 3-4個 |
| 第3層 | キーワード/エンティティ特化質問 | 5-6個 |

---

## パフォーマンス最適化

### 1. バッチ処理によるAPI呼び出し削減

| 処理 | 通常 | バッチ | 削減率 |
|------|------|--------|--------|
| LLM呼び出し | 100回 | 10回 | 90% |
| 埋め込み呼び出し | 100回 | 1回 | 99% |

### 2. temperature非対応モデルの処理

```python
if self.model not in self.no_temperature_models:
    api_params["temperature"] = 0.7
```

### 3. コスト計算 (L2867-2889)

```python
pricing = {
    "gpt-5-mini": {"input": 0.15, "output": 0.60},
    "gpt-5": {"input": 1.50, "output": 6.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o3-mini": {"input": 3.00, "output": 12.00}
}
```

---

## 使用例

### 例1: 基本的なキーワード抽出

```python
from helper_rag_qa import BestKeywordSelector

selector = BestKeywordSelector()
result = selector.extract_best(text, top_n=10)

print(f"最良手法: {result['best_method']}")
print(f"キーワード: {result['keywords']}")
print(f"理由: {result['reason']}")
```

### 例2: スマートキーワード抽出

```python
from helper_rag_qa import SmartKeywordSelector

smart_selector = SmartKeywordSelector()
result = smart_selector.extract_best_auto(text, mode="auto")

print(f"自動決定top_n: {result['top_n']}")
print(f"キーワード: {result['keywords']}")
```

### 例3: Q/A生成（ハイブリッド）

```python
from helper_rag_qa import OptimizedHybridQAGenerator

generator = OptimizedHybridQAGenerator(model="gpt-5-mini")
result = generator.generate_optimized_hybrid_qa(
    text,
    qa_count=10,
    use_llm=True,
    calculate_coverage=True
)

print(f"生成数: {len(result['qa_pairs'])}")
print(f"カバレージ: {result['coverage']['coverage_percentage']:.1%}")
```

### 例4: バッチQ/A生成（英語）

```python
from helper_rag_qa import BatchHybridQAGenerator

batch_generator = BatchHybridQAGenerator(
    model="gpt-5-mini",
    batch_size=10,
    embedding_batch_size=100
)

results = batch_generator.generate_batch_hybrid_qa(
    texts=english_text_list,
    qa_count=12,
    use_llm=True,
    calculate_coverage=True,
    show_progress=True,
    lang="en"
)

for i, result in enumerate(results):
    print(f"文書{i+1}: {len(result['qa_pairs'])}個のQ/A生成")
```

### 例5: バッチQ/A生成（日本語）

```python
from helper_rag_qa import BatchHybridQAGenerator

batch_generator = BatchHybridQAGenerator(
    model="gpt-5-mini",
    batch_size=10,
    embedding_batch_size=100
)

results = batch_generator.generate_batch_hybrid_qa(
    texts=japanese_text_list,
    qa_count=12,
    use_llm=True,
    calculate_coverage=True,
    show_progress=True,
    lang="ja"
)
```

### 例6: 品質重視モード（カバレッジ95%目標）

```python
from helper_rag_qa import BatchHybridQAGenerator

quality_generator = BatchHybridQAGenerator(
    model="gpt-5-mini",
    quality_mode=True,
    target_coverage=0.95
)

result = quality_generator.generate_with_coverage_feedback(
    text=document,
    target_coverage=0.95,
    max_iterations=3,
    lang="ja"
)

print(f"最終カバレージ: {result['final_coverage']:.1%}")
print(f"反復回数: {result['iterations']}")
print(f"Q/A総数: {result['total_qa']}")
```

---

## 依存ライブラリ

```python
from regex_mecab import KeywordExtractor
import re, math, json, os
from collections import defaultdict
import numpy as np
import tiktoken
from openai import OpenAI
from pydantic import BaseModel
import spacy
from tqdm import tqdm
from dotenv import load_dotenv
```

---

## 注意事項

1. **APIキー設定**: 環境変数 `OPENAI_API_KEY` が必須
2. **モデル選択**: temperature非対応モデル（gpt-5系、o1/o3/o4系）に注意
3. **バッチサイズ**: LLM推奨10-20、埋め込み推奨100-150
4. **日本語処理**: spaCy `ja_core_news_lg` モデル必須
5. **トークン制限**: 各モデルの最大トークン数を確認
6. **MeCab**: 日本語文分割にはMeCabが推奨（未インストールでも動作）

---

## トラブルシューティング

### 問題1: KeywordExtractor not found

**対処**: `regex_mecab.py`が同ディレクトリに存在するか確認

### 問題2: spaCy model not found

**対処**:
```bash
python -m spacy download ja_core_news_lg
```

### 問題3: temperature error

**対処**: モデルが`no_temperature_models`リストにあるか確認

### 問題4: バッチ処理メモリエラー

**対処**: `batch_size`を減らす

### 問題5: カバレッジが低い

**対処**:
- `quality_mode=True`を使用
- `target_coverage`を調整
- `max_iterations`を増やす

---

## 更新履歴

**2025-11-27**:
- ドキュメント全面更新（行番号更新）
- BatchHybridQAGeneratorの品質重視モード追記
- 階層的Q/A生成、カバレッジフィードバックループの説明追加

**2025-11-05**:
- `BatchHybridQAGenerator.generate_batch_hybrid_qa()` に `lang` パラメータを追加
- `_batch_enhance_with_llm()` に言語パラメータ対応を追加
- `_create_batch_prompt()` に多言語プロンプト生成機能を実装
- 英語・日本語それぞれに最適化されたプロンプトテンプレートを追加
- 文書タイプ別（news/technical/academic）の言語別指示文を実装

**2025-11-04**:
- `SemanticCoverage` にMeCab利用可能性チェック機能を追加
- `_split_into_sentences()` に言語自動判定とMeCab統合を実装
- `_split_sentences_mecab()` による高精度な日本語文分割を追加
- MeCab未インストール環境でも正常動作する自動フォールバック機能を実装

---

## まとめ

helper_rag_qa.pyは、RAG Q/A生成のための包括的なユーティリティモジュールです。

### 主要な特徴

1. **多様なキーワード抽出**: 3手法比較選択、スマート自動調整
2. **柔軟なQ/A生成**: ルール、テンプレート、LLM、ハイブリッド
3. **バッチ処理最適化**: API呼び出し削減90-99%
4. **カバレッジ分析**: セマンティック類似度計算、品質評価
5. **多言語対応**: 英語・日本語の自動判定と言語別Q/A生成
6. **品質重視モード**: カバレッジ95%目標、階層的生成、フィードバックループ

### 推奨用途

- 大規模文書のQ/A生成
- カバレッジ重視の分析
- コスト最適化が必要な環境
- バッチ処理による高速化
- 多言語データセットの処理（英語・日本語）

---

最終更新: 2025-11-27
作成者: OpenAI RAG Q/A JP Development Team
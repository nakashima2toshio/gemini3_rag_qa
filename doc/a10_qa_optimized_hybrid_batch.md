# a10_qa_optimized_hybrid_batch.py - 【推奨】最適化ハイブリッドバッチQ/A生成システム

作成日: 2025-11-28 (最終更新: Gemini移行対応)

## 目次

1. [概要](#1-概要)
2. [アーキテクチャ](#2-アーキテクチャ)
3. [データセット設定](#3-データセット設定)
4. [バッチ処理の詳細](#4-バッチ処理の詳細)
5. [品質重視モード](#5-品質重視モード)
6. [キャッシュ機能](#6-キャッシュ機能)
7. [比較実行モード](#7-比較実行モード)
8. [コマンドラインオプション](#8-コマンドラインオプション)
9. [実行方法](#9-実行方法)
10. [出力ファイル](#10-出力ファイル)
11. [パフォーマンス](#11-パフォーマンス)
12. [トラブルシューティング](#12-トラブルシューティング)

---

## 1. 概要

### 1.1 目的

`a10_qa_optimized_hybrid_batch.py`は、**RAGシステム構築におけるデフォルトの選択肢**として設計された、最適化ハイブリッドQ/A生成システムです。
Gemini API (`gemini-2.0-flash`) の長文脈処理能力を活かし、複数文書をバッチ処理することで、**API呼び出し回数を約90%削減**しながら、品質重視モードで**95%のカバレッジ**を実現します。

> **【推奨】**
> 通常のユースケース（ニュース記事、社内文書、Webページ等）では、まずこのスクリプトを使用してください。
> 最も高速かつ低コストに、高品質なQ/Aペアを大量生成できます。

### 1.2 起動コマンド

```bash
# 基本使用
python a10_qa_optimized_hybrid_batch.py --dataset cc_news --model gemini-2.0-flash

# 品質重視モード（推奨）
python a10_qa_optimized_hybrid_batch.py 
  --dataset cc_news 
  --model gemini-2.0-flash 
  --quality-mode 
  --target-coverage 0.95 
  --batch-size 10 
  --embedding-batch-size 300
```

### 1.3 主要機能

-   **大規模バッチ処理**: 複数文書を一度のLLM呼び出しで処理し、APIコストとレイテンシを削減
-   **ハイブリッド生成戦略**: ルールベースとLLM (`gemini-2.0-flash`) を組み合わせた効率的なQ/A生成
-   **品質重視モード**: `gemini-embedding-001` を使用したカバレッジ分析に基づき、目標95%達成を目指す高品質生成
-   **キャッシュ機能**: 埋め込みベクトルのキャッシュにより、2回目以降の実行時間を短縮
-   **段階的品質向上**: 初回は速度優先、後から品質を向上させる戦略をサポート
-   **MeCab対応**: 日本語データセットで高精度文境界検出（利用可能時）
-   **比較実行モード**: 通常版とバッチ版の性能を比較し、最適化効果を可視化

### 1.4 MeCab対応

| 言語 | MeCab利用可能時 | MeCab利用不可時 |
|------|----------------|----------------|
| 日本語（ja） | MeCabによる高精度文境界検出 | 正規表現にフォールバック |
| 英語（en） | 正規表現ベースの文分割 | 正規表現ベースの文分割 |

### 1.5 他システムとの比較

| 項目 | a02（LLM版） | a03（テンプレート版） | a10（ハイブリッドバッチ） |
|------|-------------|---------------------|----------------------|
| **Q/A生成手法** | LLM (`gemini-2.0-flash`) のみ | テンプレートのみ | ルールベース+LLM (Gemini) |
| **API呼び出し** | 多い | 最小（埋め込みのみ） | 中程度（バッチ化） |
| **処理時間** | 60-80分 | 60-90分 | **大幅短縮 (例: 61分)** |
| **コスト** | 高い | 極めて低い | **中程度（効率化）** |
| **カバレッジ** | 90-95% | 95%+ | **95%（品質モード）** |
| **Q/A品質** | 非常に高い | 高い | **非常に高い（効率的）** |

---

## 2. アーキテクチャ

### 2.1 システム構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                a10_qa_optimized_hybrid_batch.py                 │
├─────────────────────────────────────────────────────────────────┤
│  [1] データ読み込み                                              │
│      load_preprocessed_data()                                   │
│                              │                                  │
│                              ▼                                  │
│  [2] バッチ生成器初期化                                          │
│      BatchHybridQAGenerator(model="gemini-2.0-flash", ...)    │
│                              │                                  │
│                              ▼                                  │
│  [3] バッチ処理Q/A生成                                           │
│      generator.generate_batch_hybrid_qa()                       │
│      (LLMとEmbeddingにGemini APIを利用)                         │
│                              │                                  │
│                              ▼                                  │
│  [4] 結果保存                                                    │
│      save_batch_results() → qa_output/a10/                      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 依存モジュール

```python
from helper_rag_qa import BatchHybridQAGenerator, OptimizedHybridQAGenerator
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
```

### 2.3 主要クラス

#### BatchHybridQAGenerator

`BatchHybridQAGenerator` は、`UnifiedLLMClient` を介してGemini LLM (`gemini-2.0-flash` など) と `gemini-embedding-001` を使用し、バッチ処理でQ/A生成を行うメインクラスです。

```python
generator = BatchHybridQAGenerator(
    model=model, # 例: "gemini-2.0-flash"
    batch_size=batch_size,
    embedding_batch_size=embedding_batch_size,
    quality_mode=quality_mode,
    target_coverage=target_coverage
)

batch_results = generator.generate_batch_hybrid_qa(
    texts=texts,
    qa_count=qa_count,
    use_llm=use_llm,
    calculate_coverage=calculate_coverage,
    document_type=doc_type,
    show_progress=True,
    lang=lang
)
```

#### OptimizedHybridQAGenerator

通常版（個別処理）のQ/A生成クラスで、比較実行モードで使用されます。こちらも内部で`UnifiedLLMClient`を利用します。

```python
normal_generator = OptimizedHybridQAGenerator(model=model) # 例: "gemini-2.0-flash"

result = normal_generator.generate_hybrid_qa(
    text=text,
    qa_count=3,
    use_llm=True,
    calculate_coverage=True,
    document_type="auto"
)
```

---

## 3. データセット設定

### 3.1 対応データセット一覧

```python
DATASET_CONFIGS = {
    "cc_news": {
        "name": "CC-News英語ニュース",
        "file": "OUTPUT/preprocessed_cc_news.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "en",
        "default_doc_type": "news"
    },
    "japanese_text": {
        "name": "日本語Webテキスト",
        "file": "OUTPUT/preprocessed_japanese_text.csv",
        "text_column": "Combined_Text",
        "title_column": None,
        "lang": "ja",
        "default_doc_type": "auto"
    },
    "wikipedia_ja": {
        "name": "Wikipedia日本語版",
        "file": "OUTPUT/preprocessed_wikipedia_ja.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja",
        "default_doc_type": "academic"
    },
    "livedoor": {
        "name": "Livedoorニュースコーパス",
        "file": "OUTPUT/preprocessed_livedoor.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "category_column": "category",
        "lang": "ja",
        "default_doc_type": "news"
    }
}
```

### 3.2 文書タイプ（doc_type）

| タイプ | 説明 | 適用データセット | Q/A生成の特徴 |
|--------|------|-----------------|--------------|
| `news` | ニュース記事 | cc_news, livedoor | 5W1H型質問が多い |
| `academic` | 学術・専門記事 | wikipedia_ja | 定義・説明型質問が多い |
| `auto` | 自動判定 | japanese_text | 内容に応じて最適化 |
| `technical` | 技術文書 | - | 手順・仕組み型質問が多い |

---

## 4. バッチ処理の詳細

### 4.1 バッチサイズの選択基準

#### LLMバッチサイズ（--batch-size）

| バッチサイズ | API削減率 | 推奨用途 | メリット | デメリット |
|------------|---------|---------|---------|---------|
| 5 | 80% | **品質最優先** | 高精度、エラー少ない | やや低速 |
| 10 | 90% | **推奨設定** | 速度と品質のバランス | - |
| 15 | 93% | 大規模データ | 高速 | プロンプトが長大化 |
| 20 | 95% | 超大規模データ | 最高速 | プロンプトが長大化 |

#### 埋め込みバッチサイズ（--embedding-batch-size）

Gemini `gemini-embedding-001` は、100トークンあたりのコストが非常に低く、効率的なバッチ処理が可能です。

| バッチサイズ | 処理速度 | 推奨用途 |
|------------|---------|---------|
| 100 | 標準（デフォルト） | 中規模データ |
| 300 | 高速 | **推奨設定** |
| 500 | 最高速 | 大規模データ |

### 4.3 バッチ統計情報

実行後、以下の統計情報が`generator.batch_stats`に格納されます:

```python
{
    "total_llm_calls": 110,
    "total_embedding_calls": 10,
    "avg_batch_size": 4.5,
    "max_batch_size": 10,
    "documents_processed": 497,
    "reduction_rate": 0.926
}
```

---

## 5. 品質重視モード

### 5.1 品質重視モードとは

`--quality-mode`を指定すると、`gemini-embedding-001` を使用したカバレッジ分析に基づき、カバレッジ95%を目標とした高品質Q/A生成を行います。

```bash
python a10_qa_optimized_hybrid_batch.py 
    --dataset cc_news 
    --quality-mode 
    --target-coverage 0.95 
    --model gemini-2.0-flash
```

### 5.2 通常モードとの違い

| 項目 | 通常モード | 品質重視モード |
|------|-----------|--------------|
| **目標カバレッジ** | 85% | 95% |
| **Q/A生成数** | 少なめ | 多め |
| **LLM呼び出し** | 最小限 | 必要に応じて増加 |
| **処理時間** | 短い | やや長い（+20-30%） |
| **コスト** | 低い | やや高い（+30-50%） |

### 5.3 カバレッジ目標の調整

```bash
# 90%カバレッジ（速度重視）
--quality-mode --target-coverage 0.90

# 95%カバレッジ（推奨）
--quality-mode --target-coverage 0.95

# 98%カバレッジ（品質最優先）
--quality-mode --target-coverage 0.98
```

---

## 6. キャッシュ機能

### 6.1 キャッシュの有効化

```bash
python a10_qa_optimized_hybrid_batch.py 
    --dataset cc_news 
    --use-cache 
    --cache-dir qa_cache 
    --model gemini-2.0-flash
```

### 6.2 キャッシュの効果

| 実行回 | 処理時間 | 効果 |
|--------|---------|------|
| 初回 | 61分 | キャッシュ作成 |
| 2回目以降 | **15-25分** | **50%短縮** |

### 6.3 キャッシュディレクトリ構造

```
qa_cache/
├── cc_news_embeddings.pkl
├── cc_news_chunks.pkl
└── cc_news_qa_pairs.pkl
```

---

## 7. 比較実行モード

### 7.1 比較モードの使用

通常版（個別処理）とバッチ版の性能を比較します:

```bash
python a10_qa_optimized_hybrid_batch.py 
    --dataset cc_news 
    --compare 
    --compare-size 10 
    --model gemini-2.0-flash
```

### 7.2 比較結果の出力

```bash
================================================================================
📊 性能比較結果
================================================================================
サンプル数: 10文書

【通常版（個別処理）】
  処理時間: 45.00秒
  API呼出: 30回
  1文書あたり: 4.50秒, 3.0回

【バッチ版（バッチ処理）】
  処理時間: 12.00秒
  API呼出: 5回
  1文書あたり: 1.20秒, 0.5回

【改善効果】
  処理時間短縮: 73.3%
  API呼出削減: 83.3%
  高速化: 3.75x
================================================================================
```

### 7.3 比較結果の保存

```
qa_output/comparison_cc_news_20251127_143052.json
```

---

## 8. コマンドラインオプション

### 8.1 全オプション一覧

| オプション | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `--dataset` | str | cc_news | データセットタイプ |
| `--model` | str | **`gemini-2.0-flash`** | 使用するLLMモデル |
| `--batch-size` | int | 10 | LLMバッチサイズ |
| `--embedding-batch-size` | int | 100 | 埋め込みバッチサイズ |
| `--max-docs` | int | None | 処理する最大文書数 |
| `--qa-count` | int | None | 文書あたりのQ/A数 |
| `--doc-type` | str | None | 文書タイプ（news/technical/academic/auto） |
| `--no-llm` | flag | False | LLMを使用しない |
| `--no-coverage` | flag | False | カバレッジ計算を行わない |
| `--output` | str | qa_output | 出力ディレクトリ |
| `--compare` | flag | False | 通常版との比較実行 |
| `--compare-size` | int | 10 | 比較実行のサンプルサイズ |
| `--quality-mode` | flag | False | 品質重視モード |
| `--target-coverage` | float | 0.95 | 目標カバレッジ率 |
| `--use-cache` | flag | False | キャッシュを使用 |
| `--cache-dir` | str | qa_cache | キャッシュディレクトリ |
| `--progressive-quality` | flag | False | 段階的品質向上モード |
| `--initial-coverage` | float | 0.85 | 初期目標カバレッジ率 |
| `--final-coverage` | float | 0.95 | 最終目標カバレッジ率 |

---

## 9. 実行方法

### 9.1 基本実行

```bash
python a10_qa_optimized_hybrid_batch.py --dataset cc_news --model gemini-2.0-flash
```

### 9.2 品質重視モード（推奨）

```bash
python a10_qa_optimized_hybrid_batch.py 
    --dataset cc_news 
    --model gemini-2.0-flash 
    --quality-mode 
    --target-coverage 0.95 
    --batch-size 10 
    --embedding-batch-size 300 
    --output qa_output
```

### 9.3 キャッシュ活用版（2回目以降）

```bash
python a10_qa_optimized_hybrid_batch.py 
    --dataset cc_news 
    --model gemini-2.0-flash 
    --quality-mode 
    --use-cache 
    --cache-dir qa_cache
```

### 9.4 段階的品質向上版

```bash
python a10_qa_optimized_hybrid_batch.py 
    --dataset cc_news 
    --model gemini-2.0-flash 
    --progressive-quality 
    --initial-coverage 0.85 
    --final-coverage 0.95 
    --batch-size 15
```

### 9.5 日本語データセット（MeCab自動利用）

```bash
# Wikipedia日本語版
python a10_qa_optimized_hybrid_batch.py --dataset wikipedia_ja --model gemini-2.0-flash

# Livedoorニュースコーパス
python a10_qa_optimized_hybrid_batch.py 
    --dataset livedoor 
    --model gemini-2.0-flash 
    --quality-mode 
    --max-docs 500 
    --batch-size 20 
    --embedding-batch-size 500 
    --use-cache 
    --cache-dir qa_cache_livedoor
```

---

## 10. 出力ファイル

### 10.1 出力ディレクトリ構造

```
qa_output/
├── a10/
│   ├── batch_summary_{dataset}_{model}_b{batch}_{timestamp}.json
│   └── batch_qa_pairs_{dataset}_{model}_b{batch}_{timestamp}.csv
└── a10_qa_pairs_{dataset}.csv    # 統一フォーマット
```

### 10.2 サマリーファイル（JSON）

```json
{
  "dataset_type": "cc_news",
  "dataset_name": "CC-News英語ニュース",
  "model_used": "gemini-2.0-flash",
  "batch_processing": true,
  "batch_sizes": {
    "llm_batch_size": 10,
    "embedding_batch_size": 300
  },
  "documents_processed": 497,
  "total_qa_generated": 2485,
  "avg_qa_per_doc": 5.0,
  "processing_time": {
    "total_seconds": 3678,
    "minutes": 61.3,
    "docs_per_second": 0.135
  },
  "api_usage": {
    "total_cost": 0.18, 
    "cost_per_doc": 0.00036,
    "batch_statistics": {
      "total_llm_calls": 110,
      "total_embedding_calls": 10
    }
  },
  "coverage": {
    "calculated": true,
    "avg_coverage": 95.2,
    "min_coverage": 87.0,
    "max_coverage": 99.0
  },
  "generation_timestamp": "2025-11-27T14:30:52"
}
```

### 10.3 Q/Aペアファイル（CSV）

**全カラム版（batch_qa_pairs_*.csv）**:
```csv
doc_id,question,answer,doc_title,text_length
cc_news_0,What is the main topic?,AI technology...,Article Title,1234
cc_news_1,How does it work?,It uses...,Another Title,2345
```

**統一フォーマット版（a10_qa_pairs_{dataset}.csv）**:
```csv
question,answer
What is the main topic?,AI technology...
How does it work?,It uses...
```

---

## 11. パフォーマンス

### 11.1 実行時間見積もり

| データセット | 文書数 | バッチサイズ | 実行時間 | コスト | カバレッジ |
|------------|--------|------------|---------|--------|----------|
| cc_news | 497 | 10 | 61分 | **低** | 95% |
| livedoor | 500 | 20 | 30-50分 | **低** | 95% |
| livedoor（キャッシュ） | 500 | 20 | 15-25分 | **極低** | 95% |

### 11.2 バッチ処理の効果

```
バッチ処理により以下の改善を実現：

1. **API呼び出し削減**
   - 通常版（推定）: 1491回
   - バッチ版（実際）: 110回
   - 削減率: 92.6%

2. **処理速度向上**
   - 処理速度: 0.14文書/秒
   - 497文書を61.3分で処理

3. **スケーラビリティ**
   - 大規模データセット処理が現実的に
   - レート制限リスクの大幅低減
```

---

## 12. トラブルシューティング

### 12.1 APIキーエラー

**症状**: `Google APIキーが設定されていません`

**解決策**:
```bash
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```

### 12.2 ファイルが見つからない

**症状**: `FileNotFoundError: ファイルが見つかりません`

**解決策**:
```bash
ls OUTPUT/preprocessed_cc_news.csv
```

### 12.3 レート制限エラー

**症状**: `Resource Exhausted` (Gemini APIの場合) または `RateLimitError`

**解決策**:
```bash
# LLMバッチサイズを小さくする
--batch-size 5

# 埋め込みバッチサイズを小さくする (helper_embedding.pyの設定を確認)
--embedding-batch-size 50
```

### 12.4 カバレッジが目標に達しない

**解決策**:
```bash
# 方法1: 品質重視モードを使用
--quality-mode

# 方法2: LLMバッチサイズを小さくする（品質向上を優先）
--batch-size 5

# 方法3: 埋め込みバッチサイズを調整する (helper_embedding.pyの設定を確認)
--embedding-batch-size 50
```

### 12.5 MeCabが利用できない

**症状**: `⚠️ MeCabが利用できません（正規表現にフォールバック）`

**影響**: 日本語の文境界検出精度がやや低下（機能には影響なし）

**解決策（オプション）**:
```bash
# macOS
brew install mecab mecab-ipadic
pip install mecab-python3

# Ubuntu/Debian
sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8
pip install mecab-python3
```

---

## 付録: 実行ログサンプル

```
=====================================
バッチ処理版ハイブリッドQ&A生成
=====================================
データセット: CC-News英語ニュース
モデル: gemini-2.0-flash
バッチサイズ: LLM=10, 埋め込み=300
出力先: qa_output
最大文書数: 制限なし

[1/3] データ読み込み...
2025-11-27 14:30:00 - INFO - データ読み込み中: OUTPUT/preprocessed_cc_news.csv
2025-11-27 14:30:01 - INFO - 読み込み完了: 497件のデータ

[2/3] バッチ処理Q/A生成...
🎯 品質重視モード: 目標カバレージ 95%
2025-11-27 14:30:01 - INFO - バッチ処理Q/A生成開始: 497件の文書
2025-11-27 14:30:01 - INFO - バッチサイズ: LLM=10, 埋め込み=300
2025-11-27 14:30:01 - INFO - データセット言語: en
2025-11-27 14:30:01 - INFO -   → 英語データセット: 正規表現ベースの文分割を使用

バッチ処理中: 100%|██████████| 50/50 [61:18<00:00, 73.57s/batch]

[3/3] 結果保存...

=====================================
処理完了
=====================================
処理文書数: 497
生成Q/A総数: 2485
平均Q/A数/文書: 5.0

処理時間:
- 合計: 3678.00秒
- 分: 61.30分
- 処理速度: 0.14文書/秒

API使用状況:
- LLM呼び出し: 110回
- 埋め込み呼び出し: 10回
- 総コスト: $0.1800

カバレージ:
- 平均: 95.2%
- 最小: 87.0%
- 最大: 99.0%

保存ファイル:
- サマリー: qa_output/a10/batch_summary_cc_news_gemini_2_0_flash_b10_20251127_153119.json
- Q/A CSV: qa_output/a10/batch_qa_pairs_cc_news_gemini_2_0_flash_b10_20251127_153119.csv
- 統一フォーマット: qa_output/a10_qa_pairs_cc_news.csv
```

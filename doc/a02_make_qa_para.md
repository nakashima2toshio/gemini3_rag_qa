# a02_make_qa_para.py - 改善版Q/Aペア自動生成システム

## 目次

1. [概要](#1-概要)
   - [1.1 目的](#11-目的)
   - [1.2 起動コマンド](#12-起動コマンド)
   - [1.3 主要機能](#13-主要機能)
   - [1.4 対応データセット](#14-対応データセット)
   - [1.5 処理モード比較](#15-処理モード比較)
   - [1.6 関連ドキュメント](#16-関連ドキュメント)
2. [アーキテクチャ](#2-アーキテクチャ)
   - [2.1 システム構成図](#21-システム構成図)
   - [2.2 処理フロー図](#22-処理フロー図)
   - [2.3 依存モジュール](#23-依存モジュール)
   - [2.4 データセット拡張設定](#24-データセット拡張設定)
3. [キーワード抽出・複雑度分析](#3-キーワード抽出複雑度分析)
   - [3.1 KeywordExtractorクラス](#31-keywordextractorクラス)
   - [3.2 複雑度分析関数](#32-複雑度分析関数)
   - [3.3 主要概念抽出](#33-主要概念抽出)
4. [セマンティックチャンク分割](#4-セマンティックチャンク分割)
   - [4.1 チャンク作成関数](#41-チャンク作成関数)
   - [4.2 文書チャンク作成](#42-文書チャンク作成)
   - [4.3 小チャンク統合](#43-小チャンク統合)
   - [4.4 SemanticCoverageクラス詳細](#44-semanticcoverageクラス詳細)
5. [プロンプト設計](#5-プロンプト設計)
   - [5.1 2段階プロンプト構造](#51-2段階プロンプト構造)
   - [5.2 システムプロンプト設計](#52-システムプロンプト設計)
   - [5.3 ユーザープロンプト構築](#53-ユーザープロンプト構築)
   - [5.4 質問タイプ階層構造](#54-質問タイプ階層構造)
   - [5.5 動的Q/A数決定ロジック](#55-動的qa数決定ロジック)
   - [5.6 JSON出力フォーマット仕様](#56-json出力フォーマット仕様)
6. [Q/Aペア生成](#6-qaペア生成)
   - [6.1 バッチ処理](#61-バッチ処理)
   - [6.2 単一チャンク処理](#62-単一チャンク処理)
   - [6.3 データセット全体処理](#63-データセット全体処理)
7. [API呼び出し方式](#7-api呼び出し方式)
   - [7.1 構造化出力API（client.responses.parse）](#71-構造化出力apiclientresponsesparse)
   - [7.2 Chat Completions API（フォールバック）](#72-chat-completions-apiフォールバック)
   - [7.3 モデル別パラメータ制約](#73-モデル別パラメータ制約)
   - [7.4 Pydanticモデル定義](#74-pydanticモデル定義)
8. [Celery非同期並列処理](#8-celery非同期並列処理)
   - [8.1 システム構成図](#81-システム構成図)
   - [8.2 ワーカー管理](#82-ワーカー管理)
   - [8.3 ワーカー起動・管理コマンド](#83-ワーカー起動管理コマンド)
   - [8.4 並列タスク投入・結果収集](#84-並列タスク投入結果収集)
   - [8.5 結果収集メカニズム（Redis直接アクセス）](#85-結果収集メカニズムredis直接アクセス)
   - [8.6 リトライ・エラーハンドリング](#86-リトライエラーハンドリング)
   - [8.7 主要ファイル](#87-主要ファイル)
9. [カバレージ分析](#9-カバレージ分析)
   - [9.1 データセット別最適閾値](#91-データセット別最適閾値)
   - [9.2 多段階カバレージ分析](#92-多段階カバレージ分析)
   - [9.3 チャンク特性別分析](#93-チャンク特性別分析)
   - [9.4 メイン分析関数](#94-メイン分析関数)
10. [出力とファイル保存](#10-出力とファイル保存)
    - [10.1 出力形式](#101-出力形式)
    - [10.2 ファイル命名規則](#102-ファイル命名規則)
    - [10.3 メタデータ付与](#103-メタデータ付与)
    - [10.4 保存ディレクトリ構造](#104-保存ディレクトリ構造)
11. [コマンドラインオプション](#11-コマンドラインオプション)
    - [11.1 全オプション一覧](#111-全オプション一覧)
    - [11.2 入力ソース](#112-入力ソース)
12. [実行方法](#12-実行方法)
    - [12.1 環境準備](#121-環境準備)
    - [12.2 テスト実行](#122-テスト実行)
    - [12.3 Celery並列実行](#123-celery並列実行)
    - [12.4 UIからの実行（Streamlit）](#124-uiからの実行streamlit)
    - [12.5 実行時の進捗表示](#125-実行時の進捗表示)
    - [12.6 実行時間の見積もり](#126-実行時間の見積もり)
13. [トラブルシューティング](#13-トラブルシューティング)
14. [次ステップ](#14-次ステップ)
    - [14.1 Qdrantへの登録](#141-qdrantへの登録)
    - [14.2 検索処理との連携](#142-検索処理との連携)
15. [付録](#15-付録)
    - [15.1 データ読み込み関数](#151-データ読み込み関数)
    - [15.2 コード参照一覧](#152-コード参照一覧)
16. [参考資料](#16-参考資料)

---

## 1. 概要

### 1.1 目的

`a02_make_qa_para.py`は、OUTPUTフォルダ内のpreprocessedファイルから高品質なQ/Aペアを自動生成するシステムです。バッチ処理による並列化でAPI呼び出し回数を大幅削減し、Celeryによる非同期並列処理をサポートします。

### 1.2 起動コマンド

```bash
# 基本実行（同期処理）
python a02_make_qa_para.py --dataset livedoor --model gpt-4o-mini --max-docs 20

# Celery並列処理
python a02_make_qa_para.py --dataset cc_news --use-celery --celery-workers 24 --batch-chunks 3
```

### 1.3 主要機能

- **セマンティック分割によるチャンク作成**（段落境界を優先）
- **バッチ処理による並列Q/A生成**（1-5チャンク同時処理）
- **Celeryによる非同期並列処理**（複数ワーカーで同時実行）
- **小チャンク自動統合による効率化**
- **多段階カバレージ分析**（strict/standard/lenient）
- **チャンク特性別カバレージ分析**（長さ別・位置別）

### 1.4 対応データセット

| データセット | キー | 言語 | 説明 |
|------------|------|------|------|
| CC-News | `cc_news` | 英語 | 英語ニュース記事（7,376件） |
| CC100日本語 | `japanese_text` | 日本語 | Webテキストコーパス |
| Wikipedia日本語版 | `wikipedia_ja` | 日本語 | 百科事典的知識 |
| Livedoorニュース | `livedoor` | 日本語 | ニュースコーパス（7,376件） |

### 1.5 処理モード比較

| モード | API呼び出し | 実行時間 | 効率化率 | 推奨用途 |
|--------|------------|---------|---------|---------|
| 同期処理 | 1800回 | 180分 | 1.0x | 小規模テスト |
| Celery並列 | 1800回 | 23分 | 7.8x | 中規模処理 |
| **ハイブリッド** | **600回** | **8分** | **22.5x** | **大規模処理（推奨）** |

### 1.6 関連ドキュメント

本ドキュメントは以下のドキュメント群の一部です：

| ドキュメント | 焦点 | 内容 |
|-------------|------|------|
| `doc/03_chunk.md` | チャンク分割技術 | SemanticCoverage、文分割、MeCab |
| `doc/04_prompt.md` | プロンプト設計 | 2段階構造、言語別対応、質問タイプ階層 |
| `doc/05_qa_pair.md` | 実行・処理フロー | 並列処理、Celery、出力、カバレージ |
| `doc/06_embedding_qdrant.md` | ベクトル化・DB登録 | Embedding、Qdrant、類似度検索 |
| **`doc/a02_make_qa_para.md`（本書）** | **実装詳細** | **a02スクリプトの全機能解説** |

---

## 2. アーキテクチャ

### 2.1 システム構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                    a02_make_qa_para.py                          │
├─────────────────────────────────────────────────────────────────┤
│  [1] データ読み込み                                              │
│      load_preprocessed_data() / load_uploaded_file()            │
│                              │                                  │
│                              ▼                                  │
│  [2] チャンク作成                                                │
│      create_document_chunks() → create_semantic_chunks()        │
│                              │                                  │
│                              ▼                                  │
│  [3] チャンク統合（オプション）                                    │
│      merge_small_chunks()                                       │
│                              │                                  │
│                              ▼                                  │
│  [4] Q/A生成                                                    │
│      ┌──────────────────┬──────────────────┐                    │
│      │  同期処理          │  Celery並列       │                  │
│      │  generate_qa_*()  │  submit_parallel_qa_generation()    │
│      └──────────────────┴──────────────────┘                    │
│                              │                                  │
│                              ▼                                  │
│  [5] カバレージ分析                                              │
│      analyze_coverage() → multi_threshold_coverage()           │
│                              │                                  │
│                              ▼                                  │
│  [6] 結果保存                                                    │
│      save_results() → qa_output/a02/                           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 処理フロー図

```
[入力データ]
    │
    ├── データセット（cc_news, livedoor等）
    └── ローカルファイル（CSV, JSON, TXT）
    │
    ▼
[1. データ読み込み]  ←── load_preprocessed_data() / load_uploaded_file()
    │
    ▼
[2. チャンク作成]  ←── create_document_chunks()
    │                    └── SemanticCoverage.create_semantic_chunks()
    │                        （詳細は doc/03_chunk.md 参照）
    │
    ▼
[3. チャンク前処理]  ←── merge_small_chunks()
    │                    小チャンク統合（min_tokens未満を統合）
    │
    ▼
[4. Q/A生成]  ←── 同期 or 非同期（Celery）
    │              （プロンプト詳細は doc/04_prompt.md 参照）
    │
    ├── 【同期処理】generate_qa_for_dataset()
    │       └── generate_qa_pairs_for_batch()
    │
    └── 【非同期処理】Celery並列処理
            ├── submit_parallel_qa_generation()
            └── collect_results()（Redis直接アクセス）
    │
    ▼
[5. カバレージ分析]  ←── analyze_coverage()（オプション）
    │
    ▼
[6. 結果保存]  ←── save_results()
    │
    ▼
[出力ファイル]
    ├── qa_pairs_{dataset}_{timestamp}.json
    ├── qa_pairs_{dataset}_{timestamp}.csv
    └── a02_qa_pairs_{dataset}.csv（統一フォーマット）
    │
    ▼
[7. Qdrant登録]  ←── doc/06_embedding_qdrant.md 参照
```

### 2.3 依存モジュール

```python
# 共通モジュール
from models import QAPairsResponse
from config import DATASET_CONFIGS, QAGenerationConfig

# ローカルモジュール
from a03_rag_qa_coverage_improved import SemanticCoverage
from helper_rag_qa import SemanticCoverage as SemanticChunker
from helper_rag import clean_text

# Celeryタスク（オプション）
from celery_tasks import submit_parallel_qa_generation, collect_results
```

### 2.4 データセット拡張設定

```python
_LOCAL_DATASET_EXTENSIONS = {
    "cc_news": {
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "en",
    },
    "japanese_text": {
        "text_column": "Combined_Text",
        "title_column": None,
        "lang": "ja",
    },
    "wikipedia_ja": {
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja",
    },
    "livedoor": {
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja",
    }
}
```

---

## 3. キーワード抽出・複雑度分析

### 3.1 KeywordExtractorクラス

MeCabと正規表現を統合したキーワード抽出クラスです。MeCabが利用可能な場合は複合名詞抽出を優先し、利用不可の場合は正規表現版に自動フォールバックします。

```python
# a02_make_qa_para.py:277-403
class KeywordExtractor:
    def __init__(self, prefer_mecab: bool = True):
        """MeCab優先設定"""

    def extract(self, text: str, top_n: int = 5) -> List[str]:
        """キーワード抽出（自動フォールバック対応）"""

    def _extract_with_mecab(self, text: str, top_n: int) -> List[str]:
        """MeCabによる複合名詞抽出"""

    def _extract_with_regex(self, text: str, top_n: int) -> List[str]:
        """正規表現によるキーワード抽出"""
```

**ストップワード**:
```python
self.stopwords = {
    'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん',
    'ます', 'です', 'ある', 'いる', 'する', 'なる', 'できる',
    'いう', '的', 'な', 'に', 'を', 'は', 'が', 'で', 'と',
    'の', 'から', 'まで', '等', 'など', 'よる', 'おく', 'くる'
}
```

### 3.2 複雑度分析関数

```python
# a02_make_qa_para.py:410-457
def analyze_chunk_complexity(chunk_text: str, lang: str = "ja") -> Dict:
    """チャンクの複雑度を分析

    Returns:
        {
            "complexity_level": "high" | "medium" | "low",
            "technical_terms": List[str],  # 上位10個
            "avg_sentence_length": float,
            "concept_density": float,
            "sentence_count": int,
            "token_count": int
        }
    """
```

**複雑度レベル判定**:

| レベル | 条件 |
|--------|------|
| high | 概念密度 > 5% OR 平均文長 > 30トークン |
| medium | 概念密度 > 2% OR 平均文長 > 20トークン |
| low | その他 |

### 3.3 主要概念抽出

```python
# a02_make_qa_para.py:459-481
def extract_key_concepts(chunk_text: str, lang: str = "ja", top_n: int = 5) -> List[str]:
    """チャンクから主要概念を抽出
    KeywordExtractorと複雑度分析の結果を統合
    """
```

---

## 4. セマンティックチャンク分割

### 4.1 チャンク作成関数

```python
# a02_make_qa_para.py:487-538
def create_semantic_chunks(
    text: str,
    lang: str = "ja",
    max_tokens: int = 200,
    chunk_id_prefix: str = "chunk"
) -> List[Dict]:
    """
    セマンティック分割によるチャンク作成（段落優先）

    helper_rag_qa.pyのSemanticCoverage.create_semantic_chunks()を使用し、
    段落境界を最優先したセマンティック分割を実行。

    Returns:
        [{
            'id': str,
            'text': str,
            'tokens': int,
            'type': 'paragraph' | 'sentence_group' | 'forced_split',
            'sentences': List[str]
        }, ...]
    """
```

### 4.2 文書チャンク作成

```python
# a02_make_qa_para.py:756-816
def create_document_chunks(
    df: pd.DataFrame,
    dataset_type: str,
    max_docs: Optional[int] = None,
    config: Optional[Dict] = None
) -> List[Dict]:
    """DataFrameから文書チャンクを作成（セマンティック分割）

    各チャンクに付加されるメタデータ:
    - doc_id: 文書ID
    - doc_idx: 文書インデックス
    - chunk_idx: チャンクインデックス
    - dataset_type: データセットタイプ
    """
```

### 4.3 小チャンク統合

```python
# a02_make_qa_para.py:819-875
def merge_small_chunks(
    chunks: List[Dict],
    min_tokens: int = 150,
    max_tokens: int = 400
) -> List[Dict]:
    """小さいチャンクを統合して適切なサイズにする

    - min_tokens未満のチャンクは統合対象
    - 同じ文書からのチャンクのみ統合
    - 統合後のトークン数がmax_tokensを超えない範囲で統合
    """
```

### 4.4 SemanticCoverageクラス詳細

**詳細は `doc/03_chunk.md` を参照**

`helper_rag_qa.py`のSemanticCoverageクラスが提供する主要メソッド：

| メソッド | 役割 |
|---------|------|
| `create_semantic_chunks()` | メイン分割処理（段落優先） |
| `_split_sentences_mecab()` | MeCabによる文分割（日本語） |
| `_split_sentences_regex()` | 正規表現による文分割（フォールバック） |
| `_chunk_by_paragraphs()` | 段落境界でのチャンク分割 |
| `_adjust_chunks_for_topic_continuity()` | トピック連続性調整 |

**チャンクタイプ**:

| タイプ | 説明 |
|--------|------|
| `paragraph` | 段落境界で分割されたチャンク |
| `sentence_group` | 複数文をグループ化したチャンク |
| `forced_split` | max_tokens超過により強制分割されたチャンク |

---

## 5. プロンプト設計

**詳細は `doc/04_prompt.md` を参照**

### 5.1 2段階プロンプト構造

プロンプトは**システムプロンプト**と**ユーザープロンプト**の2段階で構成される。

| 構成要素 | 役割 |
|---------|------|
| システムプロンプト | LLMの役割定義、生成ルール明示 |
| ユーザープロンプト | チャンクテキスト、質問タイプ指示、JSON出力形式 |

### 5.2 システムプロンプト設計

**日本語版**:
```python
# a02_make_qa_para.py:946-953
system_prompt = """あなたは教育コンテンツ作成の専門家です。
複数の日本語テキストから、学習効果の高いQ&Aペアを生成してください。

生成ルール:
1. 質問は明確で具体的に
2. 回答は簡潔で正確に（1-2文程度）
3. テキストの内容に忠実に
4. 多様な観点から質問を作成"""
```

**英語版**:
```python
# a02_make_qa_para.py:993-1000
system_prompt = """You are an expert in educational content creation.
Generate high-quality Q&A pairs from multiple English texts.

Generation rules:
1. Questions should be clear and specific
2. Answers should be concise and accurate (1-2 sentences)
3. Stay faithful to the text content
4. Create questions from diverse perspectives"""
```

### 5.3 ユーザープロンプト構築

ユーザープロンプトは以下の要素で構成：

1. **タスク指示**: 生成するQ/Aペア数の指定
2. **テキスト**: チャンクテキスト（バッチ処理時は複数）
3. **質問タイプ指示**: 使用する質問タイプの説明
4. **JSON出力形式**: 期待する出力形式の指定

```python
# a02_make_qa_para.py:972-990
user_prompt = f"""以下の{len(chunks)}個のテキストから、合計{total_pairs}個のQ&Aペアを生成してください。
{combined_text}

質問タイプ:
- fact: 事実確認型（〜は何ですか？）
- reason: 理由説明型（なぜ〜ですか？）
- comparison: 比較型（〜と〜の違いは？）
- application: 応用型（〜はどのように活用されますか？）

JSON形式で出力:
{{
  "qa_pairs": [
    {{
      "question": "質問文",
      "answer": "回答文",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""
```

### 5.4 質問タイプ階層構造

認知レベルに基づく3階層構造（`config.py`で定義）：

| 階層 | タイプ | 説明 |
|------|--------|------|
| **basic** | definition | 定義型（〜は何ですか？） |
| | identification | 識別型（〜を特定してください） |
| | enumeration | 列挙型（〜を列挙してください） |
| **understanding** | cause_effect | 因果関係型（なぜ〜ですか？） |
| | process | プロセス型（どのような手順で〜？） |
| | mechanism | メカニズム型（どのように機能する？） |
| | comparison | 比較型（〜と〜の違いは？） |
| **application** | synthesis | 統合型（〜を組み合わせると？） |
| | evaluation | 評価型（〜の効果は？） |
| | prediction | 予測型（〜の結果は？） |
| | practical | 実践型（どのように活用する？） |

**プロンプト内では簡略化した4タイプを使用**:
- `fact`: 事実確認型
- `reason`: 理由説明型
- `comparison`: 比較型
- `application`: 応用型

### 5.5 動的Q/A数決定ロジック

```python
# a02_make_qa_para.py:882-913
def determine_qa_count(chunk: Dict, config: Dict) -> int:
    """チャンクに最適なQ/A数を決定（動的調整）"""
```

| トークン数 | 基本Q/A数 | 備考 |
|-----------|----------|------|
| < 50 | 2個 | 短いチャンクでも最低2個 |
| 50-100 | 3個 | Shortチャンク強化 |
| 100-200 | base + 1 | Mediumチャンク |
| 200-300 | base + 2 | Longチャンク |
| > 300 | base + 3 | 超長文（上限8個） |

**位置バイアス補正**: 文書後半（6番目以降のチャンク）は+1個追加

### 5.6 JSON出力フォーマット仕様

```json
{
  "qa_pairs": [
    {
      "question": "質問文（必須）",
      "answer": "回答文（必須）",
      "question_type": "fact/reason/comparison/application（必須）"
    }
  ]
}
```

**フィールド仕様**:

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| question | string | ✓ | 質問文テキスト |
| answer | string | ✓ | 回答文テキスト（1-2文程度） |
| question_type | string | ✓ | fact/reason/comparison/application |

---

## 6. Q/Aペア生成

### 6.1 バッチ処理

```python
# a02_make_qa_para.py:916-1103
def generate_qa_pairs_for_batch(
    chunks: List[Dict],
    config: Dict,
    model: str = "gpt-4o-mini",
    client: Optional[OpenAI] = None
) -> List[Dict]:
    """複数チャンクから一度にQ/Aペアを生成（バッチ処理対応）

    - 1-5チャンクを1つのプロンプトに統合
    - OpenAI Responses API (client.responses.parse) を使用
    - QAPairsResponse Pydanticモデルで型安全な出力
    """
```

**バッチ処理の効果**:

| バッチサイズ | API呼び出し削減率 |
|------------|-----------------|
| 1 | 0% |
| 2 | 50% |
| **3** | **67%（推奨）** |
| 4 | 75% |
| 5 | 80% |

### 6.2 単一チャンク処理

```python
# a02_make_qa_para.py:1106-1291
def generate_qa_pairs_for_chunk(
    chunk: Dict,
    config: Dict,
    model: str = "gpt-4o-mini",
    client: Optional[OpenAI] = None
) -> List[Dict]:
    """単一チャンクからQ/Aペアを生成（後方互換性のため維持）"""
```

### 6.3 データセット全体処理

```python
# a02_make_qa_para.py:1294-1394
def generate_qa_for_dataset(
    chunks: List[Dict],
    dataset_type: str,
    model: str = "gpt-4o-mini",
    chunk_batch_size: int = 3,
    merge_chunks: bool = True,
    min_tokens: int = 150,
    max_tokens: int = 400,
    config: Optional[Dict] = None
) -> List[Dict]:
    """データセット全体のQ/Aペア生成

    - チャンク統合 → バッチ分割 → API呼び出し
    - 最大3回リトライ
    - フォールバック: バッチ失敗時は個別処理
    """
```

---

## 7. API呼び出し方式

### 7.1 構造化出力API（client.responses.parse）

**主要API（推奨）** - Pydanticモデルを使用した型安全な出力

```python
# a02_make_qa_para.py:1044-1049
response = client.responses.parse(
    input=combined_input,           # システムプロンプト + ユーザープロンプト
    model=model,                    # "gpt-4o-mini", "gpt-5-mini" 等
    text_format=QAPairsResponse,    # Pydanticモデル
    max_output_tokens=4000          # バッチ処理時は4000、単一は1000
)
```

**レスポンス解析**:
```python
# a02_make_qa_para.py:1052-1082
for output in response.output:
    if output.type == "message":
        for item in output.content:
            if item.type == "output_text" and item.parsed:
                parsed_data = item.parsed  # QAPairsResponse型
                for qa_data in parsed_data.qa_pairs:
                    qa = {
                        "question": qa_data.question,
                        "answer": qa_data.answer,
                        "question_type": qa_data.question_type,
                        ...
                    }
```

### 7.2 Chat Completions API（フォールバック）

構造化出力APIが失敗した場合のフォールバック。

```python
# celery_tasks.py:329-381
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    max_tokens=max_tokens,
    temperature=0.7,
    response_format={"type": "json_object"}
)
```

### 7.3 モデル別パラメータ制約

| モデル | temperatureサポート | トークン制限パラメータ |
|--------|-------------------|---------------------|
| GPT-4o系 | ✓ | max_tokens |
| GPT-5系 | ✗ | max_completion_tokens |
| O-series (o1, o3, o4) | ✗ | max_completion_tokens |

**temperature非対応モデルの判定**:
```python
# doc/04_prompt.md参照
NO_TEMPERATURE_MODELS = ['o1-', 'o3-', 'o4-', 'gpt-5']

def supports_temperature(model: str) -> bool:
    return not any(model.startswith(prefix) for prefix in NO_TEMPERATURE_MODELS)
```

### 7.4 Pydanticモデル定義

```python
# models.py
from pydantic import BaseModel, Field
from typing import List, Optional

class QAPair(BaseModel):
    """Q/Aペア単体"""
    question: str = Field(..., description="質問文")
    answer: str = Field(..., description="回答文")
    question_type: str = Field(default="fact", description="質問タイプ")

class QAPairsResponse(BaseModel):
    """Q/Aペアリストのレスポンス"""
    qa_pairs: List[QAPair] = Field(default_factory=list)
```

---

## 8. Celery非同期並列処理

### 8.1 システム構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                    Celery並列処理アーキテクチャ                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [a02_make_qa_para.py]                                          │
│         │                                                       │
│         │ submit_parallel_qa_generation()                       │
│         ▼                                                       │
│  ┌─────────────┐                                                │
│  │   Redis     │◄─── Broker (タスクキュー)                      │
│  │  (6379)     │◄─── Backend (結果格納)                         │
│  └─────────────┘                                                │
│         │                                                       │
│         │ タスク配信                                             │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Celeryワーカープール                      ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        ││
│  │  │Worker 1 │  │Worker 2 │  │Worker 3 │  │Worker N │ ...    ││
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘        ││
│  │       │            │            │            │              ││
│  │       ▼            ▼            ▼            ▼              ││
│  │  [OpenAI API] [OpenAI API] [OpenAI API] [OpenAI API]        ││
│  └─────────────────────────────────────────────────────────────┘│
│         │                                                       │
│         │ 結果格納                                               │
│         ▼                                                       │
│  ┌─────────────┐                                                │
│  │   Redis     │ celery-task-meta-{task_id}                     │
│  └─────────────┘                                                │
│         │                                                       │
│         │ collect_results()（Redis直接アクセス）                  │
│         ▼                                                       │
│  [Q/Aペア結果]                                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 ワーカー管理

```python
# a02_make_qa_para.py:1401-1459
def check_celery_workers(required_workers: int = 8) -> bool:
    """Celeryワーカーの状態を確認（リトライ機能付き）

    - 最大3回リトライ
    - ワーカー数不足でも続行可能
    """
```

### 8.3 ワーカー起動・管理コマンド

```bash
# 起動
./start_celery.sh start -w 8

# ステータス確認
./start_celery.sh status

# 停止
./start_celery.sh stop

# 再起動（推奨）
redis-cli FLUSHDB
./start_celery.sh restart -w 24
```

### 8.4 並列タスク投入・結果収集

```python
from celery_tasks import submit_parallel_qa_generation, collect_results

# タスク投入
tasks = submit_parallel_qa_generation(
    processed_chunks, config, model, batch_size=3
)

# 結果収集（タイムアウト計算: タスク数 × 10秒、最低600秒、最大1800秒）
timeout_seconds = min(max(len(tasks) * 10, 600), 1800)
qa_pairs = collect_results(tasks, timeout=timeout_seconds)
```

### 8.5 結果収集メカニズム（Redis直接アクセス）

Celeryの`task.get()`では、タスク状態が`PENDING`と誤認識される場合がある。
Redis直接アクセスにより、確実に結果を取得できる。

```python
# celery_tasks.py:688-920
def collect_results(tasks, timeout):
    redis_client = redis.Redis(host='localhost', port=6379, db=0)

    for task in tasks:
        # Redis直接アクセス
        redis_key = f"celery-task-meta-{task.id}"
        redis_data = redis_client.get(redis_key)

        if redis_data:
            result = json.loads(redis_data)
            if result['status'] == 'SUCCESS':
                qa_pairs.extend(result['result']['qa_pairs'])
```

**Redis直接アクセスのメリット**:
- タスク状態の誤認識を回避
- 高速な結果取得
- 確実なエラー検出

### 8.6 リトライ・エラーハンドリング

**タスクレベルのリトライ**:
```python
# celery_tasks.py
@app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    soft_time_limit=300,
    time_limit=360
)
def generate_qa_task(self, chunk_data, config, model):
    try:
        # Q/A生成処理
        ...
    except Exception as e:
        self.retry(exc=e, countdown=5)
```

**バッチレベルのフォールバック**:
```python
# a02_make_qa_para.py:1367-1382
except Exception as e:
    if attempt == max_retries - 1:
        # 最終試行失敗時は個別処理にフォールバック
        logger.info("個別処理にフォールバック...")
        for chunk in batch:
            try:
                qa_pairs = generate_qa_pairs_for_chunk(chunk, config, model, client)
                ...
```

### 8.7 主要ファイル

| ファイル | 役割 |
|---------|------|
| `celery_tasks.py` | タスク定義とワーカー設定 |
| `a02_make_qa_para.py` | タスク投入と結果収集 |
| `start_celery.sh` | ワーカー管理スクリプト |

---

## 9. カバレージ分析

### 9.1 データセット別最適閾値

```python
# a02_make_qa_para.py:1467-1488
OPTIMAL_THRESHOLDS = {
    "cc_news": {
        "strict": 0.80,
        "standard": 0.70,
        "lenient": 0.60
    },
    "japanese_text": {
        "strict": 0.75,
        "standard": 0.65,
        "lenient": 0.55
    },
    "wikipedia_ja": {
        "strict": 0.85,
        "standard": 0.75,
        "lenient": 0.65
    },
    "livedoor": {
        "strict": 0.78,
        "standard": 0.68,
        "lenient": 0.58
    }
}
```

### 9.2 多段階カバレージ分析

```python
# a02_make_qa_para.py:1505-1539
def multi_threshold_coverage(
    coverage_matrix: np.ndarray,
    chunks: List[Dict],
    qa_pairs: List[Dict],
    thresholds: Dict[str, float]
) -> Dict:
    """複数閾値でカバレージを評価

    Returns:
        {
            "strict": {
                "threshold": float,
                "covered_chunks": int,
                "coverage_rate": float,
                "uncovered_count": int,
                "uncovered_chunks": List[Dict]
            },
            "standard": {...},
            "lenient": {...}
        }
    """
```

### 9.3 チャンク特性別分析

```python
# a02_make_qa_para.py:1542-1644
def analyze_chunk_characteristics_coverage(
    chunks: List[Dict],
    coverage_matrix: np.ndarray,
    qa_pairs: List[Dict],
    threshold: float = 0.7
) -> Dict:
    """チャンク特性別のカバレージ分析

    Returns:
        {
            "by_length": {
                "short": {"count", "covered", "avg_similarity", "coverage_rate"},
                "medium": {...},
                "long": {...}
            },
            "by_position": {
                "beginning": {...},
                "middle": {...},
                "end": {...}
            },
            "summary": {
                "total_chunks": int,
                "total_qa_pairs": int,
                "threshold_used": float,
                "insights": List[str]
            }
        }
    """
```

**長さ別分類**:
- short: < 100トークン
- medium: 100-200トークン
- long: >= 200トークン

**位置別分類**:
- beginning: 前半33%
- middle: 中盤33%
- end: 後半33%

### 9.4 メイン分析関数

```python
# a02_make_qa_para.py:1647-1769
def analyze_coverage(
    chunks: List[Dict],
    qa_pairs: List[Dict],
    dataset_type: str = "wikipedia_ja",
    custom_threshold: Optional[float] = None
) -> Dict:
    """生成されたQ/Aペアのカバレージを分析

    - 埋め込み生成（バッチAPI最適化）
    - カバレージ行列計算
    - 多段階カバレージ分析
    - チャンク特性別分析
    """
```

---

## 10. 出力とファイル保存

### 10.1 出力形式

| 形式 | ファイル | 内容 |
|------|---------|------|
| JSON | `qa_pairs_{dataset}_{timestamp}.json` | Q/Aペア全データ |
| CSV（詳細） | `qa_pairs_{dataset}_{timestamp}.csv` | 全カラム含む |
| CSV（統一） | `a02_qa_pairs_{dataset}.csv` | question/answerのみ |
| JSON | `coverage_{dataset}_{timestamp}.json` | カバレージ分析結果 |
| JSON | `summary_{dataset}_{timestamp}.json` | サマリー情報 |

### 10.2 ファイル命名規則

```
qa_output/a02/
├── qa_pairs_{dataset}_{YYYYMMDD_HHMMSS}.json
├── qa_pairs_{dataset}_{YYYYMMDD_HHMMSS}.csv
├── coverage_{dataset}_{YYYYMMDD_HHMMSS}.json
└── summary_{dataset}_{YYYYMMDD_HHMMSS}.json

qa_output/
└── a02_qa_pairs_{dataset}.csv  # 統一フォーマット（Qdrant登録用）
```

### 10.3 メタデータ付与

各Q/Aペアに付与されるメタデータ：

| フィールド | 説明 |
|-----------|------|
| question | 質問文 |
| answer | 回答文 |
| question_type | 質問タイプ（fact/reason/comparison/application） |
| source_chunk_id | ソースチャンクID |
| doc_id | 文書ID |
| dataset_type | データセットタイプ |
| chunk_idx | チャンクインデックス |

### 10.4 保存ディレクトリ構造

```
qa_output/
├── a02/                        # a02スクリプト専用出力
│   ├── qa_pairs_*.json
│   ├── qa_pairs_*.csv
│   ├── coverage_*.json
│   └── summary_*.json
├── a02_qa_pairs_cc_news.csv    # 統一フォーマット（Qdrant登録用）
├── a02_qa_pairs_livedoor.csv
└── a02_qa_pairs_wikipedia_ja.csv
```

---

## 11. コマンドラインオプション

### 11.1 全オプション一覧

| オプション | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `--dataset` | str | None | データセット（--input-fileと排他） |
| `--input-file` | str | None | ローカルファイルパス（--datasetと排他） |
| `--model` | str | gpt-4o-mini | 使用するOpenAIモデル |
| `--output` | str | qa_output/a02 | 出力ディレクトリ |
| `--max-docs` | int | None | 処理する最大文書数 |
| `--analyze-coverage` | flag | False | カバレージ分析を実行 |
| `--batch-chunks` | int | 3 | バッチサイズ（1-5） |
| `--merge-chunks` | flag | True | チャンク統合を有効化 |
| `--no-merge-chunks` | flag | - | チャンク統合を無効化 |
| `--min-tokens` | int | 150 | 統合対象の最小トークン数 |
| `--max-tokens` | int | 400 | 統合後の最大トークン数 |
| `--use-celery` | flag | False | Celery並列処理を使用 |
| `--celery-workers` | int | 4 | Celeryワーカー数 |
| `--coverage-threshold` | float | None | カスタム閾値 |

### 11.2 入力ソース

**--dataset**: OUTPUTフォルダのpreprocessedファイルを使用
```bash
python a02_make_qa_para.py --dataset livedoor
```

**--input-file**: 任意のローカルファイルを使用
```bash
python a02_make_qa_para.py --input-file qa_output/qa_pairs_upload_20251122_182355.csv
```

---

## 12. 実行方法

### 12.1 環境準備

```bash
# 1. Redisを起動
brew services start redis  # macOS
# または: redis-server

# 2. 既存タスクのクリア（推奨）
redis-cli FLUSHDB

# 3. Celeryワーカーを起動
./start_celery.sh restart -w 24
```

### 12.2 テスト実行

```bash
# 同期処理（小規模テスト）
python a02_make_qa_para.py \
  --dataset livedoor \
  --batch-chunks 3 \
  --model gpt-4o-mini \
  --max-docs 20 \
  --analyze-coverage
```

### 12.3 Celery並列実行

```bash
python a02_make_qa_para.py \
  --dataset cc_news \
  --use-celery \
  --celery-workers 24 \
  --batch-chunks 3 \
  --merge-chunks \
  --min-tokens 150 \
  --max-tokens 400 \
  --model gpt-4o-mini \
  --analyze-coverage
```

### 12.4 UIからの実行（Streamlit）

**詳細は `doc/05_qa_pair.md` を参照**

```bash
# Streamlit UIを起動
streamlit run rag_qa_pair_qdrant.py
```

UIでは以下の操作が可能：
- データセット/ファイルの選択
- モデル・パラメータの設定
- Celery並列処理のON/OFF
- リアルタイム進捗表示
- 結果のダウンロード

### 12.5 実行時の進捗表示

```
進捗: 成功=3/17, 失敗=0, 実行中=4, 待機中=10, 経過時間=15.2秒
進捗: 成功=7/17, 失敗=0, 実行中=4, 待機中=6, 経過時間=20.4秒
進捗: 成功=17/17, 失敗=0, 実行中=0, 待機中=0, 経過時間=45.8秒
✓ すべてのタスクが完了しました
```

### 12.6 実行時間の見積もり

| 項目 | 値 |
|-----|-----|
| 処理文書数 | 497件（全件） |
| チャンク数 | ~1,825個 → 統合後 ~1,820個 |
| API呼び出し | 約365回（バッチサイズ5） |
| 推定実行時間 | 60-75分 |
| カバレージ分析 | +3-5分 |
| 合計 | 約65-80分 |

---

## 13. トラブルシューティング

### 13.1 Celeryワーカーが起動しない

**診断手順**:
```bash
# Redisの状態確認
redis-cli ping

# ログを確認
tail -f logs/celery_qa_*.log

# プロセスを確認
ps aux | grep celery
```

**解決策**:
```bash
# 既存ワーカーを停止
./start_celery.sh stop

# タスクキューをクリア
redis-cli FLUSHDB

# 再起動
./start_celery.sh start -w 8
```

### 13.2 タスクが処理されない

**診断**:
```bash
# キューの状態確認
redis-cli LLEN celery
redis-cli LLEN qa_generation

# ワーカーの状態確認
celery -A celery_tasks inspect active
```

**解決策**:
```bash
# ワーカー数を減らして再起動
./start_celery.sh restart -w 2

# それでも解決しない場合は同期処理に切り替え
python a02_make_qa_para.py --dataset cc_news --batch-chunks 3 --max-docs 20
```

### 13.3 JSONDecodeError

**症状**: `Expecting value: line 1 column 1 (char 0)`

**原因**: OpenAI APIが空のレスポンスを返している

**解決策**:
1. ワーカーを再起動
```bash
./start_celery.sh restart -w 8
```

2. ログを確認
```bash
tail -f logs/celery_qa_*.log
```

### 13.4 空レスポンスエラー

**症状**: `No parseable response from OpenAI API`

**対策（コード内で自動処理）**:
```python
# 空レスポンスチェック
if parsed_count == 0:
    logger.error("OpenAI APIから解析可能なレスポンスが返されませんでした")
    raise ValueError("No parseable response from OpenAI API")
```

### 13.5 タイムアウトエラー

**症状**: `task_time_limit exceeded`

**解決策**: `celery_tasks.py`でタイムアウトを延長
```python
app.conf.update(
    task_time_limit=600,  # 10分に延長
    task_soft_time_limit=540,
)
```

---

## 14. 次ステップ

### 14.1 Qdrantへの登録

Q/Aペア生成後、以下のフローでQdrantに登録できます：

```
[Q/Aペア]
    │
    │ a02_qa_pairs_{dataset}.csv
    ▼
[Embedding生成]  ←── text-embedding-3-small (1536次元)
    │
    ▼
[Qdrant登録]  ←── upsert_points_to_qdrant()
    │
    ▼
[検索可能状態]
```

**詳細**: `doc/06_embedding_qdrant.md` 参照

**登録コマンド**:
```bash
# CLI経由
python a30_qdrant_registration.py --recreate --limit 100

# または Streamlit UI
streamlit run rag_qa_pair_qdrant.py
# → 「Qdrant登録」ページで操作
```

### 14.2 検索処理との連携

登録後は以下の検索処理が可能：

```bash
# CLI検索
python a50_rag_search_local_qdrant.py

# Streamlit UI検索
streamlit run rag_qa_pair_qdrant.py
# → 「Qdrant検索」ページで操作
```

**RAGフロー**:
```
[ユーザー質問]
    ▼
[クエリベクトル化]  ←── embed_query_for_search()
    ▼
[Qdrant検索]  ←── client.search()
    ▼
[Top-K結果取得]  (question, answer, score)
    ▼
[AI応答生成]  ←── OpenAI GPT-4o-mini
    ▼
[最終回答]
```

---

## 15. 付録

### 15.1 データ読み込み関数

#### A.1 ローカルファイル読み込み

```python
# a02_make_qa_para.py:545-640
def load_uploaded_file(file_path: str) -> pd.DataFrame:
    """
    ローカルファイルを読み込み

    対応形式: CSV, TXT, JSON, JSONL
    Combined_Textカラムを自動生成
    """
```

#### A.2 Q/A CSVファイル読み込み

```python
# a02_make_qa_para.py:643-701
def load_local_qa_file(file_path: str) -> pd.DataFrame:
    """ローカルのQ/A CSVファイルを読み込み

    question, answerカラムを自動検出
    空データ・重複を除去
    """
```

#### A.3 preprocessedデータ読み込み

```python
# a02_make_qa_para.py:704-753
def load_preprocessed_data(dataset_type: str) -> pd.DataFrame:
    """preprocessedデータを読み込み

    - 固定名ファイルを優先
    - タイムスタンプ付きファイルを自動検索
    - 最新ファイルを自動選択
    """
```

### 15.2 コード参照一覧

| 機能 | ファイル | 関数/クラス | 行番号 |
|-----|---------|------------|-------|
| キーワード抽出 | a02_make_qa_para.py | KeywordExtractor | 277-403 |
| 複雑度分析 | a02_make_qa_para.py | analyze_chunk_complexity() | 410-457 |
| セマンティックチャンク作成 | a02_make_qa_para.py | create_semantic_chunks() | 487-538 |
| 小チャンク統合 | a02_make_qa_para.py | merge_small_chunks() | 819-875 |
| Q/A数決定 | a02_make_qa_para.py | determine_qa_count() | 882-913 |
| バッチQ/A生成 | a02_make_qa_para.py | generate_qa_pairs_for_batch() | 916-1103 |
| 単一チャンクQ/A生成 | a02_make_qa_para.py | generate_qa_pairs_for_chunk() | 1106-1291 |
| データセット全体処理 | a02_make_qa_para.py | generate_qa_for_dataset() | 1294-1394 |
| Celeryワーカー確認 | a02_make_qa_para.py | check_celery_workers() | 1401-1459 |
| 最適閾値設定 | a02_make_qa_para.py | OPTIMAL_THRESHOLDS | 1467-1488 |
| 多段階カバレージ | a02_make_qa_para.py | multi_threshold_coverage() | 1505-1539 |
| チャンク特性分析 | a02_make_qa_para.py | analyze_chunk_characteristics_coverage() | 1542-1644 |
| カバレージ分析 | a02_make_qa_para.py | analyze_coverage() | 1647-1769 |
| 結果保存 | a02_make_qa_para.py | save_results() | 1776-1859 |
| メイン処理 | a02_make_qa_para.py | main() | 1866-2155 |
| Pydanticモデル | models.py | QAPair, QAPairsResponse | 24-75 |
| Celeryタスク投入 | celery_tasks.py | submit_parallel_qa_generation() | - |
| Celery結果収集 | celery_tasks.py | collect_results() | 688-920 |

---

## 16. 参考資料

| ドキュメント | 内容 |
|-------------|------|
| `doc/03_chunk.md` | チャンク分割技術の詳細（SemanticCoverage、MeCab文分割） |
| `doc/04_prompt.md` | プロンプト設計の詳細（2段階構造、質問タイプ階層、API呼び出し） |
| `doc/05_qa_pair.md` | Q/Aペア生成処理フローの詳細（Celery並列処理、カバレージ） |
| `doc/06_embedding_qdrant.md` | Embedding・Qdrant登録の詳細（ベクトル化、検索処理） |
| `doc/helper_api.md` | OpenAI API関連のドキュメント |
| `CLAUDE.md` | プロジェクト全体のガイドライン |
# RAGシステム プロンプト設計・処理方式ドキュメント

本ドキュメントでは、RAG Q/A生成システムにおけるプロンプト設計と処理方式について解説する。

## 目次

- [1. 概要](#1-概要)
  - [1.1 本ドキュメントの目的](#11-本ドキュメントの目的)
  - [1.2 関連ファイル一覧](#12-関連ファイル一覧)
  - [1.3 プロンプト処理フロー図](#13-プロンプト処理フロー図)
- [2. プロンプト設計の特徴](#2-プロンプト設計の特徴)
  - [2.1 2段階プロンプト構造](#21-2段階プロンプト構造)
  - [2.2 言語別対応](#22-言語別対応)
  - [2.3 動的プロンプト調整](#23-動的プロンプト調整)
  - [2.4 型安全な出力](#24-型安全な出力)
- [3. システムプロンプト設計](#3-システムプロンプト設計)
  - [3.1 日本語版システムプロンプト](#31-日本語版システムプロンプト)
  - [3.2 英語版システムプロンプト](#32-英語版システムプロンプト)
  - [3.3 設計原則](#33-設計原則)
- [4. 質問タイプ階層構造](#4-質問タイプ階層構造)
  - [4.1 3階層構造の定義](#41-3階層構造の定義)
  - [4.2 質問タイプ一覧と用途](#42-質問タイプ一覧と用途)
  - [4.3 プロンプトでの活用方法](#43-プロンプトでの活用方法)
- [5. ユーザープロンプト構築](#5-ユーザープロンプト構築)
  - [5.1 単一チャンク処理プロンプト](#51-単一チャンク処理プロンプト)
  - [5.2 バッチ処理プロンプト](#52-バッチ処理プロンプト)
  - [5.3 JSON出力フォーマット仕様](#53-json出力フォーマット仕様)
- [6. チャンク前処理とプロンプト最適化](#6-チャンク前処理とプロンプト最適化)
  - [6.1 セマンティックチャンク分割](#61-セマンティックチャンク分割)
  - [6.2 チャンク複雑度分析](#62-チャンク複雑度分析)
  - [6.3 キーワード抽出](#63-キーワード抽出)
  - [6.4 Q/A数の動的決定ロジック](#64-qa数の動的決定ロジック)
- [7. API呼び出し方式](#7-api呼び出し方式)
  - [7.1 構造化出力API（client.responses.parse）](#71-構造化出力apiclientresponsesparse)
  - [7.2 Chat Completions API（フォールバック）](#72-chat-completions-apiフォールバック)
  - [7.3 モデル別パラメータ制約](#73-モデル別パラメータ制約)
- [8. Pydanticモデル定義](#8-pydanticモデル定義)
  - [8.1 QAPair / QAPairsResponse](#81-qapair--qapairsresponse)
  - [8.2 ChunkComplexity](#82-chunkcomplexity)
  - [8.3 構造化出力との連携](#83-構造化出力との連携)
- [9. データセット別設定](#9-データセット別設定)
  - [9.1 言語・チャンクサイズ設定](#91-言語チャンクサイズ設定)
  - [9.2 カバレージ閾値設定](#92-カバレージ閾値設定)
  - [9.3 Q/Aペア数の基本設定](#93-qaペア数の基本設定)
- [10. 実装パターンと拡張方法](#10-実装パターンと拡張方法)
  - [10.1 新規質問タイプの追加手順](#101-新規質問タイプの追加手順)
  - [10.2 プロンプトテンプレートのカスタマイズ](#102-プロンプトテンプレートのカスタマイズ)
  - [10.3 デバッグ・トラブルシューティング](#103-デバッグトラブルシューティング)
- [参考資料](#参考資料)

---

## 1. 概要

### 1.1 本ドキュメントの目的

本ドキュメントは以下を目的とする：

- RAGシステムで使用されるプロンプトの設計思想と構造の理解
- Q/A生成における言語別・データセット別の処理方式の把握
- プロンプトのカスタマイズや拡張を行う際の参考資料

### 1.2 関連ファイル一覧

| ファイル | 役割 | 主要な処理 |
|---------|------|-----------|
| `a02_make_qa_para.py` | 主要なプロンプト処理・Q/A生成 | プロンプト構築、API呼び出し、バッチ処理 |
| `celery_tasks.py` | Celery非同期処理 | 並列Q/A生成、リトライ処理 |
| `config.py` | 設定管理 | 質問タイプ階層、データセット設定 |
| `models.py` | Pydanticモデル | QAPair、QAPairsResponse定義 |
| `services/qa_service.py` | サービスラッパー | サブプロセス実行、結果収集 |
| `helper_rag_qa.py` | RAG補助機能 | セマンティックチャンク分割 |

### 1.3 プロンプト処理フロー図

```
[入力データ]
    │
    ▼
[セマンティックチャンク分割]  ←── helper_rag_qa.py: SemanticCoverage.create_semantic_chunks()
    │
    ▼
[チャンク統合（小チャンクのマージ）]  ←── a02_make_qa_para.py: merge_small_chunks()
    │
    ▼
[複雑度分析・キーワード抽出]  ←── 動的調整の根拠
    │   ├── analyze_chunk_complexity()
    │   └── KeywordExtractor.extract()
    │
    ▼
[プロンプト構築]
    │   ├── システムプロンプト（役割・ルール）
    │   └── ユーザープロンプト（チャンク + 質問タイプ + JSON形式）
    │
    ▼
[API呼び出し]
    │   ├── client.responses.parse()  ←── 主要API（構造化出力）
    │   └── client.chat.completions.create()  ←── フォールバック
    │
    ▼
[レスポンス解析]  ←── Pydanticモデル（QAPairsResponse）
    │
    ▼
[Q/Aペア出力]
```

---

## 2. プロンプト設計の特徴

本システムのプロンプト設計には4つの主要な特徴がある。

### 2.1 2段階プロンプト構造

プロンプトは**システムプロンプト**と**ユーザープロンプト**の2段階で構成される。

#### システムプロンプトの役割

- LLMの**役割定義**（教育コンテンツ作成の専門家）
- **生成ルール**の明示（明確性、簡潔性、忠実性、多様性）

#### ユーザープロンプトの構成

1. **チャンクテキスト**: 処理対象のテキスト
2. **質問タイプ指示**: fact/reason/comparison/application等
3. **JSON出力形式**: 構造化された出力フォーマット指定

```
[システムプロンプト]
    ↓ 役割・ルール設定
[ユーザープロンプト]
    ├── チャンクテキスト
    ├── 質問タイプ説明
    └── JSON出力形式指定
```

**実装箇所**: `a02_make_qa_para.py:946-990`（バッチ処理）、`a02_make_qa_para.py:1128-1178`（単一チャンク処理）

### 2.2 言語別対応

日本語と英語で完全に分離されたプロンプトテンプレートを持つ。

#### 切替メカニズム

```python
lang = config["lang"]  # "ja" or "en"

if lang == "ja":
    system_prompt = """あなたは教育コンテンツ作成の専門家です。..."""
else:
    system_prompt = """You are an expert in educational content creation..."""
```

#### 言語判定の流れ

1. データセット設定（`config.py`）で `lang` 属性を定義
2. `DATASET_CONFIGS[dataset_type]["lang"]` で取得
3. プロンプト構築時に言語別テンプレートを選択

**実装箇所**: `a02_make_qa_para.py:945`、`config.py:151-213`

### 2.3 動的プロンプト調整

チャンクの特性に基づいてQ/A生成数を動的に調整する。

#### 調整要因

| 要因 | 判定基準 | Q/A数への影響 |
|------|---------|--------------|
| トークン数 | < 50 tokens | 基本数 - 1（最低2） |
| トークン数 | 50-100 tokens | 基本数（3） |
| トークン数 | 100-200 tokens | 基本数 + 1 |
| トークン数 | 200-300 tokens | 基本数 + 2 |
| トークン数 | > 300 tokens | 基本数 + 3 |
| 文書位置 | 6番目以降のチャンク | +1（後半補正） |

#### 実装コード

```python
def determine_qa_count(chunk: Dict, config: Dict) -> int:
    base_count = config["qa_per_chunk"]
    token_count = len(tokenizer.encode(chunk['text']))
    chunk_position = chunk.get('chunk_idx', 0)

    # トークン数に基づく調整
    if token_count < 50:
        qa_count = 2
    elif token_count < 100:
        qa_count = 3
    # ... 省略 ...

    # 文書後半の位置バイアス補正
    if isinstance(chunk_position, int) and chunk_position >= 5:
        qa_count += 1

    return min(qa_count, 8)  # 上限8
```

**実装箇所**: `a02_make_qa_para.py:882-913`

### 2.4 型安全な出力

Pydanticモデルを使用した構造化出力により、型安全性を確保する。

#### 主要API（推奨）

```python
response = client.responses.parse(
    input=combined_input,
    model=model,
    text_format=QAPairsResponse,  # Pydanticモデル
    max_output_tokens=4000
)
```

#### フォールバックAPI

```python
response = client.chat.completions.create(
    model=model,
    messages=[...],
    response_format={"type": "json_object"}
)
```

#### API選択の判断フロー

```
[responses.parse() 試行]
    │
    ├── 成功 → Pydanticモデルで自動パース
    │
    └── 失敗 → [chat.completions.create() にフォールバック]
                    │
                    └── JSON手動パース
```

**実装箇所**: `a02_make_qa_para.py:1044-1049`、`celery_tasks.py:299-304`

#### 【実装コード】OpenAI API クライアント（helper_api.py:693-735）

```python
# helper_api.py - OpenAIClient クラス
class OpenAIClient:
    """OpenAI API クライアント"""

    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = config.get("api.openai_api_key") or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(config.get("error_messages.api_key_missing", "APIキーが設定されていません"))

        self.client = OpenAI(api_key=api_key)

    @error_handler
    @timer
    def create_response(
        self,
        messages: List[EasyInputMessageParam] = None,
        *,
        input: List[EasyInputMessageParam] = None,
        model: str = None,
        **kwargs,
    ) -> Response:
        """Responses API呼び出し

        `messages` 引数（旧仕様）と `input` 引数（新仕様）の両方に対応する。
        """
        if model is None:
            model = config.get("models.default", "gpt-4o-mini")

        # 新旧両方の引数名をサポート
        if input is None:
            input = messages
        if input is None:
            raise ValueError("messages or input must be provided")

        params = {
            "model": model,
            "input": input,
        }
        params.update(kwargs)

        return self.client.responses.create(**params)
```

#### 【実装コード】Q/Aペア生成（services/qa_service.py:234-298）

```python
# services/qa_service.py - generate_qa_pairs 関数
def generate_qa_pairs(
    text: str,
    dataset_type: str,
    chunk_id: str,
    model: str = "gpt-4o-mini",
    qa_per_chunk: int = 3,
    log_callback=None,
) -> List[QAPair]:
    """テキストからQ/Aペアを生成"""
    client = OpenAI()

    prompt = f"""以下のテキストから、{qa_per_chunk}個の質問と回答のペアを生成してください。

テキスト:
{text}

要件:
1. 質問は具体的で明確なものにする
2. 回答はテキストの内容に基づいた正確なものにする
3. 質問タイプは以下から選択: factual, conceptual, application, analysis
4. テキストの重要な情報を網羅するようにする

JSON形式で出力してください。
"""

    # モデルに応じてtemperatureを調整
    # GPT-5シリーズ、O-seriesはtemperatureパラメータをサポートしない
    model_lower = model.lower()
    no_temp_models = ["gpt-5", "o1", "o3", "o4"]
    use_temperature = not any(no_temp in model_lower for no_temp in no_temp_models)

    # API呼び出しパラメータを構築
    api_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": "あなたは教育用Q/Aペア生成の専門家です。"},
            {"role": "user", "content": prompt},
        ],
        "response_format": QAPairsResponse,  # Pydanticモデルで型安全な出力
    }

    # temperatureをサポートするモデルの場合のみ追加
    if use_temperature:
        api_params["temperature"] = 0.7

    response = client.beta.chat.completions.parse(**api_params)
    qa_response = response.choices[0].message.parsed

    # Q/Aペアにメタデータを追加
    for qa in qa_response.qa_pairs:
        qa.source_chunk_id = chunk_id
        qa.dataset_type = dataset_type
        qa.auto_generated = True

    return qa_response.qa_pairs
```

---

## 3. システムプロンプト設計

### 3.1 日本語版システムプロンプト

```
あなたは教育コンテンツ作成の専門家です。
与えられた日本語テキストから、学習効果の高いQ&Aペアを生成してください。

生成ルール:
1. 質問は明確で具体的に
2. 回答は簡潔で正確に（1-2文程度）
3. テキストの内容に忠実に
4. 多様な観点から質問を作成
```

**使用箇所**:
- 単一チャンク処理: `a02_make_qa_para.py:1129-1136`
- バッチ処理: `a02_make_qa_para.py:946-953`
- Celeryタスク: `celery_tasks.py:220-227`

### 3.2 英語版システムプロンプト

```
You are an expert in educational content creation.
Generate high-quality Q&A pairs from the given English text.

Generation rules:
1. Questions should be clear and specific
2. Answers should be concise and accurate (1-2 sentences)
3. Stay faithful to the text content
4. Create questions from diverse perspectives
```

**使用箇所**:
- 単一チャンク処理: `a02_make_qa_para.py:1144-1151`
- バッチ処理: `a02_make_qa_para.py:993-1000`
- Celeryタスク: `celery_tasks.py:254-261`

### 3.3 設計原則

#### 役割定義

| 要素 | 日本語 | 英語 |
|------|--------|------|
| 専門性 | 教育コンテンツ作成の専門家 | expert in educational content creation |
| 目的 | 学習効果の高いQ&Aペア生成 | high-quality Q&A pairs |

#### 生成ルール（4原則）

| # | 原則 | 説明 |
|---|------|------|
| 1 | 明確性 | 質問は具体的で曖昧さがない |
| 2 | 簡潔性 | 回答は1-2文程度で端的に |
| 3 | 忠実性 | テキスト内容に基づく正確さ |
| 4 | 多様性 | 様々な観点からの質問作成 |

---

## 4. 質問タイプ階層構造

### 4.1 3階層構造の定義

質問タイプは認知レベルに基づく3階層構造で定義される（`config.py:316-334`）。

```python
QUESTION_TYPES_HIERARCHY = {
    "basic": {          # 基礎レベル
        "definition": "定義型（〜とは何ですか？）",
        "identification": "識別型（〜の例を挙げてください）",
        "enumeration": "列挙型（〜の種類/要素は？）"
    },
    "understanding": {  # 理解レベル
        "cause_effect": "因果関係型（〜の結果/影響は？）",
        "process": "プロセス型（〜はどのように行われますか？）",
        "mechanism": "メカニズム型（〜の仕組みは？）",
        "comparison": "比較型（〜と〜の違いは？）"
    },
    "application": {    # 応用レベル
        "synthesis": "統合型（〜を組み合わせるとどうなりますか？）",
        "evaluation": "評価型（〜の長所と短所は？）",
        "prediction": "予測型（〜の場合どうなりますか？）",
        "practical": "実践型（〜はどのように活用されますか？）"
    }
}
```

### 4.2 質問タイプ一覧と用途

#### 基礎レベル（basic）

| タイプ | 説明 | 質問例 |
|--------|------|--------|
| definition | 定義型 | 「〜とは何ですか？」 |
| identification | 識別型 | 「〜の例を挙げてください」 |
| enumeration | 列挙型 | 「〜の種類は何がありますか？」 |

#### 理解レベル（understanding）

| タイプ | 説明 | 質問例 |
|--------|------|--------|
| cause_effect | 因果関係型 | 「〜の結果どうなりますか？」 |
| process | プロセス型 | 「〜はどのように行われますか？」 |
| mechanism | メカニズム型 | 「〜の仕組みはどうなっていますか？」 |
| comparison | 比較型 | 「〜と〜の違いは何ですか？」 |

#### 応用レベル（application）

| タイプ | 説明 | 質問例 |
|--------|------|--------|
| synthesis | 統合型 | 「〜を組み合わせるとどうなりますか？」 |
| evaluation | 評価型 | 「〜の長所と短所は何ですか？」 |
| prediction | 予測型 | 「〜の場合どうなりますか？」 |
| practical | 実践型 | 「〜はどのように活用されますか？」 |

### 4.3 プロンプトでの活用方法

実際のプロンプトでは、簡略化した4タイプを使用する。

#### 日本語プロンプト内の記述

```
質問タイプ:
- fact: 事実確認型（〜は何ですか？）
- reason: 理由説明型（なぜ〜ですか？）
- comparison: 比較型（〜と〜の違いは？）
- application: 応用型（〜はどのように活用されますか？）
```

#### 英語プロンプト内の記述

```
Question types:
- fact: Factual questions (What is...?)
- reason: Explanatory questions (Why...?)
- comparison: Comparative questions (What's the difference...?)
- application: Application questions (How is... used?)
```

#### 階層構造とプロンプトタイプの対応

| 階層 | 階層内タイプ | プロンプトタイプ |
|------|------------|----------------|
| basic | definition, identification, enumeration | fact |
| understanding | cause_effect, process, mechanism | reason |
| understanding | comparison | comparison |
| application | synthesis, evaluation, prediction, practical | application |

---

## 5. ユーザープロンプト構築

### 5.1 単一チャンク処理プロンプト

単一のチャンクからQ/Aペアを生成する場合のプロンプト構造。

#### 日本語版

```
以下のテキストから{num_pairs}個のQ&Aペアを生成してください。

質問タイプ:
- fact: 事実確認型（〜は何ですか？）
- reason: 理由説明型（なぜ〜ですか？）
- comparison: 比較型（〜と〜の違いは？）
- application: 応用型（〜はどのように活用されますか？）

テキスト:
{chunk_text}

JSON形式で出力:
{
  "qa_pairs": [
    {
      "question": "質問文",
      "answer": "回答文",
      "question_type": "fact/reason/comparison/application"
    }
  ]
}
```

#### 英語版

```
Generate {num_pairs} Q&A pairs from the following text.

Question types:
- fact: Factual questions (What is...?)
- reason: Explanatory questions (Why...?)
- comparison: Comparative questions (What's the difference...?)
- application: Application questions (How is... used?)

Text:
{chunk_text}

Output in JSON format:
{
  "qa_pairs": [
    {
      "question": "question text",
      "answer": "answer text",
      "question_type": "fact/reason/comparison/application"
    }
  ]
}
```

**実装箇所**: `a02_make_qa_para.py:1161-1197`

### 5.2 バッチ処理プロンプト

複数チャンク（1-5個）を統合して一度にQ/Aペアを生成する場合のプロンプト構造。

#### 構造

```
以下の{N}個のテキストから、合計{total_pairs}個のQ&Aペアを生成してください。

【テキスト1】
{chunk_1_text}

【テキスト2】
{chunk_2_text}

【テキスト3】
{chunk_3_text}

質問タイプ:
- fact: 事実確認型（〜は何ですか？）
- reason: 理由説明型（なぜ〜ですか？）
- comparison: 比較型（〜と〜の違いは？）
- application: 応用型（〜はどのように活用されますか？）

JSON形式で出力:
{
  "qa_pairs": [
    {
      "question": "質問文",
      "answer": "回答文",
      "question_type": "fact/reason/comparison/application"
    }
  ]
}
```

#### バッチ処理の利点

| 観点 | 単一処理 | バッチ処理 |
|------|---------|-----------|
| API呼び出し回数 | チャンク数と同じ | チャンク数 / バッチサイズ |
| コスト | 高い | 最大1/5に削減 |
| 処理速度 | 遅い | 高速 |
| 文脈理解 | 単一チャンクのみ | 複数チャンク間の関連も考慮可能 |

**実装箇所**: `a02_make_qa_para.py:972-1037`、`celery_tasks.py:482-521`

### 5.3 JSON出力フォーマット仕様

#### 基本構造

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

#### フィールド仕様

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| qa_pairs | array | Yes | Q/Aペアの配列 |
| question | string | Yes | 質問文 |
| answer | string | Yes | 回答文（1-2文程度） |
| question_type | string | Yes | fact/reason/comparison/application |

#### 生成後に追加されるメタデータ

Q/Aペア生成後、以下のメタデータが自動付与される：

```python
qa = {
    "question": qa_data.question,
    "answer": qa_data.answer,
    "question_type": qa_data.question_type,
    "source_chunk_id": chunk.get('id', ''),      # 元チャンクID
    "doc_id": chunk.get('doc_id', ''),            # ドキュメントID
    "dataset_type": chunk.get('dataset_type', ''), # データセットタイプ
    "chunk_idx": chunk.get('chunk_idx', 0)        # チャンクインデックス
}
```

---

## 6. チャンク前処理とプロンプト最適化

### 6.1 セマンティックチャンク分割

文書を意味的なまとまりで分割する。段落境界を最優先とする。

#### 処理フロー

```
[入力テキスト]
    │
    ▼
[段落検出]  ←── 空行・改行パターンで分割
    │
    ▼
[トークン数チェック]
    │
    ├── max_tokens以下 → そのまま1チャンク
    │
    └── max_tokens超過 → [文単位で分割]
                              │
                              └── 最小トークン数を満たすまで結合
    │
    ▼
[チャンクリスト出力]
```

#### 実装コード

```python
def create_semantic_chunks(text: str, lang: str = "ja",
                          max_tokens: int = 200,
                          chunk_id_prefix: str = "chunk") -> List[Dict]:
    semantic_analyzer = SemanticCoverage(embedding_model="text-embedding-3-small")

    semantic_chunks = semantic_analyzer.create_semantic_chunks(
        document=text,
        max_tokens=max_tokens,
        min_tokens=50,
        prefer_paragraphs=True,  # 段落優先モード
        verbose=False
    )
    # ... チャンク形式変換 ...
```

**実装箇所**: `a02_make_qa_para.py:487-538`

#### 【実装コード】SemanticCoverageクラス（helper_rag_qa.py:1484-1584）

```python
# helper_rag_qa.py - SemanticCoverage クラス
class SemanticCoverage:
    """意味的な網羅性を測定するクラス"""

    def __init__(self, embedding_model="text-embedding-3-small"):
        self.embedding_model = embedding_model
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and api_key != 'your-openai-api-key-here':
            self.client = OpenAI()
            self.has_api_key = True
        else:
            self.client = None
            self.has_api_key = False
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.mecab_available = self._check_mecab_availability()

    def create_semantic_chunks(self, document: str, max_tokens: int = 200,
                               min_tokens: int = 50, prefer_paragraphs: bool = True,
                               verbose: bool = True) -> List[Dict]:
        """
        文書を意味的に区切られたチャンクに分割（段落優先のセマンティック分割）

        重要ポイント：
        1. 段落の境界で分割（最優先 - 筆者の意図したセマンティック境界）
        2. 文の境界で分割（意味の断絶を防ぐ）
        3. トピックの変化を検出
        4. 適切なサイズを維持（埋め込みモデルの制限内）

        Args:
            document: 分割対象の文書
            max_tokens: チャンクの最大トークン数（デフォルト: 200）
            min_tokens: チャンクの最小トークン数（デフォルト: 50）
            prefer_paragraphs: 段落ベースの分割を優先するか（デフォルト: True）
            verbose: 詳細な出力を行うか

        Returns:
            チャンク辞書のリスト（id, text, type, sentences等を含む）
        """
        # Step 1: 段落ベースの分割を試行（prefer_paragraphs=Trueの場合）
        if prefer_paragraphs:
            para_chunks = self._chunk_by_paragraphs(document, max_tokens, min_tokens)

            if verbose:
                print(f"段落ベースのチャンク数: {len(para_chunks)}")
                type_counts = {}
                for chunk in para_chunks:
                    chunk_type = chunk.get('type', 'unknown')
                    type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
                print(f"チャンクタイプ内訳: {type_counts}")

            # 段落ベースのチャンクを標準フォーマットに変換
            chunks = []
            for i, chunk in enumerate(para_chunks):
                chunk_text = chunk['text']
                sentences = self._split_into_sentences(chunk_text)
                chunks.append({
                    "id"                : f"chunk_{i}",
                    "text"              : chunk_text,
                    "type"              : chunk['type'],
                    "sentences"         : sentences,
                    "start_sentence_idx": 0,
                    "end_sentence_idx"  : len(sentences) - 1
                })
        else:
            # 文単位で分割（旧方式）
            sentences = self._split_into_sentences(document)
            chunks = self._group_sentences_into_chunks(sentences, max_tokens)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        テキストを文単位に分割（日本語対応）

        日本語の文末パターン：
        - 句点「。」
        - 感嘆符「！」
        - 疑問符「？」
        """
        # 日本語の文区切りパターン
        pattern = r'(?<=[。！？\.\!\?])\s*'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _chunk_by_paragraphs(self, document: str, max_tokens: int,
                             min_tokens: int) -> List[Dict]:
        """段落単位でチャンクを作成"""
        # 段落分割（空行で区切る）
        paragraphs = re.split(r'\n\s*\n', document)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = len(self.tokenizer.encode(para))

            if current_tokens + para_tokens <= max_tokens:
                current_chunk.append(para)
                current_tokens += para_tokens
            else:
                # 現在のチャンクを保存
                if current_chunk:
                    chunks.append({
                        'text': '\n\n'.join(current_chunk),
                        'type': 'paragraph_group'
                    })
                # 新しいチャンク開始
                if para_tokens > max_tokens:
                    # 大きすぎる段落は文単位で分割
                    sub_chunks = self._split_large_paragraph(para, max_tokens)
                    chunks.extend(sub_chunks)
                    current_chunk = []
                    current_tokens = 0
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens

        # 残りを処理
        if current_chunk:
            chunks.append({
                'text': '\n\n'.join(current_chunk),
                'type': 'paragraph_group'
            })

        return chunks
```

#### 【実装コード】テキストクレンジング（helper_rag.py:357-378）

```python
# helper_rag.py - clean_text 関数
def clean_text(text: str) -> str:
    """テキストのクレンジング処理

    処理内容：
    1. 改行を空白に変換
    2. 連続した空白を1つにまとめる
    3. 先頭・末尾の空白を除去
    4. 引用符の正規化（全角・半角統一）
    """
    if pd.isna(text) or text == "":
        return ""

    text = str(text)

    # 改行を空白に置換
    text = text.replace('\n', ' ').replace('\r', ' ')

    # 連続した空白を1つにまとめる
    text = re.sub(r'\s+', ' ', text)

    # 先頭・末尾の空白を除去
    text = text.strip()

    # 引用符の正規化
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    return text
```

### 6.2 チャンク複雑度分析

チャンクの複雑度を分析し、Q/A生成の難易度を推定する。

#### 分析指標

| 指標 | 計算方法 | 用途 |
|------|---------|------|
| avg_sentence_length | トークン数 / 文数 | 文の複雑度 |
| concept_density | 専門用語数 / トークン数 × 100 | 概念密度 |
| technical_terms | 正規表現パターンマッチ | 専門用語リスト |

#### 複雑度レベル判定

```python
if concept_density > 5 or avg_sentence_length > 30:
    complexity_level = "high"
elif concept_density > 2 or avg_sentence_length > 20:
    complexity_level = "medium"
else:
    complexity_level = "low"
```

#### 専門用語検出パターン

| 言語 | パターン | 例 |
|------|---------|-----|
| 日本語 | `[ァ-ヴー]{4,}` または `[一-龥]{4,}` | カタカナ語、漢字複合語 |
| 英語 | `[A-Z][a-z]+(?:[A-Z][a-z]+)+` または `\b\w{10,}\b` | CamelCase、長い単語 |

**実装箇所**: `a02_make_qa_para.py:410-457`

### 6.3 キーワード抽出

MeCabと正規表現の2方式でキーワードを抽出する。

#### KeywordExtractorクラス

```python
class KeywordExtractor:
    def __init__(self, prefer_mecab: bool = True):
        self.mecab_available = self._check_mecab_availability()
        self.stopwords = {'こと', 'もの', 'これ', ...}

    def extract(self, text: str, top_n: int = 5) -> List[str]:
        if self.mecab_available and self.prefer_mecab:
            return self._extract_with_mecab(text, top_n)
        return self._extract_with_regex(text, top_n)
```

#### 抽出方式の比較

| 方式 | 利点 | 欠点 |
|------|------|------|
| MeCab | 複合名詞の正確な抽出 | 依存ライブラリ必要 |
| 正規表現 | 依存なし、高速 | 精度がやや低い |

#### フォールバック機構

```
[MeCab利用可能チェック]
    │
    ├── 利用可能 → MeCab抽出
    │                │
    │                └── 失敗時 → 正規表現抽出
    │
    └── 利用不可 → 正規表現抽出
```

**実装箇所**: `a02_make_qa_para.py:277-403`

### 6.4 Q/A数の動的決定ロジック

チャンクの特性に基づいてQ/A生成数を決定する。

#### 決定アルゴリズム

```python
def determine_qa_count(chunk: Dict, config: Dict) -> int:
    base_count = config["qa_per_chunk"]  # データセット設定の基本値
    token_count = len(tokenizer.encode(chunk['text']))
    chunk_position = chunk.get('chunk_idx', 0)

    # Step 1: トークン数による基本Q/A数決定
    if token_count < 50:
        qa_count = 2
    elif token_count < 100:
        qa_count = 3
    elif token_count < 200:
        qa_count = base_count + 1
    elif token_count < 300:
        qa_count = base_count + 2
    else:
        qa_count = base_count + 3

    # Step 2: 文書後半の位置バイアス補正
    if isinstance(chunk_position, int) and chunk_position >= 5:
        qa_count += 1

    # Step 3: 上限適用
    return min(qa_count, 8)
```

#### 設計意図

| 調整 | 理由 |
|------|------|
| 短いチャンクでも最低2個 | カバレージ確保 |
| 長いチャンクは+3まで | 情報量に比例 |
| 後半チャンクは+1 | 文書後半のカバレージ低下を補正 |
| 上限8個 | 品質維持（多すぎると質が低下） |

**実装箇所**: `a02_make_qa_para.py:882-913`

#### 【実装コード】Q/A数最適化クラス（helper_rag_qa.py:479-665）

```python
# helper_rag_qa.py - QACountOptimizer クラス
class QACountOptimizer:
    """Q/Aペア数の最適化を行うクラス"""

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def calculate_optimal_qa_count(self, document: str, mode: str = "auto") -> Dict[str, Any]:
        """
        文書特性から最適なQ/A数を算出

        Args:
            document: 対象文書
            mode: 決定モード
                - "auto": 自動決定（文書長ベース）
                - "evaluation": 評価用（網羅性重視）
                - "learning": 学習用（主要概念重視）
                - "search_test": 検索テスト用（多様性重視）
                - "faq": FAQ生成用（実用性重視）

        Returns:
            最適なQ/A数と決定根拠を含む辞書
        """
        # 基本メトリクスの計算
        metrics = self._analyze_document_metrics(document)

        # モード別の決定
        if mode == "evaluation":
            # 評価用：網羅性重視（文書長の5-10%）
            base_count = int(metrics['sentence_count'] * 0.075)
        elif mode == "learning":
            # 学習用：主要概念をカバー（10-20個）
            base_count = min(20, max(10, metrics['keyword_count'] // 2))
        elif mode == "search_test":
            # 検索テスト：多様な質問パターン（20-30個）
            base_count = min(30, max(20, metrics['sentence_count'] // 3))
        elif mode == "faq":
            # FAQ生成：ユーザーニーズベース（5-15個）
            base_count = min(15, max(5, metrics['keyword_count'] // 3))
        else:  # auto
            # 文書長ベースの自動決定
            base_count = self._calculate_by_document_length(metrics)

        # 情報密度による調整
        adjusted_count = self._adjust_by_information_density(document, base_count, metrics)

        # カバレッジ目標による調整
        final_count = self._adjust_by_coverage_target(document, adjusted_count, metrics)

        return {
            'optimal_count': final_count,
            'base_count': base_count,
            'adjusted_count': adjusted_count,
            'metrics': metrics,
            'mode': mode,
            'reasoning': self._generate_reasoning(metrics, base_count, adjusted_count, final_count, mode)
        }

    def _analyze_document_metrics(self, document: str) -> Dict:
        """文書の基本メトリクスを分析"""
        sentences = re.split(r'[。！？\.\!\?]+', document)
        sentences = [s.strip() for s in sentences if s.strip()]

        # トークン数の計算
        token_count = len(self.tokenizer.encode(document))

        # キーワード候補の抽出
        technical_terms = re.findall(r'[ァ-ヴー]{3,}|[A-Z]{2,}[A-Z0-9]*|[一-龥]{4,}', document)

        # 段落数の計算
        paragraphs = document.split('\n\n')
        paragraphs = [p for p in paragraphs if p.strip()]

        return {
            'doc_length': len(document),
            'token_count': token_count,
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_sentence_length': len(document) / max(1, len(sentences)),
            'keyword_count': len(set(technical_terms)),
            'keyword_density': len(technical_terms) / max(1, len(document) / 100),
            'complexity_score': self._calculate_complexity_score(document, sentences, technical_terms)
        }

    def _calculate_by_document_length(self, metrics: Dict) -> int:
        """文書長に基づく基本Q/A数の計算"""
        doc_length = metrics['doc_length']

        if doc_length < 500:
            return 3
        elif doc_length < 1000:
            return 5
        elif doc_length < 2000:
            return 8
        elif doc_length < 5000:
            return 12
        elif doc_length < 10000:
            return 18
        else:
            # 対数的増加
            extra = int(math.log(doc_length / 10000, 2) * 3)
            return min(30, 18 + extra)

    def _adjust_by_information_density(self, text: str, base_count: int, metrics: Dict) -> int:
        """情報密度に応じて調整"""
        keyword_density = metrics['keyword_density']
        complexity = metrics['complexity_score']

        # 情報密度による調整係数
        if keyword_density > 4 and complexity > 0.7:
            # 高密度・高複雑度
            multiplier = 1.5
        elif keyword_density > 2.5 or complexity > 0.5:
            # 中密度または中複雑度
            multiplier = 1.2
        elif keyword_density < 1 and complexity < 0.3:
            # 低密度・低複雑度
            multiplier = 0.7
        else:
            multiplier = 1.0

        return max(3, int(base_count * multiplier))

    def _adjust_by_coverage_target(self, text: str, current_count: int,
                                   metrics: Dict, target_coverage: float = 0.7) -> int:
        """カバレッジ目標による調整"""
        # セマンティックチャンク数の推定
        estimated_chunks = metrics['token_count'] // 150  # 150トークン/チャンク

        # 1つのQ/Aが平均2-3チャンクをカバーすると仮定
        coverage_per_qa = 2.5
        required_qa = int(estimated_chunks * target_coverage / coverage_per_qa)

        # 現在のカウントと必要数の中間を取る
        final_count = int((current_count + required_qa) / 2)

        # 範囲制限（3-50）
        return max(3, min(50, final_count))
```

---

## 7. API呼び出し方式

### 7.1 構造化出力API（client.responses.parse）

Pydanticモデルを使用した型安全な出力を得るための主要API。

#### 基本構文

```python
from models import QAPairsResponse

response = client.responses.parse(
    input=combined_input,          # システムプロンプト + ユーザープロンプト
    model=model,                    # "gpt-4o-mini", "gpt-5-mini" 等
    text_format=QAPairsResponse,   # Pydanticモデル
    max_output_tokens=4000          # バッチ処理時は4000、単一は1000
)
```

#### レスポンス解析

```python
# GPT-5形式（output_parsed属性）
if hasattr(response, 'output_parsed') and response.output_parsed:
    parsed_response = response.output_parsed

# GPT-4o形式（output配列 -> parsed）
for output in response.output:
    if output.type == "message":
        for item in output.content:
            if item.type == "output_text" and item.parsed:
                parsed_response = item.parsed
```

#### 利点

- 型安全性：Pydanticモデルによる自動バリデーション
- コード簡潔化：手動JSONパース不要
- エラー検出：スキーマ違反を自動検出

**実装箇所**: `a02_make_qa_para.py:1044-1049`、`celery_tasks.py:299-304`

### 7.2 Chat Completions API（フォールバック）

構造化出力APIが失敗した場合のフォールバック。

#### 基本構文

```python
completion_params = {
    "model": model,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    "response_format": {"type": "json_object"},
}

# temperatureサポートモデルのみ
if supports_temperature(model):
    completion_params["temperature"] = 0.7

# GPT-5/O-seriesはmax_completion_tokens使用
if model.startswith("gpt-5") or model.startswith("o3") or model.startswith("o4"):
    completion_params["max_completion_tokens"] = 1000
else:
    completion_params["max_tokens"] = 1000

response = client.chat.completions.create(**completion_params)
```

#### レスポンス解析

```python
import json

response_text = response.choices[0].message.content
parsed_data = json.loads(response_text)

for qa_data in parsed_data.get('qa_pairs', []):
    qa = {
        "question": qa_data.get('question', ''),
        "answer": qa_data.get('answer', ''),
        "question_type": qa_data.get('question_type', 'fact'),
        # ... メタデータ追加 ...
    }
```

**実装箇所**: `celery_tasks.py:329-381`

### 7.3 モデル別パラメータ制約

モデルによってサポートされるパラメータが異なる。

#### temperatureパラメータ

```python
NO_TEMPERATURE_MODELS = [
    "gpt-5", "gpt-5-mini", "gpt-5-nano",  # GPT-5シリーズ
    "o1", "o1-mini",                       # O1シリーズ
    "o3", "o3-mini",                       # O3シリーズ
    "o4", "o4-mini",                       # O4シリーズ
]

def supports_temperature(model: str) -> bool:
    return model not in NO_TEMPERATURE_MODELS
```

#### トークン制限パラメータ

| モデル | パラメータ名 | 備考 |
|--------|------------|------|
| GPT-4o系 | max_tokens | 従来のパラメータ |
| GPT-5系 | max_completion_tokens | 新パラメータ |
| O-series | max_completion_tokens | 新パラメータ |

#### 判定ロジック

```python
if model.startswith("gpt-5") or model.startswith("o3") or model.startswith("o4"):
    completion_params["max_completion_tokens"] = 1000
else:
    completion_params["max_tokens"] = 1000
```

**実装箇所**: `config.py:52-57`、`celery_tasks.py:347-351`

#### 【実装コード】モデル設定クラス（helper_rag.py:37-97）

```python
# helper_rag.py - AppConfig クラス
class AppConfig:
    """アプリケーション設定（全アプリ共通）"""

    # 利用可能なモデル（全OpenAIモデル対応）
    AVAILABLE_MODELS = [
        "gpt-5-mini",    # GPT-5シリーズ（推論特化）
        "gpt-5-nano",
        "gpt-5",
        "gpt-4o",        # GPT-4oシリーズ（マルチモーダル）
        "gpt-4o-mini",
        "gpt-4o-audio-preview",
        "gpt-4o-mini-audio-preview",
        "gpt-4.1",       # GPT-4.1シリーズ
        "gpt-4.1-mini",
        "o1",            # O-series（推論特化）
        "o1-mini",
        "o3",
        "o3-mini",
        "o4",
        "o4-mini"
    ]

    DEFAULT_MODEL = "gpt-5-mini"

    # モデル料金（1000トークンあたりのドル）
    MODEL_PRICING = {
        "gpt-5"     : {"input": 0.01, "output": 0.03},
        "gpt-5-mini": {"input": 0.0001, "output": 0.0004},
        "gpt-5-nano": {"input": 0.00005, "output": 0.0002},
        "gpt-4o"    : {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4.1"   : {"input": 0.0025, "output": 0.01},
        "gpt-4.1-mini": {"input": 0.0001, "output": 0.0004},
        "o1"        : {"input": 0.015, "output": 0.06},
        "o1-mini"   : {"input": 0.003, "output": 0.012},
        "o3"        : {"input": 0.03, "output": 0.12},
        "o3-mini"   : {"input": 0.006, "output": 0.024},
        "o4"        : {"input": 0.05, "output": 0.20},
        "o4-mini"   : {"input": 0.01, "output": 0.04},
    }

    # モデル制限（トークン上限）
    MODEL_LIMITS = {
        "gpt-5"     : {"max_tokens": 256000, "max_output": 8192},
        "gpt-5-mini": {"max_tokens": 128000, "max_output": 4096},
        "gpt-5-nano": {"max_tokens": 64000, "max_output": 2048},
        "gpt-4o"    : {"max_tokens": 128000, "max_output": 4096},
        "gpt-4o-mini": {"max_tokens": 128000, "max_output": 4096},
        "gpt-4.1"   : {"max_tokens": 128000, "max_output": 4096},
        "gpt-4.1-mini": {"max_tokens": 128000, "max_output": 4096},
        "o1"        : {"max_tokens": 128000, "max_output": 32768},
        "o1-mini"   : {"max_tokens": 128000, "max_output": 65536},
        "o3"        : {"max_tokens": 200000, "max_output": 100000},
        "o3-mini"   : {"max_tokens": 200000, "max_output": 100000},
        "o4"        : {"max_tokens": 256000, "max_output": 128000},
        "o4-mini"   : {"max_tokens": 256000, "max_output": 128000},
    }

    @classmethod
    def get_model_limits(cls, model: str) -> Dict[str, int]:
        """モデルの制限を取得"""
        return cls.MODEL_LIMITS.get(model, {"max_tokens": 128000, "max_output": 4096})

    @classmethod
    def get_model_pricing(cls, model: str) -> Dict[str, float]:
        """モデルの料金を取得"""
        return cls.MODEL_PRICING.get(model, {"input": 0.00015, "output": 0.0006})
```

#### 【実装コード】トークン管理クラス（helper_api.py:514-598）

```python
# helper_api.py - TokenManager クラス
class TokenManager:
    """トークン数の管理（新モデル対応）"""

    # モデル別のエンコーディング対応表
    MODEL_ENCODINGS = {
        "gpt-4o"    : "cl100k_base",
        "gpt-4o-mini": "cl100k_base",
        "gpt-4.1"   : "cl100k_base",
        "gpt-4.1-mini": "cl100k_base",
        "o1"        : "cl100k_base",
        "o1-mini"   : "cl100k_base",
        "o3"        : "cl100k_base",
        "o3-mini"   : "cl100k_base",
        "o4"        : "cl100k_base",
        "o4-mini"   : "cl100k_base",
    }

    @classmethod
    def count_tokens(cls, text: str, model: str = None) -> int:
        """テキストのトークン数をカウント"""
        if model is None:
            model = config.get("models.default", "gpt-4o-mini")

        try:
            encoding_name = cls.MODEL_ENCODINGS.get(model, "cl100k_base")
            enc = tiktoken.get_encoding(encoding_name)
            return len(enc.encode(text))
        except Exception as e:
            logger.error(f"トークンカウントエラー: {e}")
            # 簡易的な推定（1文字 = 0.5トークン）
            return len(text) // 2

    @classmethod
    def truncate_text(cls, text: str, max_tokens: int, model: str = None) -> str:
        """テキストを指定トークン数に切り詰め"""
        try:
            encoding_name = cls.MODEL_ENCODINGS.get(model, "cl100k_base")
            enc = tiktoken.get_encoding(encoding_name)
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return enc.decode(tokens[:max_tokens])
        except Exception as e:
            logger.error(f"テキスト切り詰めエラー: {e}")
            estimated_chars = max_tokens * 2
            return text[:estimated_chars]

    @classmethod
    def estimate_cost(cls, input_tokens: int, output_tokens: int, model: str = None) -> float:
        """API使用コストの推定"""
        pricing = AppConfig.get_model_pricing(model)
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost
```

---

## 8. Pydanticモデル定義

### 8.1 QAPair / QAPairsResponse

Q/Aペアの構造を定義するPydanticモデル。

#### QAPair

```python
class QAPair(BaseModel):
    question: str = Field(..., description="質問文")
    answer: str = Field(..., description="回答文")
    question_type: str = Field(
        default="fact",
        description="質問タイプ: fact/reason/comparison/application/..."
    )
    difficulty_level: Optional[str] = Field(
        default="medium",
        description="難易度: easy/medium/hard"
    )
    question_category: Optional[str] = Field(
        default="understanding",
        description="質問カテゴリ: basic/understanding/application"
    )
    source_chunk_id: Optional[str] = Field(default=None)
    dataset_type: Optional[str] = Field(default=None)
    auto_generated: bool = Field(default=False)
    confidence_score: Optional[float] = Field(default=None)
    quality_score: Optional[float] = Field(default=None)
```

#### QAPairsResponse

```python
class QAPairsResponse(BaseModel):
    qa_pairs: List[QAPair] = Field(
        default_factory=list,
        description="生成されたQ/Aペアのリスト"
    )
```

**実装箇所**: `models.py:24-75`

### 8.2 ChunkComplexity

チャンクの複雑度分析結果を格納するモデル。

```python
class ChunkComplexity(BaseModel):
    complexity_level: str = Field(
        default="medium",
        description="複雑度レベル: low/medium/high"
    )
    technical_terms: List[str] = Field(
        default_factory=list,
        description="専門用語リスト"
    )
    avg_sentence_length: float = Field(default=0.0)
    concept_density: float = Field(default=0.0)
    sentence_count: int = Field(default=0)
    token_count: int = Field(default=0)
```

**実装箇所**: `models.py:98-119`

### 8.3 構造化出力との連携

Pydanticモデルを `text_format` パラメータに指定することで、APIレスポンスが自動的にモデルインスタンスに変換される。

#### 連携フロー

```
[プロンプト送信]
    │
    ▼
[OpenAI API]
    │
    ▼
[JSON出力]
    │
    ▼
[Pydanticモデル自動変換]  ←── text_format=QAPairsResponse
    │
    ▼
[型安全なオブジェクト]
    │
    ├── response.output_parsed.qa_pairs[0].question
    └── response.output_parsed.qa_pairs[0].answer
```

---

## 9. データセット別設定

### 9.1 言語・チャンクサイズ設定

データセットごとに最適化された設定を持つ。

#### 設定一覧

| データセット | 言語 | チャンクサイズ | Q/A数/チャンク |
|-------------|------|--------------|---------------|
| wikipedia_ja | ja | 250 tokens | 3 |
| japanese_text | ja | 200 tokens | 2 |
| cc_news | en | 300 tokens | 5 |
| livedoor | ja | 200 tokens | 4 |

#### 設定定義（config.py）

```python
DATASETS = {
    "wikipedia_ja": DatasetInfo(
        name="Wikipedia日本語版",
        lang="ja",
        chunk_size=250,
        qa_per_chunk=3,
        min_text_length=200,
        # ...
    ),
    "cc_news": DatasetInfo(
        name="CC-News（英語ニュース）",
        lang="en",
        chunk_size=300,
        qa_per_chunk=5,
        min_text_length=100,
        # ...
    ),
    # ...
}
```

**実装箇所**: `config.py:151-213`

### 9.2 カバレージ閾値設定

データセットの特性に応じた最適カバレージ閾値。

#### 閾値一覧

| データセット | strict | standard | lenient |
|-------------|--------|----------|---------|
| cc_news | 0.80 | 0.70 | 0.60 |
| japanese_text | 0.75 | 0.65 | 0.55 |
| wikipedia_ja | 0.85 | 0.75 | 0.65 |
| livedoor | 0.78 | 0.68 | 0.58 |

#### 設定定義

```python
OPTIMAL_THRESHOLDS = {
    "cc_news": {
        "strict": 0.80,
        "standard": 0.70,
        "lenient": 0.60
    },
    "wikipedia_ja": {
        "strict": 0.85,    # 専門的な内容 → 高い類似度要求
        "standard": 0.75,
        "lenient": 0.65
    },
    # ...
}
```

#### 閾値設計の考え方

| データセット | 特徴 | 閾値傾向 |
|-------------|------|---------|
| wikipedia_ja | 専門的・百科事典的 | 高め |
| cc_news | 一般ニュース | 中程度 |
| livedoor | 日本語ニュース | 中程度 |
| japanese_text | Webテキスト・多様 | 低め |

**実装箇所**: `a02_make_qa_para.py:1467-1488`

### 9.3 Q/Aペア数の基本設定

データセットごとの `qa_per_chunk` 設定。

| データセット | qa_per_chunk | 理由 |
|-------------|--------------|------|
| wikipedia_ja | 3 | 情報密度が高いが文章が整理されている |
| japanese_text | 2 | Webテキストで品質にばらつき |
| cc_news | 5 | 英語ニュースは構造化されており情報量が多い |
| livedoor | 4 | 日本語ニュースで中程度の情報密度 |

---

## 10. 実装パターンと拡張方法

### 10.1 新規質問タイプの追加手順

#### Step 1: config.pyの階層構造を更新

```python
QUESTION_TYPES_HIERARCHY = {
    # ... 既存 ...
    "application": {
        # 新規タイプを追加
        "hypothetical": "仮説型（もし〜だったらどうなりますか？）",
    }
}
```

#### Step 2: プロンプトテンプレートを更新

`a02_make_qa_para.py` の日本語・英語プロンプトに追加：

```python
# 日本語
question_types_desc = """
- fact: 事実確認型（〜は何ですか？）
- reason: 理由説明型（なぜ〜ですか？）
- comparison: 比較型（〜と〜の違いは？）
- application: 応用型（〜はどのように活用されますか？）
- hypothetical: 仮説型（もし〜だったらどうなりますか？）  # 追加
"""
```

#### Step 3: celery_tasks.pyも同様に更新

同じプロンプトテンプレートを `celery_tasks.py` にも反映。

### 10.2 プロンプトテンプレートのカスタマイズ

#### システムプロンプトの変更

役割や生成ルールを変更する場合：

```python
# a02_make_qa_para.py
system_prompt = """あなたは{専門分野}の専門家です。
与えられたテキストから、{目的}に適したQ&Aペアを生成してください。

生成ルール:
1. {ルール1}
2. {ルール2}
...
"""
```

#### JSON出力形式の拡張

追加フィールドが必要な場合：

1. `models.py` の `QAPair` にフィールド追加
2. プロンプトのJSON形式指定を更新
3. レスポンス解析処理を更新

### 10.3 デバッグ・トラブルシューティング

#### よくある問題と対処

| 問題 | 原因 | 対処 |
|------|------|------|
| 空のレスポンス | max_output_tokens不足 | トークン数を増加（4000等） |
| JSONパースエラー | 出力が途中で切れている | max_output_tokensを増加 |
| 型エラー | Pydanticスキーマ不一致 | モデル定義とプロンプトを確認 |
| Q/A数不足 | チャンクが短すぎる | min_tokensを調整 |

#### デバッグログの有効化

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# API呼び出し前後でログ出力
logger.debug(f"プロンプト: {combined_input[:500]}...")
logger.debug(f"レスポンス: {response}")
```

#### プロンプトのテスト

小規模なテスト実行：

```bash
python a02_make_qa_para.py \
  --dataset livedoor \
  --max-docs 1 \
  --batch-chunks 1 \
  --model gpt-4o-mini
```

---

## 参考資料

- `doc/a02_make_qa_para.md`: a02_make_qa_para.pyの詳細ドキュメント
- `doc/celery_tasks.md`: Celeryタスクのドキュメント
- `doc/models.md`: Pydanticモデルのドキュメント
- `doc/helper_api.md`: OpenAI API関連のドキュメント

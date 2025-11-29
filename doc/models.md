# models.py 仕様書

作成日: 2025-11-27

## 概要

プロジェクト全体で使用されるPydanticモデル定義を一元管理するモジュール。Q/Aペア、チャンク、Celeryタスク結果、カバレージ分析、Qdrant関連のデータモデルを提供。

## ファイル情報

- **ファイル名**: models.py
- **行数**: 227行
- **主な機能**: Pydanticデータモデル定義
- **クラス数**: 10クラス + 1エイリアス
- **依存ライブラリ**: pydantic、typing、datetime

## 使用箇所

- `rag_qa_pair_qdrant.py`
- `celery_tasks.py`
- `a02_make_qa_para.py`
- `helper_rag_qa.py`

---

## アーキテクチャ

### モジュール構造

```
models.py
├── インポート (L15-17)
│
├── Q/A関連モデル (L20-76)
│   ├── QAPair (L24-63)
│   └── QAPairsResponse (L66-75)
│
├── チャンク関連モデル (L78-119)
│   ├── ChunkData (L82-95)
│   └── ChunkComplexity (L98-119)
│
├── Celeryタスク関連モデル (L122-140)
│   └── QAGenerationResult (L126-140)
│
├── カバレージ分析関連モデル (L143-160)
│   └── CoverageResult (L147-160)
│
├── Qdrant関連モデル (L163-192)
│   ├── QdrantPointPayload (L167-180)
│   └── QdrantCollectionStats (L183-192)
│
├── 処理結果モデル (L195-219)
│   ├── ProcessingResult (L199-210)
│   └── SavedFilesResult (L213-219)
│
└── 後方互換性エイリアス (L222-227)
    └── QAPairsList = QAPairsResponse
```

---

## モデル一覧

| モデル | 行番号 | 説明 |
|--------|--------|------|
| QAPair | L24-63 | Q/Aペアのデータモデル |
| QAPairsResponse | L66-75 | Q/Aペア生成レスポンス |
| ChunkData | L82-95 | テキストチャンクのデータモデル |
| ChunkComplexity | L98-119 | チャンクの複雑度分析結果 |
| QAGenerationResult | L126-140 | Q/A生成タスクの結果 |
| CoverageResult | L147-160 | カバレージ分析結果 |
| QdrantPointPayload | L167-180 | Qdrantポイントのペイロード |
| QdrantCollectionStats | L183-192 | Qdrantコレクション統計情報 |
| ProcessingResult | L199-210 | 汎用処理結果 |
| SavedFilesResult | L213-219 | ファイル保存結果 |

---

## Q/A関連モデル

### QAPair (L24-63)

Q/Aペアのデータモデル。基本的なQ/Aペア情報に加え、品質・難易度のメタデータを含む。

```python
class QAPair(BaseModel):
    question: str = Field(..., description="質問文")
    answer: str = Field(..., description="回答文")
    question_type: str = Field(
        default="fact",
        description="質問タイプ"
    )
    difficulty_level: Optional[str] = Field(
        default="medium",
        description="難易度"
    )
    question_category: Optional[str] = Field(
        default="understanding",
        description="質問カテゴリ"
    )
    source_chunk_id: Optional[str] = Field(
        default=None,
        description="ソースチャンクID"
    )
    dataset_type: Optional[str] = Field(
        default=None,
        description="データセットタイプ"
    )
    auto_generated: bool = Field(
        default=False,
        description="自動生成フラグ"
    )
    confidence_score: Optional[float] = Field(
        default=None,
        description="生成の確信度 (0.0-1.0)"
    )
    quality_score: Optional[float] = Field(
        default=None,
        description="品質スコア (0.0-1.0)"
    )
```

#### フィールド詳細

| フィールド | 型 | 必須 | デフォルト | 説明 |
|-----------|-----|------|-----------|------|
| `question` | str | Yes | - | 質問文 |
| `answer` | str | Yes | - | 回答文 |
| `question_type` | str | No | "fact" | 質問タイプ |
| `difficulty_level` | str | No | "medium" | 難易度 |
| `question_category` | str | No | "understanding" | 質問カテゴリ |
| `source_chunk_id` | str | No | None | ソースチャンクID |
| `dataset_type` | str | No | None | データセットタイプ |
| `auto_generated` | bool | No | False | 自動生成フラグ |
| `confidence_score` | float | No | None | 生成の確信度 (0.0-1.0) |
| `quality_score` | float | No | None | 品質スコア (0.0-1.0) |

#### question_typeの値

| 値 | 説明 |
|----|------|
| fact | 事実確認型 |
| reason | 理由説明型 |
| comparison | 比較型 |
| application | 応用型 |
| definition | 定義型 |
| process | プロセス型 |
| evaluation | 評価型 |

#### difficulty_levelの値

| 値 | 説明 |
|----|------|
| easy | 簡単 |
| medium | 中程度 |
| hard | 難しい |

#### question_categoryの値

| 値 | 説明 |
|----|------|
| basic | 基本 |
| understanding | 理解 |
| application | 応用 |

---

### QAPairsResponse (L66-75)

Q/Aペア生成レスポンス。OpenAI APIの構造化出力で使用。

```python
class QAPairsResponse(BaseModel):
    qa_pairs: List[QAPair] = Field(
        default_factory=list,
        description="生成されたQ/Aペアのリスト"
    )
```

#### 使用例（OpenAI構造化出力）

```python
response = client.responses.parse(
    input=combined_input,
    model=model,
    text_format=QAPairsResponse,  # このモデルを指定
    max_output_tokens=2000
)
```

---

## チャンク関連モデル

### ChunkData (L82-95)

テキストチャンクのデータモデル。

```python
class ChunkData(BaseModel):
    id: str = Field(..., description="チャンクID")
    text: str = Field(..., description="チャンクテキスト")
    tokens: int = Field(default=0, description="トークン数")
    doc_id: Optional[str] = Field(default=None, description="ドキュメントID")
    dataset_type: Optional[str] = Field(default=None, description="データセットタイプ")
    chunk_idx: int = Field(default=0, description="チャンクインデックス")
    position: Optional[str] = Field(
        default=None,
        description="チャンクの位置: start/middle/end"
    )
```

#### フィールド詳細

| フィールド | 型 | 必須 | デフォルト | 説明 |
|-----------|-----|------|-----------|------|
| `id` | str | Yes | - | チャンクID |
| `text` | str | Yes | - | チャンクテキスト |
| `tokens` | int | No | 0 | トークン数 |
| `doc_id` | str | No | None | ドキュメントID |
| `dataset_type` | str | No | None | データセットタイプ |
| `chunk_idx` | int | No | 0 | チャンクインデックス |
| `position` | str | No | None | チャンクの位置 |

#### positionの値

| 値 | 説明 |
|----|------|
| start | 最初のチャンク |
| middle | 中間のチャンク |
| end | 最後のチャンク |

---

### ChunkComplexity (L98-119)

チャンクの複雑度分析結果。

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
    avg_sentence_length: float = Field(
        default=0.0,
        description="平均文長"
    )
    concept_density: float = Field(
        default=0.0,
        description="概念密度"
    )
    sentence_count: int = Field(default=0, description="文数")
    token_count: int = Field(default=0, description="トークン数")
```

#### フィールド詳細

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `complexity_level` | str | "medium" | 複雑度レベル |
| `technical_terms` | List[str] | [] | 専門用語リスト |
| `avg_sentence_length` | float | 0.0 | 平均文長 |
| `concept_density` | float | 0.0 | 概念密度 |
| `sentence_count` | int | 0 | 文数 |
| `token_count` | int | 0 | トークン数 |

---

## Celeryタスク関連モデル

### QAGenerationResult (L126-140)

Q/A生成タスクの結果。

```python
class QAGenerationResult(BaseModel):
    success: bool = Field(..., description="成功フラグ")
    chunk_id: Optional[str] = Field(default=None, description="チャンクID")
    chunk_ids: Optional[List[str]] = Field(
        default=None,
        description="バッチ処理時のチャンクIDリスト"
    )
    qa_pairs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="生成されたQ/Aペア"
    )
    error: Optional[str] = Field(default=None, description="エラーメッセージ")
```

#### フィールド詳細

| フィールド | 型 | 必須 | デフォルト | 説明 |
|-----------|-----|------|-----------|------|
| `success` | bool | Yes | - | 成功フラグ |
| `chunk_id` | str | No | None | チャンクID |
| `chunk_ids` | List[str] | No | None | バッチ処理時のチャンクIDリスト |
| `qa_pairs` | List[Dict] | No | [] | 生成されたQ/Aペア |
| `error` | str | No | None | エラーメッセージ |

---

## カバレージ分析関連モデル

### CoverageResult (L147-160)

カバレージ分析結果。

```python
class CoverageResult(BaseModel):
    coverage_rate: float = Field(
        default=0.0,
        description="カバレージ率 (0.0-1.0)"
    )
    covered_chunks: int = Field(default=0, description="カバーされたチャンク数")
    total_chunks: int = Field(default=0, description="総チャンク数")
    uncovered_chunks: List[str] = Field(
        default_factory=list,
        description="未カバーチャンクIDリスト"
    )
```

#### フィールド詳細

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `coverage_rate` | float | 0.0 | カバレージ率 (0.0-1.0) |
| `covered_chunks` | int | 0 | カバーされたチャンク数 |
| `total_chunks` | int | 0 | 総チャンク数 |
| `uncovered_chunks` | List[str] | [] | 未カバーチャンクIDリスト |

---

## Qdrant関連モデル

### QdrantPointPayload (L167-180)

Qdrantポイントのペイロード。

```python
class QdrantPointPayload(BaseModel):
    domain: str = Field(..., description="ドメイン名")
    question: str = Field(..., description="質問文")
    answer: str = Field(..., description="回答文")
    source: str = Field(..., description="ソースファイル名")
    created_at: str = Field(..., description="作成日時 (ISO形式)")
    schema_version: str = Field(default="qa:v1", description="スキーマバージョン")
    generation_method: Optional[str] = Field(
        default=None,
        description="生成方法"
    )
```

#### フィールド詳細

| フィールド | 型 | 必須 | デフォルト | 説明 |
|-----------|-----|------|-----------|------|
| `domain` | str | Yes | - | ドメイン名 |
| `question` | str | Yes | - | 質問文 |
| `answer` | str | Yes | - | 回答文 |
| `source` | str | Yes | - | ソースファイル名 |
| `created_at` | str | Yes | - | 作成日時 (ISO形式) |
| `schema_version` | str | No | "qa:v1" | スキーマバージョン |
| `generation_method` | str | No | None | 生成方法 |

---

### QdrantCollectionStats (L183-192)

Qdrantコレクション統計情報。

```python
class QdrantCollectionStats(BaseModel):
    total_points: int = Field(default=0, description="総ポイント数")
    vector_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="ベクトル設定"
    )
    status: str = Field(default="unknown", description="ステータス")
```

#### フィールド詳細

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `total_points` | int | 0 | 総ポイント数 |
| `vector_config` | Dict | {} | ベクトル設定 |
| `status` | str | "unknown" | ステータス |

---

## 処理結果モデル

### ProcessingResult (L199-210)

汎用処理結果。

```python
class ProcessingResult(BaseModel):
    success: bool = Field(..., description="成功フラグ")
    message: Optional[str] = Field(default=None, description="メッセージ")
    data: Optional[Dict[str, Any]] = Field(default=None, description="データ")
    error: Optional[str] = Field(default=None, description="エラーメッセージ")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="タイムスタンプ"
    )
```

#### フィールド詳細

| フィールド | 型 | 必須 | デフォルト | 説明 |
|-----------|-----|------|-----------|------|
| `success` | bool | Yes | - | 成功フラグ |
| `message` | str | No | None | メッセージ |
| `data` | Dict | No | None | データ |
| `error` | str | No | None | エラーメッセージ |
| `timestamp` | str | No | (現在時刻) | タイムスタンプ |

---

### SavedFilesResult (L213-219)

ファイル保存結果。

```python
class SavedFilesResult(BaseModel):
    csv_path: Optional[str] = Field(default=None, description="CSVファイルパス")
    json_path: Optional[str] = Field(default=None, description="JSONファイルパス")
    txt_path: Optional[str] = Field(default=None, description="テキストファイルパス")
```

#### フィールド詳細

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `csv_path` | str | None | CSVファイルパス |
| `json_path` | str | None | JSONファイルパス |
| `txt_path` | str | None | テキストファイルパス |

---

## 後方互換性エイリアス (L222-227)

旧名称でのインポートをサポート。

```python
QAPairsList = QAPairsResponse
```

---

## 使用例

### 例1: QAPairの作成

```python
from models import QAPair

qa = QAPair(
    question="RAGとは何ですか？",
    answer="RAGはRetrieval-Augmented Generationの略で...",
    question_type="definition",
    difficulty_level="easy",
    auto_generated=True,
    confidence_score=0.95
)

print(qa.model_dump())
```

### 例2: QAPairsResponse（OpenAI構造化出力）

```python
from models import QAPairsResponse
from openai import OpenAI

client = OpenAI()

response = client.responses.parse(
    input=messages,
    model="gpt-5-mini",
    text_format=QAPairsResponse,
    max_output_tokens=2000
)

# レスポンスからQ/Aペアを取得
qa_pairs = response.output_parsed.qa_pairs
for qa in qa_pairs:
    print(f"Q: {qa.question}")
    print(f"A: {qa.answer}")
```

### 例3: ChunkDataの作成

```python
from models import ChunkData

chunk = ChunkData(
    id="doc_001_chunk_0",
    text="これはサンプルテキストです。",
    tokens=15,
    doc_id="doc_001",
    chunk_idx=0,
    position="start"
)
```

### 例4: QAGenerationResultの使用

```python
from models import QAGenerationResult

# 成功ケース
result = QAGenerationResult(
    success=True,
    chunk_id="chunk_001",
    qa_pairs=[
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"}
    ]
)

# 失敗ケース
error_result = QAGenerationResult(
    success=False,
    chunk_id="chunk_002",
    error="API rate limit exceeded"
)
```

### 例5: CoverageResultの使用

```python
from models import CoverageResult

coverage = CoverageResult(
    coverage_rate=0.85,
    covered_chunks=17,
    total_chunks=20,
    uncovered_chunks=["chunk_3", "chunk_12", "chunk_18"]
)

print(f"カバレージ率: {coverage.coverage_rate:.1%}")
```

### 例6: QdrantPointPayloadの作成

```python
from models import QdrantPointPayload
from datetime import datetime

payload = QdrantPointPayload(
    domain="customer_support",
    question="パスワードを忘れた場合は？",
    answer="パスワードリセット機能をご利用ください。",
    source="faq.csv",
    created_at=datetime.now().isoformat(),
    generation_method="hybrid"
)
```

### 例7: ProcessingResultの使用

```python
from models import ProcessingResult

# 成功結果
result = ProcessingResult(
    success=True,
    message="処理が完了しました",
    data={"processed_count": 100, "skipped_count": 5}
)

# エラー結果
error_result = ProcessingResult(
    success=False,
    error="ファイルが見つかりません"
)
```

---

## 依存ライブラリ

```python
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
```

---

## 注意事項

1. **Pydantic V2**: このモジュールはPydantic V2を使用
2. **OpenAI構造化出力**: `QAPairsResponse`は`client.responses.parse()`で使用
3. **後方互換性**: `QAPairsList`は`QAPairsResponse`のエイリアス
4. **タイムスタンプ**: `ProcessingResult.timestamp`は自動生成

---

## エクスポート一覧

```python
# Q/A関連
QAPair
QAPairsResponse

# チャンク関連
ChunkData
ChunkComplexity

# Celeryタスク関連
QAGenerationResult

# カバレージ分析関連
CoverageResult

# Qdrant関連
QdrantPointPayload
QdrantCollectionStats

# 処理結果
ProcessingResult
SavedFilesResult

# 後方互換性エイリアス
QAPairsList  # = QAPairsResponse
```

---

## まとめ

models.pyは、プロジェクト全体で使用されるPydanticデータモデルを一元管理するモジュールです。

### 主要な特徴

1. **型安全性**: Pydanticによる厳密な型定義
2. **バリデーション**: Field定義による入力検証
3. **ドキュメント**: 各フィールドにdescription付き
4. **OpenAI連携**: 構造化出力用モデル（QAPairsResponse）
5. **後方互換性**: 旧名称のエイリアスをサポート

### 推奨用途

- Q/Aペアのデータ構造定義
- Celeryタスクの結果管理
- Qdrantへのデータ登録
- API レスポンスの型定義

---

作成日: 2025-11-27
作成者: OpenAI RAG Q/A JP Development Team
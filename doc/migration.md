# OpenAI API から Gemini 3 API への移行計画書

作成日: 2025-11-29
更新日: 2025-11-29

---

## 目次

1. [概要](#1-概要)
2. [現状分析：OpenAI API使用箇所](#2-現状分析openai-api使用箇所)
3. [Gemini 3 API仕様](#3-gemini-3-api仕様)
4. [API対応表：OpenAI vs Gemini](#4-api対応表openai-vs-gemini)
5. [非互換点と対応方針](#5-非互換点と対応方針)
6. [移行計画](#6-移行計画)
7. [コード変換例](#7-コード変換例)
8. [検証計画](#8-検証計画)
9. [参考資料](#9-参考資料)

---

## 1. 概要

### 1.1 移行目的

本プロジェクトは、現在OpenAI APIを利用しているRAG Q/Aシステムを、Google Gemini 3 APIに移行・移植し、動作と効果を検証することを目的とする。

### 1.2 対象システム

- **プロジェクト名**: gemini3_rag_qa（旧: openai_rag_qa）
- **主要機能**:
  - Q/Aペア生成（LLMによる構造化出力）
  - テキストEmbedding生成
  - Qdrantベクトルデータベース連携
  - Semantic Coverage分析

### 1.3 移行方針

| 項目 | 方針 |
|------|------|
| SDK | `google-genai` Python SDKを使用 |
| モデル | Gemini 3 Pro Preview (`gemini-3-pro-preview`) |
| Embedding | `gemini-embedding-001` (3072次元: Gemini 3のアドバンテージ活用) |
| 構造化出力 | Pydantic優先、使用不可の場合はJSONスキーマ |

---

## 2. 現状分析：OpenAI API使用箇所

### 2.1 使用APIパターン一覧

| API種別 | OpenAIメソッド | 用途 | 主要使用ファイル |
|---------|----------------|------|------------------|
| **Responses API** | `client.responses.create()` | テキスト生成 | `helper_api.py:735` |
| **Structured Outputs** | `client.responses.parse()` | 構造化出力（Q/A生成） | `celery_tasks.py:299`, `a02_make_qa_para.py:1044` |
| **Chat Completions** | `client.chat.completions.create()` | チャット形式対話 | `helper_api.py:750`, `celery_tasks.py:353` |
| **Embeddings** | `client.embeddings.create()` | ベクトル化 | `qdrant_client_wrapper.py:476`, `services/qdrant_service.py:508` |

### 2.2 影響ファイル一覧

#### コアモジュール（要移行：高優先度）

| ファイル | 行数 | OpenAI使用内容 | 影響度 |
|----------|------|----------------|--------|
| `helper_api.py` | 840 | APIクライアント、メッセージ管理、トークン管理 | **高** |
| `celery_tasks.py` | 600+ | Q/A生成（構造化出力）、並列処理 | **高** |
| `a02_make_qa_para.py` | 1400+ | 並列Q/A生成 | **高** |
| `helper_rag_qa.py` | 3600+ | SemanticCoverage、Embeddings | **高** |
| `qdrant_client_wrapper.py` | 900+ | Embedding生成、Qdrant連携 | **高** |
| `services/qdrant_service.py` | 700+ | Embedding生成（検索用） | **高** |

#### アプリケーションモジュール（中優先度）

| ファイル | OpenAI使用内容 |
|----------|----------------|
| `a50_rag_search_local_qdrant.py` | 検索UI、Embedding |
| `a42_qdrant_registration.py` | Qdrant登録、Embedding |
| `a02_make_qa_single.py` | 単一Q/A生成 |
| `services/qa_service.py` | Q/Aサービス |
| `ui/pages/qdrant_search_page.py` | 検索UI |

### 2.3 OpenAI固有機能の使用状況

#### 2.3.1 構造化出力（Structured Outputs）

```python
# 現在の実装: celery_tasks.py:299-304
response = client.responses.parse(
    input=combined_input,
    model=model,
    text_format=QAPairsResponse,  # Pydanticモデル
    max_output_tokens=2000
)
```

#### 2.3.2 Embeddings

```python
# 現在の実装: qdrant_client_wrapper.py:476
resp = client.embeddings.create(
    model="text-embedding-3-small",  # 1536次元
    input=texts
)
```

#### 2.3.3 メッセージ形式

```python
# 現在の実装: helper_api.py
EasyInputMessageParam(role="developer", content=developer_text)
EasyInputMessageParam(role="user", content=user_text)
EasyInputMessageParam(role="assistant", content=assistant_text)
```

---

## 3. Gemini 3 API仕様

### 3.1 利用可能なモデル

| モデル名 | 入力トークン | 出力トークン | 特徴 |
|----------|-------------|-------------|------|
| `gemini-3-pro-preview` | 1,000,000 | 64,000 | 最新の推論モデル、思考レベル制御対応 |
| `gemini-3-pro-image-preview` | 65,000 | 32,000 | 画像生成・編集対応 |
| `gemini-2.5-flash-preview` | 1,000,000 | 64,000 | 高速・低コスト |
| `gemini-2.5-pro-preview` | 1,000,000 | 64,000 | 高品質推論 |

### 3.2 Embedding モデル

| モデル名 | 次元数 | 入力上限 | 備考 |
|----------|--------|---------|------|
| `gemini-embedding-001` | 128〜3072（可変） | 2,048トークン | 推奨: 768, 1536, 3072 |
| `text-embedding-004` | 768 | 2,048トークン | 2026年1月廃止予定 |

### 3.3 料金体系

| モデル | 入力（100万トークン） | 出力（100万トークン） |
|--------|----------------------|----------------------|
| gemini-3-pro-preview | $2 | $12 |
| gemini-2.5-flash | $0.15 | $0.60 |
| gemini-embedding-001 | $0.00 (無料枠あり) | - |

### 3.4 Python SDK基本構成

```python
# インストール
pip install google-genai

# クライアント初期化
from google import genai
client = genai.Client(api_key='GEMINI_API_KEY')

# または環境変数から自動取得
# export GOOGLE_API_KEY=your-api-key
client = genai.Client()
```

### 3.5 主要パラメータ

#### 3.5.1 思考レベル（thinking_level）

Gemini 3の新機能。モデルの推論深度を制御：

| 値 | 説明 | 用途 |
|----|------|------|
| `low` | レイテンシ最小化 | 簡単なタスク |
| `high`（デフォルト） | 推論深化 | 複雑なタスク |

```python
response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents="複雑な質問",
    config={"thinking_level": "high"}
)
```

#### 3.5.2 温度設定（temperature）

> **注意**: Gemini 3ではデフォルト値の1.0を維持することを強く推奨。
> 変更すると複雑なタスクで予期しない動作が発生する可能性がある。

---

## 4. API対応表：OpenAI vs Gemini

### 4.1 テキスト生成

| 項目 | OpenAI | Gemini 3 |
|------|--------|----------|
| **メソッド** | `client.responses.create()` | `client.models.generate_content()` |
| **入力パラメータ** | `input`, `model` | `contents`, `model` |
| **出力取得** | `response.output_text` | `response.text` |
| **ストリーミング** | `stream=True` | `stream=True` |

### 4.2 構造化出力

| 項目 | OpenAI | Gemini 3 |
|------|--------|----------|
| **メソッド** | `client.responses.parse()` | `client.models.generate_content()` |
| **スキーマ指定** | `text_format=PydanticModel` | `response_json_schema=Model.model_json_schema()` |
| **MIMEタイプ** | 自動 | `response_mime_type="application/json"` |
| **出力取得** | `response.output_parsed` | `json.loads(response.text)` |

### 4.3 Embeddings

| 項目 | OpenAI | Gemini 3 |
|------|--------|----------|
| **メソッド** | `client.embeddings.create()` | `client.models.embed_content()` |
| **モデル** | `text-embedding-3-small` | `gemini-embedding-001` |
| **デフォルト次元** | 1536 | 3072（768/1536選択可） |
| **バッチ処理** | `input=[texts]` | Batch API（50%割引） |

### 4.4 メッセージロール

| OpenAI | Gemini | 対応方針 |
|--------|--------|----------|
| `developer` | なし | システムプロンプトとして `contents` に含める |
| `system` | なし | `system_instruction` パラメータを使用 |
| `user` | `user` | そのまま対応 |
| `assistant` | `model` | ロール名を変換 |

---

## 5. 非互換点と対応方針

### 5.1 Embedding次元数の違い

| 課題 | 対応方針 |
|------|----------|
| OpenAI: 1536次元 / Gemini: 768〜3072次元 | **Gemini側を3072次元に設定（Gemini 3のアドバンテージ活用）** |
| Qdrantコレクション再構築が必要 | 新コレクション作成（3072次元）、全データ再Embedding |

> **設計判断**: OpenAIとの互換性より、Gemini 3の高次元Embeddingによる精度向上を優先

```python
# Gemini Embedding: 3072次元で生成（最大精度）
response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text,
    config={"output_dimensionality": 3072}  # Gemini 3最大次元数
)
```

### 5.2 構造化出力の方式差異

| 課題 | 対応方針 |
|------|----------|
| OpenAI: `text_format` で Pydanticモデル直接指定 | **Pydantic優先**: Gemini SDKがPydantic対応なら使用、非対応なら`response_json_schema`でJSONスキーマ指定 |
| レスポンス形式が異なる | パーサーラッパーを実装、Pydanticでバリデーション |

### 5.3 トークン管理

| 課題 | 対応方針 |
|------|----------|
| OpenAI: tiktoken (`cl100k_base`) | Gemini: 独自トークナイザー |
| 正確なトークン数計算が困難 | 文字数ベースの推定 + Gemini API `count_tokens` |

### 5.4 レスポンス構造

| OpenAI | Gemini | 対応 |
|--------|--------|------|
| `response.output` | `response.candidates` | レスポンスパーサー実装 |
| `response.output_text` | `response.text` | プロパティマッピング |
| `response.usage.total_tokens` | `response.usage_metadata` | 使用量取得ラッパー |

---

## 6. 移行計画

### 6.1 フェーズ概要

```
Phase 1: 抽象化レイヤー設計・実装（1週間）
    ↓
Phase 2: Gemini APIクライアント実装（1週間）
    ↓
Phase 3: 既存コードの移行（2週間）
    ↓
Phase 4: Qdrantコレクション再構築（1週間）
    ↓
Phase 5: 動作検証・比較評価（1週間）
```

### 6.2 Phase 1: 抽象化レイヤー設計

**目的**: OpenAI/Gemini両方に対応する統一インターフェースを設計

**成果物**:
- `helper_llm.py`: LLMクライアント抽象化
- `helper_embedding.py`: Embedding抽象化

**クラス設計**:

```python
# helper_llm.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class LLMClient(ABC):
    """LLMクライアント抽象基底クラス"""

    @abstractmethod
    def generate_content(
        self,
        prompt: str,
        model: str = None,
        **kwargs
    ) -> str:
        """テキスト生成"""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        model: str = None,
        **kwargs
    ) -> BaseModel:
        """構造化出力生成"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI実装（既存互換用）"""
    pass


class GeminiClient(LLMClient):
    """Gemini 3実装"""
    pass
```

```python
# helper_embedding.py
from abc import ABC, abstractmethod
from typing import List

class EmbeddingClient(ABC):
    """Embeddingクライアント抽象基底クラス"""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """単一テキストのEmbedding"""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """バッチEmbedding"""
        pass


class OpenAIEmbedding(EmbeddingClient):
    """OpenAI Embedding実装"""
    pass


class GeminiEmbedding(EmbeddingClient):
    """Gemini Embedding実装"""
    pass
```

### 6.3 Phase 2: Gemini APIクライアント実装

**目的**: Gemini 3 API用のクライアント実装

**タスク**:

1. **GeminiClientクラス実装**
   - `generate_content()`: テキスト生成
   - `generate_structured()`: 構造化出力
   - 思考レベル制御対応

2. **GeminiEmbeddingクラス実装**
   - `embed_text()`: 単一Embedding
   - `embed_texts()`: バッチEmbedding
   - 次元数設定対応（3072次元: Gemini 3最大精度）

3. **設定管理**
   - `config.yml` にGemini設定追加
   - 環境変数対応（`GOOGLE_API_KEY`）

### 6.4 Phase 3: 既存コードの移行

**目的**: 既存のOpenAI呼び出しをGemini対応に変換

**対象ファイルと作業内容**:

| ファイル | 作業内容 |
|----------|----------|
| `helper_api.py` | `GeminiClient` への切り替え、設定追加 |
| `celery_tasks.py` | Q/A生成をGemini構造化出力に変更 |
| `a02_make_qa_para.py` | 並列処理のGemini対応 |
| `helper_rag_qa.py` | SemanticCoverageのEmbedding切り替え |
| `qdrant_client_wrapper.py` | `GeminiEmbedding` 使用 |
| `services/qdrant_service.py` | Embedding生成の切り替え |

### 6.5 Phase 4: Qdrantコレクション再構築

**目的**: Gemini Embeddingでベクトルデータを再生成

**タスク**:

1. 既存コレクションのバックアップ
2. 新コレクション作成（3072次元: Gemini 3アドバンテージ活用）
3. 全テキストデータの再Embedding
4. Qdrantへの再登録
5. 検索精度の検証

**スクリプト例**:

```python
# migration_qdrant.py
from helper_embedding import GeminiEmbedding
from qdrant_client_wrapper import create_qdrant_client

def migrate_collection(collection_name: str):
    """コレクションをGemini Embeddingで再構築（3072次元）"""
    embedding_client = GeminiEmbedding(dims=3072)
    qdrant = create_qdrant_client()

    # 既存データ取得
    points = qdrant.scroll(collection_name, limit=10000)

    # 再Embedding
    for point in points:
        text = point.payload.get("text", "")
        new_vector = embedding_client.embed_text(text)
        # 更新処理...
```

### 6.6 Phase 5: 動作検証・比較評価

**目的**: OpenAI版とGemini版の性能・品質比較

**評価項目**:

| 評価項目 | 測定方法 |
|----------|----------|
| Q/A生成品質 | 人手評価 + 自動評価（BLEU, ROUGE） |
| Embedding精度 | コサイン類似度比較 |
| 応答時間 | レイテンシ計測 |
| コスト | トークン使用量 × 単価 |
| 日本語性能 | 日本語テストセットでの評価 |

---

## 7. コード変換例

### 7.1 テキスト生成

**OpenAI（現在）**:
```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
)
print(response.output_text)
```

**Gemini 3（移行後）**:
```python
from google import genai

client = genai.Client()
response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents="Hello, how are you?",
    config={
        "system_instruction": "You are a helpful assistant.",
        "thinking_level": "low"
    }
)
print(response.text)
```

### 7.2 構造化出力（Q/A生成）

**OpenAI（現在）**:
```python
from openai import OpenAI
from models import QAPairsResponse

client = OpenAI()
response = client.responses.parse(
    input=prompt,
    model="gpt-4o-mini",
    text_format=QAPairsResponse,
    max_output_tokens=2000
)
qa_pairs = response.output_parsed.qa_pairs
```

**Gemini 3（移行後）**:
```python
from google import genai
from models import QAPairsResponse
import json

client = genai.Client()
response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "response_json_schema": QAPairsResponse.model_json_schema(),
        "thinking_level": "high"
    }
)
qa_data = json.loads(response.text)
qa_pairs = QAPairsResponse(**qa_data).qa_pairs
```

### 7.3 Embedding生成

**OpenAI（現在）**:
```python
from openai import OpenAI

client = OpenAI()
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["Hello world", "How are you?"]
)
vectors = [item.embedding for item in response.data]
```

**Gemini 3（移行後）**:
```python
from google import genai

client = genai.Client()

# 単一テキスト（3072次元: Gemini 3最大精度）
response = client.models.embed_content(
    model="gemini-embedding-001",
    contents="Hello world",
    config={"output_dimensionality": 3072}
)
vector = response.embeddings[0].values

# バッチ処理（複数テキスト）
vectors = []
for text in ["Hello world", "How are you?"]:
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={"output_dimensionality": 3072}
    )
    vectors.append(resp.embeddings[0].values)
```

### 7.4 抽象化クライアント使用例

**移行後の統一インターフェース**:
```python
from helper_llm import create_llm_client
from helper_embedding import create_embedding_client
from models import QAPairsResponse

# 設定に基づいてクライアント生成（OpenAI/Gemini切り替え可能）
llm = create_llm_client(provider="gemini")
embedding = create_embedding_client(provider="gemini")

# テキスト生成
text = llm.generate_content("Hello, how are you?")

# 構造化出力
qa_response = llm.generate_structured(prompt, QAPairsResponse)

# Embedding
vector = embedding.embed_text("Hello world")
vectors = embedding.embed_texts(["Hello", "World"], batch_size=100)
```

---

## 8. 検証計画

### 8.1 単体テスト

| テスト対象 | テスト内容 |
|------------|------------|
| `GeminiClient.generate_content()` | テキスト生成の正常動作 |
| `GeminiClient.generate_structured()` | 構造化出力のスキーマ準拠 |
| `GeminiEmbedding.embed_text()` | 3072次元ベクトル生成 |
| `GeminiEmbedding.embed_texts()` | バッチ処理の正常動作 |

### 8.2 統合テスト

| テストシナリオ | 期待結果 |
|----------------|----------|
| Q/Aペア生成（日本語） | 有効なQ/Aペアが生成される |
| Qdrant検索（Gemini Embedding） | 関連ドキュメントが検索される |
| Semantic Coverage計算 | カバレッジ率が計算される |

### 8.3 性能比較テスト

| 比較項目 | OpenAI基準値 | Gemini目標値 |
|----------|-------------|-------------|
| Q/A生成時間 | 測定 | 同等以下 |
| Embedding生成時間 | 測定 | 同等以下 |
| 検索精度（Top-5 Recall） | 測定 | 90%以上維持 |
| API費用（1000リクエスト） | 測定 | 同等以下 |

---

## 9. 参考資料

### 9.1 公式ドキュメント

- [Gemini 3 API ドキュメント](https://ai.google.dev/gemini-api/docs/gemini-3?hl=ja)
- [Gemini API リファレンス](https://ai.google.dev/api?hl=ja)
- [Python Gen AI SDK](https://googleapis.github.io/python-genai/)
- [Gemini Embedding ドキュメント](https://ai.google.dev/gemini-api/docs/embeddings?hl=ja)
- [構造化出力ガイド](https://ai.google.dev/gemini-api/docs/structured-output?hl=ja)

### 9.2 プロジェクト内ドキュメント

- `doc/helper_api.md`: 現行OpenAI APIヘルパー仕様
- `doc/celery_tasks.md`: Celeryタスク仕様
- `doc/qdrant_client_wrapper.md`: Qdrantラッパー仕様
- `CLAUDE.md`: プロジェクト開発ガイドライン

### 9.3 移行チェックリスト

- [ ] `google-genai` パッケージインストール
- [ ] `GOOGLE_API_KEY` 環境変数設定
- [ ] `helper_llm.py` 抽象化レイヤー実装
- [ ] `helper_embedding.py` 抽象化レイヤー実装
- [ ] `GeminiClient` クラス実装
- [ ] `GeminiEmbedding` クラス実装
- [ ] `config.yml` Gemini設定追加
- [ ] `celery_tasks.py` Gemini対応
- [ ] `helper_rag_qa.py` Gemini対応
- [ ] `qdrant_client_wrapper.py` Gemini対応
- [ ] Qdrantコレクション再構築
- [ ] 単体テスト実施
- [ ] 統合テスト実施
- [ ] 性能比較テスト実施

---

## 付録A: 環境変数設定

```bash
# .env ファイル
# OpenAI（既存・互換用）
OPENAI_API_KEY=sk-xxx

# Gemini（新規）
GOOGLE_API_KEY=AIzaXXX

# Qdrant
QDRANT_URL=http://localhost:6333

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

## 付録B: config.yml 追加設定

```yaml
# Gemini API設定
gemini:
  default_model: "gemini-3-pro-preview"
  embedding_model: "gemini-embedding-001"
  embedding_dims: 3072  # Gemini 3最大精度
  thinking_level: "high"
  temperature: 1.0  # Gemini 3推奨値

  models:
    - gemini-3-pro-preview
    - gemini-2.5-flash-preview
    - gemini-2.5-pro-preview

  pricing:
    gemini-3-pro-preview:
      input: 0.002   # per 1K tokens
      output: 0.012  # per 1K tokens
    gemini-2.5-flash-preview:
      input: 0.00015
      output: 0.0006

# API切り替え設定
llm_provider: "gemini"  # "openai" or "gemini"
embedding_provider: "gemini"  # "openai" or "gemini"
```

---

## 10. 詳細TODO計画

### 10.1 フェーズ別タスク一覧

```
================================================================================
                    OpenAI → Gemini 3 Migration TODO
================================================================================

Phase 0: 環境準備
├── [P0-1] google-genai パッケージインストール
├── [P0-2] GOOGLE_API_KEY 取得・設定
├── [P0-3] Gemini API動作確認（サンプルスクリプト）
└── [P0-4] requirements.txt 更新

Phase 1: 抽象化レイヤー設計・実装
├── [P1-1] helper_llm.py 新規作成（LLMClient抽象クラス）
├── [P1-2] helper_embedding.py 新規作成（EmbeddingClient抽象クラス）
├── [P1-3] OpenAIClient実装（既存互換）
├── [P1-4] OpenAIEmbedding実装（既存互換）
├── [P1-5] 単体テスト作成（tests/test_helper_llm.py）
└── [P1-6] 単体テスト作成（tests/test_helper_embedding.py）

Phase 2: Gemini APIクライアント実装
├── [P2-1] GeminiClient.generate_content() 実装
├── [P2-2] GeminiClient.generate_structured() 実装
├── [P2-3] GeminiEmbedding.embed_text() 実装
├── [P2-4] GeminiEmbedding.embed_texts() 実装
├── [P2-5] config.py にGeminiConfig追加
├── [P2-6] config.yml にGemini設定追加
├── [P2-7] 単体テスト実施・修正
└── [P2-8] Gemini API動作検証

Phase 3: 既存コード移行
├── [P3-1] helper_api.py 修正（プロバイダー切り替え対応）
├── [P3-2] celery_tasks.py 修正（Q/A生成をGemini対応）
├── [P3-3] a02_make_qa_para.py 修正
├── [P3-4] a02_make_qa_single.py 修正
├── [P3-5] helper_rag_qa.py 修正（SemanticCoverage）
├── [P3-6] qdrant_client_wrapper.py 修正（Embedding切り替え）
├── [P3-7] services/qdrant_service.py 修正
├── [P3-8] services/qa_service.py 修正
├── [P3-9] a50_rag_search_local_qdrant.py 修正
├── [P3-10] a42_qdrant_registration.py 修正
├── [P3-11] ui/pages/qdrant_search_page.py 修正
└── [P3-12] 統合テスト実施

Phase 4: Qdrantコレクション再構築
├── [P4-1] 既存コレクションバックアップ
├── [P4-2] migration_qdrant.py スクリプト作成
├── [P4-3] テストデータで再Embedding検証
├── [P4-4] 全コレクション再構築実行
└── [P4-5] 検索精度検証

Phase 5: 検証・評価
├── [P5-1] Q/A生成品質評価
├── [P5-2] Embedding精度比較
├── [P5-3] レイテンシ測定
├── [P5-4] コスト比較
├── [P5-5] 日本語性能評価
└── [P5-6] 最終レポート作成
```

---

### 10.2 タスク詳細

#### Phase 0: 環境準備

| タスクID | タスク名 | 詳細 | 成果物 | 依存 |
|----------|----------|------|--------|------|
| P0-1 | google-genai インストール | `pip install google-genai` | requirements.txt更新 | - |
| P0-2 | APIキー設定 | Google AI Studio でキー取得、.env設定 | .env | - |
| P0-3 | 動作確認 | サンプルスクリプトでAPI疎通確認 | test_gemini_api.py | P0-1, P0-2 |
| P0-4 | requirements.txt更新 | google-genai追加 | requirements.txt | P0-1 |

#### Phase 1: 抽象化レイヤー設計・実装

| タスクID | タスク名 | 詳細 | 成果物 | 依存 |
|----------|----------|------|--------|------|
| P1-1 | LLMClient抽象クラス | ABC継承、generate_content/generate_structured定義 | helper_llm.py | P0-3 |
| P1-2 | EmbeddingClient抽象クラス | ABC継承、embed_text/embed_texts定義 | helper_embedding.py | P0-3 |
| P1-3 | OpenAIClient実装 | 既存helper_api.pyの機能をラップ | helper_llm.py | P1-1 |
| P1-4 | OpenAIEmbedding実装 | 既存Embedding機能をラップ | helper_embedding.py | P1-2 |
| P1-5 | LLMテスト | OpenAIClientの単体テスト | tests/test_helper_llm.py | P1-3 |
| P1-6 | Embeddingテスト | OpenAIEmbeddingの単体テスト | tests/test_helper_embedding.py | P1-4 |

#### Phase 2: Gemini APIクライアント実装

| タスクID | タスク名 | 詳細 | 成果物 | 依存 |
|----------|----------|------|--------|------|
| P2-1 | generate_content実装 | テキスト生成、thinking_level対応 | helper_llm.py | P1-1 |
| P2-2 | generate_structured実装 | JSON schema、Pydantic連携 | helper_llm.py | P2-1 |
| P2-3 | embed_text実装 | 単一テキスト、3072次元指定 | helper_embedding.py | P1-2 |
| P2-4 | embed_texts実装 | バッチ処理、レート制限対応 | helper_embedding.py | P2-3 |
| P2-5 | GeminiConfig追加 | config.pyにGemini設定クラス追加 | config.py | P2-1 |
| P2-6 | config.yml更新 | Gemini設定セクション追加 | config.yml | P2-5 |
| P2-7 | 単体テスト | Geminiクライアントのテスト | tests/ | P2-4 |
| P2-8 | 動作検証 | 実際のAPIで検証 | 検証レポート | P2-7 |

#### Phase 3: 既存コード移行

| タスクID | タスク名 | 詳細 | 影響ファイル | 依存 |
|----------|----------|------|--------------|------|
| P3-1 | helper_api.py | create_llm_client()ファクトリ追加 | helper_api.py | P2-8 |
| P3-2 | celery_tasks.py | Q/A生成をGemini構造化出力に | celery_tasks.py | P3-1 |
| P3-3 | a02_make_qa_para.py | 並列Q/A生成のGemini対応 | a02_make_qa_para.py | P3-2 |
| P3-4 | a02_make_qa_single.py | 単一Q/A生成のGemini対応 | a02_make_qa_single.py | P3-2 |
| P3-5 | helper_rag_qa.py | SemanticCoverage Embedding切り替え | helper_rag_qa.py | P2-4 |
| P3-6 | qdrant_client_wrapper.py | embed_texts()をGemini対応 | qdrant_client_wrapper.py | P2-4 |
| P3-7 | qdrant_service.py | 検索用Embedding切り替え | services/qdrant_service.py | P3-6 |
| P3-8 | qa_service.py | Q/Aサービス修正 | services/qa_service.py | P3-2 |
| P3-9 | 検索UI | Streamlit検索ページ修正 | a50_rag_search_local_qdrant.py | P3-7 |
| P3-10 | 登録スクリプト | Qdrant登録のGemini対応 | a42_qdrant_registration.py | P3-6 |
| P3-11 | 検索ページUI | UIコンポーネント修正 | ui/pages/qdrant_search_page.py | P3-7 |
| P3-12 | 統合テスト | 全体フロー検証 | tests/ | P3-11 |

#### Phase 4: Qdrantコレクション再構築

| タスクID | タスク名 | 詳細 | 成果物 | 依存 |
|----------|----------|------|--------|------|
| P4-1 | バックアップ | 既存コレクションのスナップショット | backup/ | P3-12 |
| P4-2 | 移行スクリプト | Gemini Embeddingで再構築 | migration_qdrant.py | P4-1 |
| P4-3 | テスト検証 | 小規模データで検証 | 検証レポート | P4-2 |
| P4-4 | 本番実行 | 全コレクション再構築 | Qdrant更新 | P4-3 |
| P4-5 | 精度検証 | 検索精度の比較検証 | 検証レポート | P4-4 |

#### Phase 5: 検証・評価

| タスクID | タスク名 | 詳細 | 成果物 | 依存 |
|----------|----------|------|--------|------|
| P5-1 | Q/A品質評価 | 生成品質の人手・自動評価 | 評価レポート | P4-5 |
| P5-2 | Embedding精度 | コサイン類似度比較 | 比較データ | P4-5 |
| P5-3 | レイテンシ測定 | 応答時間の計測・比較 | 性能データ | P4-5 |
| P5-4 | コスト比較 | API費用の比較 | コストレポート | P5-3 |
| P5-5 | 日本語評価 | 日本語テストセットでの評価 | 評価レポート | P5-1 |
| P5-6 | 最終レポート | 総合評価レポート作成 | doc/migration_report.md | P5-5 |

---

### 10.3 依存関係図

```
Phase 0 (環境準備)
    │
    ├── P0-1 google-genai インストール
    │     │
    │     └── P0-4 requirements.txt更新
    │
    ├── P0-2 APIキー設定
    │
    └── P0-3 動作確認 ─────────────────────────────────┐
                                                        │
Phase 1 (抽象化レイヤー)                                │
    │                                                   │
    ├── P1-1 LLMClient ←────────────────────────────────┤
    │     │                                             │
    │     └── P1-3 OpenAIClient                         │
    │           │                                       │
    │           └── P1-5 テスト                         │
    │                                                   │
    └── P1-2 EmbeddingClient ←──────────────────────────┘
          │
          └── P1-4 OpenAIEmbedding
                │
                └── P1-6 テスト

Phase 2 (Geminiクライアント)
    │
    ├── P2-1 generate_content
    │     │
    │     └── P2-2 generate_structured
    │
    ├── P2-3 embed_text
    │     │
    │     └── P2-4 embed_texts
    │
    ├── P2-5 GeminiConfig
    │     │
    │     └── P2-6 config.yml
    │
    └── P2-7 テスト
          │
          └── P2-8 動作検証 ───────────────────────────┐
                                                        │
Phase 3 (既存コード移行)                                │
    │                                                   │
    ├── P3-1 helper_api.py ←────────────────────────────┤
    │     │                                             │
    │     └── P3-2 celery_tasks.py                      │
    │           │                                       │
    │           ├── P3-3 a02_make_qa_para.py            │
    │           ├── P3-4 a02_make_qa_single.py          │
    │           └── P3-8 qa_service.py                  │
    │                                                   │
    └── P3-5 helper_rag_qa.py ←─────────────────────────┘
          │
          └── P3-6 qdrant_client_wrapper.py
                │
                ├── P3-7 qdrant_service.py
                │     │
                │     └── P3-11 UI
                │
                ├── P3-9 検索UI
                │
                └── P3-10 登録スクリプト
                      │
                      └── P3-12 統合テスト ─────────────┐
                                                        │
Phase 4 (Qdrant再構築)                                  │
    │                                                   │
    └── P4-1 → P4-2 → P4-3 → P4-4 → P4-5 ←─────────────┘
                                      │
Phase 5 (検証・評価)                  │
    │                                 │
    └── P5-1 → P5-2 → P5-3 → P5-4 → P5-5 → P5-6
```

---

### 10.4 優先順位とクリティカルパス

**クリティカルパス（最長経路）:**
```
P0-1 → P0-3 → P1-1 → P2-1 → P2-2 → P2-8 → P3-1 → P3-2 → P3-12 → P4-1 → P4-5 → P5-6
```

**優先度マトリクス:**

| 優先度 | タスク群 | 理由 |
|--------|----------|------|
| **最高** | P0-1〜P0-3 | 全ての前提条件 |
| **高** | P1-1, P1-2, P2-1〜P2-4 | コア機能の実装 |
| **中** | P3-1〜P3-6 | 主要ファイルの移行 |
| **低** | P3-7〜P3-11 | 周辺ファイルの移行 |
| **最低** | P5-1〜P5-6 | 検証・評価 |

---

### 10.5 リスクと対策

| リスク | 影響度 | 発生確率 | 対策 |
|--------|--------|----------|------|
| Gemini API レート制限 | 高 | 中 | バッチ処理にスリープ挿入、Batch API活用 |
| Embedding次元不一致 | 高 | 低 | 3072次元で統一、Qdrantコレクション再構築、事前検証 |
| 構造化出力のスキーマエラー | 中 | 中 | JSONスキーマの厳密な検証、フォールバック実装 |
| 日本語性能の劣化 | 中 | 低 | 事前に日本語テストセットで評価 |
| Celery連携の問題 | 中 | 低 | ワーカー再起動、タイムアウト調整 |
| コスト超過 | 低 | 中 | 開発中は小規模データで検証、料金アラート設定 |

---

### 10.6 マイルストーン

| マイルストーン | 完了条件 | 目標日 |
|----------------|----------|--------|
| **M1: 環境準備完了** | P0-1〜P0-4完了、Gemini API疎通確認 | Phase 0終了時 |
| **M2: 抽象化レイヤー完成** | P1-1〜P1-6完了、OpenAI互換動作確認 | Phase 1終了時 |
| **M3: Geminiクライアント完成** | P2-1〜P2-8完了、Gemini単体動作確認 | Phase 2終了時 |
| **M4: コード移行完了** | P3-1〜P3-12完了、統合テストパス | Phase 3終了時 |
| **M5: データ移行完了** | P4-1〜P4-5完了、Qdrant検索動作確認 | Phase 4終了時 |
| **M6: 移行完了** | P5-1〜P5-6完了、最終レポート提出 | Phase 5終了時 |

---

### 10.7 初期着手タスク（推奨）

**今すぐ開始可能なタスク:**

1. **P0-1: google-genai インストール**
   ```bash
   pip install google-genai
   ```

2. **P0-2: APIキー取得**
   - Google AI Studio (https://aistudio.google.com/) でAPIキー取得
   - `.env` に `GOOGLE_API_KEY=AIza...` を追加

3. **P0-3: 動作確認スクリプト作成**
   ```python
   # test_gemini_api.py
   from google import genai
   import os
   from dotenv import load_dotenv

   load_dotenv()

   client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

   # テキスト生成テスト
   response = client.models.generate_content(
       model="gemini-3-pro-preview",
       contents="Hello, how are you?",
       config={"thinking_level": "low"}
   )
   print("Text Generation:", response.text)

   # Embeddingテスト（3072次元: Gemini 3最大精度）
   embed_response = client.models.embed_content(
       model="gemini-embedding-001",
       contents="Hello world",
       config={"output_dimensionality": 3072}
   )
   print("Embedding dims:", len(embed_response.embeddings[0].values))
   ```

---

作成者: RAG Q/A Development Team

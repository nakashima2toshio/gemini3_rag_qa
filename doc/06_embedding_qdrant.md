# Embedding・Qdrant登録・検索ドキュメント

本ドキュメントでは、Q/AペアデータのEmbedding（ベクトル化）、Qdrantへの登録、および検索処理について解説する。

## 目次

- [1. 概要](#1-概要)
  - [1.1 本ドキュメントの位置づけ](#11-本ドキュメントの位置づけ)
  - [1.2 関連ファイル一覧](#12-関連ファイル一覧)
  - [1.3 データフロー図](#13-データフロー図)
- [2. Embedding（ベクトル化）](#2-embeddingベクトル化)
  - [2.1 使用モデルと設定](#21-使用モデルと設定)
  - [2.2 embed_texts_for_qdrant() 関数の処理フロー](#22-embed_texts_for_qdrant-関数の処理フロー)
  - [2.3 バッチ処理とトークン制限](#23-バッチ処理とトークン制限)
  - [2.4 埋め込み入力の構築（question + answer）](#24-埋め込み入力の構築question--answer)
  - [2.5 空文字列・エッジケース処理](#25-空文字列エッジケース処理)
- [3. Qdrant登録](#3-qdrant登録)
  - [3.1 コレクション設計](#31-コレクション設計)
  - [3.2 ベクトルパラメータ設定](#32-ベクトルパラメータ設定)
  - [3.3 ポイント構造（PointStruct）](#33-ポイント構造pointstruct)
  - [3.4 ペイロードスキーマ](#34-ペイロードスキーマ)
  - [3.5 バッチアップサート処理](#35-バッチアップサート処理)
  - [3.6 ペイロードインデックス](#36-ペイロードインデックス)
- [4. コレクション統合](#4-コレクション統合)
  - [4.1 統合機能の概要](#41-統合機能の概要)
  - [4.2 scroll_all_points_with_vectors()](#42-scroll_all_points_with_vectors)
  - [4.3 merge_collections()](#43-merge_collections)
  - [4.4 統合時のペイロード拡張](#44-統合時のペイロード拡張)
- [5. 検索処理](#5-検索処理)
  - [5.1 クエリのベクトル化](#51-クエリのベクトル化)
  - [5.2 コサイン類似度検索](#52-コサイン類似度検索)
  - [5.3 検索結果の構造](#53-検索結果の構造)
  - [5.4 AI応答生成との連携](#54-ai応答生成との連携)
- [6. 運用・設定](#6-運用設定)
  - [6.1 Qdrant設定（QDRANT_CONFIG）](#61-qdrant設定qdrant_config)
  - [6.2 コレクション管理（CRUD）](#62-コレクション管理crud)
  - [6.3 ヘルスチェック](#63-ヘルスチェック)
  - [6.4 統計情報取得](#64-統計情報取得)
- [7. 付録](#7-付録)
  - [7.1 コレクション名とCSVファイルの対応表](#71-コレクション名とcsvファイルの対応表)
  - [7.2 コード参照一覧](#72-コード参照一覧)

---

## 1. 概要

### 1.1 本ドキュメントの位置づけ

本ドキュメントは「ベクトル化・Qdrant登録・検索」に焦点を当てる。

| ドキュメント | 焦点 | 内容 |
|-------------|------|------|
| `doc/03_chunk.md` | チャンク分割技術 | SemanticCoverage、文分割、MeCab |
| `doc/04_prompt.md` | プロンプト設計 | 2段階構造、言語別対応、動的調整 |
| `doc/05_qa_pair.md` | 実行・処理フロー | 並列処理、Celery、出力、カバレージ |
| `doc/06_embedding_qdrant.md`（本書） | ベクトル化・DB登録・検索 | Embedding、Qdrant、類似度検索、コレクション統合 |

### 1.2 関連ファイル一覧

| ファイル | 役割 |
|---------|------|
| `services/qdrant_service.py` | Qdrant操作サービス層（メイン実装） |
| `ui/pages/qdrant_registration_page.py` | 登録UI（CSV登録・コレクション統合） |
| `ui/pages/qdrant_search_page.py` | 検索UI |
| `ui/pages/qdrant_show_page.py` | コレクション表示UI |
| `a30_qdrant_registration.py` | CLI登録スクリプト |
| `a50_rag_search_local_qdrant.py` | CLI検索スクリプト |

### 1.3 データフロー図

```
[Q/Aペアデータ]
    │
    │ qa_output/*.csv (question, answer列)
    ▼
[1. データ読み込み]  ←── load_csv_for_qdrant()
    │
    │ DataFrame (question, answer)
    ▼
[2. 埋め込み入力構築]  ←── build_inputs_for_embedding()
    │
    │ List[str] ("question\nanswer" or "question")
    ▼
[3. Embedding生成]  ←── embed_texts_for_qdrant()
    │
    │ OpenAI API (text-embedding-3-small)
    │ List[List[float]] (1536次元ベクトル)
    ▼
[4. ポイント構築]  ←── build_points_for_qdrant()
    │
    │ List[PointStruct] (id, vector, payload)
    ▼
[5. Qdrant登録]  ←── upsert_points_to_qdrant()
    │
    │ コレクションにバッチアップサート
    ▼
[Qdrant Vector Database]
    │
    ├──[6a. 検索クエリ]  ←── embed_query_for_search() + client.search()
    │       │
    │       │ コサイン類似度検索
    │       ▼
    │   [検索結果] (score, question, answer, source)
    │
    └──[6b. コレクション統合]  ←── merge_collections()
            │
            │ 複数コレクションを1つに統合
            ▼
        [統合コレクション]
```

---

## 2. Embedding（ベクトル化）

### 2.1 使用モデルと設定

本システムでは、OpenAIの`text-embedding-3-small`モデルを使用する。

```python
# services/qdrant_service.py - COLLECTION_EMBEDDINGS_SEARCH
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_VECTOR_SIZE = 1536
```

| 項目 | 値 | 説明 |
|-----|-----|-----|
| モデル | text-embedding-3-small | OpenAI第3世代埋め込みモデル |
| 次元数 | 1536 | 高精度版（デフォルト） |
| 代替次元数 | 384 | 高速版（dimensionsパラメータで指定可能） |
| エンコーディング | cl100k_base | トークンカウント用（tiktoken） |
| 最大トークン/リクエスト | 8000 | バッチ処理の上限 |

**text-embedding-3シリーズの特徴:**

- 可変次元数サポート（384〜3072）
- コサイン類似度に最適化
- 多言語対応（日本語含む）
- text-embedding-ada-002の後継

### 2.2 embed_texts_for_qdrant() 関数の処理フロー

```python
# services/qdrant_service.py:469-531
def embed_texts_for_qdrant(
    texts: List[str],
    model: str,
    batch_size: int = 128
) -> List[List[float]]:
```

**処理フロー図:**

```
[入力: texts (List[str])]
    │
    ▼
┌─────────────────────────────────────┐
│ 1. 空文字列フィルタリング           │
│    valid_texts = []                 │
│    valid_indices = []               │
│    for i, text in enumerate(texts): │
│        if text and text.strip():    │
│            valid_texts.append(text) │
│            valid_indices.append(i)  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. トークンカウント（tiktoken）      │
│    enc = tiktoken.get_encoding(     │
│        "cl100k_base"                │
│    )                                │
│    text_tokens = len(enc.encode(t)) │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. バッチ分割                        │
│    MAX_TOKENS_PER_REQUEST = 8000    │
│    current_tokens + text_tokens     │
│        > MAX_TOKENS → 新バッチ開始   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. OpenAI API呼び出し               │
│    resp = client.embeddings.create( │
│        model=model,                 │
│        input=current_batch          │
│    )                                │
│    vecs = [d.embedding for d       │
│            in resp.data]            │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. ベクトル再配置                    │
│    元インデックスに合わせて並べ替え   │
│    空文字列位置には                  │
│    [0.0] * 1536 のダミーベクトル     │
└─────────────────────────────────────┘
    │
    ▼
[出力: List[List[float]] (1536次元 × N件)]
```

<details>
<summary>📝 embed_texts_for_qdrant() 完全実装コード</summary>

```python
# services/qdrant_service.py:469-531

def embed_texts_for_qdrant(
    texts: List[str], model: str, batch_size: int = 128
) -> List[List[float]]:
    """テキストをバッチ処理でEmbeddingに変換"""
    enc = tiktoken.get_encoding("cl100k_base")
    client = OpenAI()

    MAX_TOKENS_PER_REQUEST = 8000

    # 空文字列・空白のみの文字列を除外
    valid_texts = []
    valid_indices = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_texts.append(text)
            valid_indices.append(i)

    if not valid_texts:
        logger.warning("全てのテキストが空文字列です。ダミーベクトルを返します。")
        return [[0.0] * 1536] * len(texts)

    # 有効なテキストのみで埋め込み生成
    valid_vecs: List[List[float]] = []
    current_batch = []
    current_tokens = 0
    batch_count = 0

    for i, text in enumerate(valid_texts):
        text_tokens = len(enc.encode(text))

        if text_tokens > MAX_TOKENS_PER_REQUEST:
            raise ValueError(
                f"Single text at index {valid_indices[i]} has {text_tokens} tokens, "
                f"which exceeds MAX_TOKENS_PER_REQUEST ({MAX_TOKENS_PER_REQUEST}). "
            )

        if current_tokens + text_tokens > MAX_TOKENS_PER_REQUEST:
            if current_batch:
                batch_count += 1
                resp = client.embeddings.create(model=model, input=current_batch)
                valid_vecs.extend([d.embedding for d in resp.data])
                current_batch = []
                current_tokens = 0

        current_batch.append(text)
        current_tokens += text_tokens

    if current_batch:
        batch_count += 1
        resp = client.embeddings.create(model=model, input=current_batch)
        valid_vecs.extend([d.embedding for d in resp.data])

    # 元のインデックスに合わせてベクトルを再配置
    vecs: List[List[float]] = []
    valid_vec_idx = 0
    for i in range(len(texts)):
        if i in valid_indices:
            vecs.append(valid_vecs[valid_vec_idx])
            valid_vec_idx += 1
        else:
            vecs.append([0.0] * 1536)

    return vecs
```

**ポイント:**
- `tiktoken.get_encoding("cl100k_base")`: OpenAI埋め込みモデル用トークナイザー
- `MAX_TOKENS_PER_REQUEST = 8000`: API制限に対応したバッチ分割
- 空文字列位置に `[0.0] * 1536` ダミーベクトル配置（インデックス整合性維持）
- 動的バッチサイズ: トークン数に応じて最適なバッチを構築

</details>

### 2.3 バッチ処理とトークン制限

OpenAI Embedding APIにはリクエストあたりのトークン制限がある。

```python
MAX_TOKENS_PER_REQUEST = 8000
```

**バッチ分割ロジック:**

```python
for i, text in enumerate(valid_texts):
    text_tokens = len(enc.encode(text))

    # 単一テキストが制限を超える場合はエラー
    if text_tokens > MAX_TOKENS_PER_REQUEST:
        raise ValueError(f"Single text has {text_tokens} tokens")

    # 累積トークンが制限を超えたら新バッチ開始
    if current_tokens + text_tokens > MAX_TOKENS_PER_REQUEST:
        # 現在のバッチを処理
        resp = client.embeddings.create(model=model, input=current_batch)
        valid_vecs.extend([d.embedding for d in resp.data])
        current_batch = []
        current_tokens = 0

    current_batch.append(text)
    current_tokens += text_tokens
```

**バッチ処理の効率:**

| シナリオ | 平均テキスト長 | バッチサイズ目安 | API呼び出し回数（1000件） |
|---------|--------------|----------------|------------------------|
| 短文Q/A | 100トークン | 80件/バッチ | 約13回 |
| 中文Q/A | 300トークン | 26件/バッチ | 約39回 |
| 長文Q/A | 500トークン | 16件/バッチ | 約63回 |

### 2.4 埋め込み入力の構築（question + answer）

Q/Aペアをベクトル化する際、questionのみか、question+answerかを選択できる。

```python
# services/qdrant_service.py:462-466
def build_inputs_for_embedding(df: pd.DataFrame, include_answer: bool) -> List[str]:
    """埋め込み用入力テキストを構築"""
    if include_answer:
        # Q+Aを連結（改行区切り）
        return (df["question"].astype(str) + "\n" + df["answer"].astype(str)).tolist()
    # questionのみ
    return df["question"].astype(str).tolist()
```

**include_answerの選択基準:**

| オプション | 用途 | メリット | デメリット |
|-----------|------|---------|-----------|
| `include_answer=True` | 回答内容での検索 | 検索精度向上、文脈理解 | ベクトルサイズ増加 |
| `include_answer=False` | 質問マッチング | 高速、質問の類似度重視 | 回答内容を考慮しない |

**推奨: `include_answer=True`**（デフォルト）
- ユーザーの質問が回答内容に近い場合に有効
- RAGシステムでは回答の関連性が重要

### 2.5 空文字列・エッジケース処理

```python
# 空文字列の処理
valid_texts = []
valid_indices = []
for i, text in enumerate(texts):
    if text and text.strip():  # 空文字列・空白のみを除外
        valid_texts.append(text)
        valid_indices.append(i)

# 全て空文字列の場合
if not valid_texts:
    logger.warning("全てのテキストが空文字列です。ダミーベクトルを返します。")
    return [[0.0] * 1536] * len(texts)

# 結果の再配置（空文字列位置にはダミーベクトル）
vecs: List[List[float]] = []
valid_vec_idx = 0
for i in range(len(texts)):
    if i in valid_indices:
        vecs.append(valid_vecs[valid_vec_idx])
        valid_vec_idx += 1
    else:
        vecs.append([0.0] * 1536)  # ダミーベクトル
```

**ダミーベクトル `[0.0] * 1536` の意味:**
- コサイン類似度計算で他の全てのベクトルと類似度0になる
- 検索結果に現れにくい（最低スコア）
- データの整合性を保持（インデックス対応）

---

## 3. Qdrant登録

### 3.1 コレクション設計

本システムでは、データセット・生成方式ごとにコレクションを分離する。

**コレクション命名規則:**

```
qa_{dataset}_{method}
```

| コレクション名 | データセット | 生成方式 |
|--------------|------------|---------|
| qa_cc_news_a02_llm | CC News | LLM生成（a02） |
| qa_cc_news_a03_rule | CC News | ルールベース（a03） |
| qa_cc_news_a10_hybrid | CC News | ハイブリッド（a10） |
| qa_livedoor_a02_20_llm | Livedoor | LLM生成（a02） |
| qa_livedoor_a03_rule | Livedoor | ルールベース（a03） |
| qa_livedoor_a10_hybrid | Livedoor | ハイブリッド（a10） |
| qa_corpus | カスタム | 汎用 |
| integration_{name} | 統合 | 複数コレクション統合 |

### 3.2 ベクトルパラメータ設定

```python
# services/qdrant_service.py:534-562
def create_or_recreate_collection_for_qdrant(
    client: QdrantClient,
    name: str,
    recreate: bool = False,
    vector_size: int = 1536
):
    vectors_config = models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE  # コサイン類似度
    )

    if recreate:
        try:
            client.delete_collection(collection_name=name)
        except Exception:
            pass
        client.create_collection(
            collection_name=name,
            vectors_config=vectors_config
        )
    else:
        # 存在しない場合のみ作成
        try:
            client.get_collection(name)
        except Exception:
            client.create_collection(
                collection_name=name,
                vectors_config=vectors_config
            )
```

**ベクトルパラメータ:**

| パラメータ | 値 | 説明 |
|-----------|-----|-----|
| size | 1536 | text-embedding-3-smallの次元数 |
| distance | COSINE | コサイン類似度（-1〜1、正規化済み） |

**距離メトリクスの選択:**

| メトリクス | 範囲 | 用途 |
|-----------|------|------|
| COSINE | -1〜1 | テキスト類似度（推奨） |
| EUCLID | 0〜∞ | 絶対距離 |
| DOT | -∞〜∞ | 非正規化ベクトル |

### 3.3 ポイント構造（PointStruct）

Qdrantの各データポイントは以下の構造を持つ。

```python
# services/qdrant_service.py:565-589
def build_points_for_qdrant(
    df: pd.DataFrame,
    vectors: List[List[float]],
    domain: str,
    source_file: str
) -> List[models.PointStruct]:

    now_iso = datetime.now(timezone.utc).isoformat()
    points: List[models.PointStruct] = []

    for i, row in enumerate(df.itertuples(index=False)):
        payload = {
            "domain": domain,
            "question": getattr(row, "question"),
            "answer": getattr(row, "answer"),
            "source": os.path.basename(source_file),
            "created_at": now_iso,
            "schema": "qa:v1"
        }

        # IDの生成（64ビット正整数）
        pid = abs(hash(f"{domain}-{source_file}-{i}")) & 0x7FFFFFFFFFFFFFFF

        points.append(models.PointStruct(
            id=pid,
            vector=vectors[i],
            payload=payload
        ))

    return points
```

**PointStruct構造図:**

```
PointStruct
├── id: int (64ビット正整数)
│       hash("domain-source_file-index") & 0x7FFFFFFFFFFFFFFF
│
├── vector: List[float] (1536次元)
│       [0.0234, -0.1567, 0.0891, ...]
│
└── payload: Dict
        ├── domain: str        → "livedoor", "cc_news", "custom"
        ├── question: str      → "質問文テキスト"
        ├── answer: str        → "回答文テキスト"
        ├── source: str        → "a02_qa_pairs_livedoor.csv"
        ├── created_at: str    → "2025-11-28T10:30:00+00:00"
        └── schema: str        → "qa:v1"
```

<details>
<summary>📝 build_points_for_qdrant() 完全実装コード</summary>

```python
# services/qdrant_service.py:565-589

def build_points_for_qdrant(
    df: pd.DataFrame, vectors: List[List[float]], domain: str, source_file: str
) -> List[models.PointStruct]:
    """Qdrantポイントを構築"""
    n = len(df)
    if len(vectors) != n:
        raise ValueError(f"vectors length mismatch: df={n}, vecs={len(vectors)}")

    now_iso = datetime.now(timezone.utc).isoformat()
    points: List[models.PointStruct] = []

    for i, row in enumerate(df.itertuples(index=False)):
        payload = {
            "domain": domain,
            "question": getattr(row, "question"),
            "answer": getattr(row, "answer"),
            "source": os.path.basename(source_file),
            "created_at": now_iso,
            "schema": "qa:v1",
        }

        # 64ビット正整数ID生成（ハッシュ衝突回避）
        pid = abs(hash(f"{domain}-{source_file}-{i}")) & 0x7FFFFFFFFFFFFFFF
        points.append(models.PointStruct(id=pid, vector=vectors[i], payload=payload))

    return points
```

**ポイント:**
- `& 0x7FFFFFFFFFFFFFFF`: 64ビット正整数に変換（Qdrant要件）
- `datetime.now(timezone.utc).isoformat()`: UTC時刻でISO 8601形式
- `schema: "qa:v1"`: スキーマバージョン管理で将来の変更に対応
- `os.path.basename(source_file)`: ファイル名のみを保存（パス非依存）

</details>

### 3.4 ペイロードスキーマ

**基本スキーマ（qa:v1）:**

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|-----|
| domain | string | ✓ | データドメイン（livedoor, cc_news, custom） |
| question | string | ✓ | 質問文 |
| answer | string | ✓ | 回答文 |
| source | string | ✓ | ソースファイル名 |
| created_at | string | ✓ | 登録日時（ISO 8601） |
| schema | string | ✓ | スキーマバージョン（"qa:v1"） |

**統合時の追加フィールド:**

| フィールド | 型 | 説明 |
|-----------|-----|-----|
| _source_collection | string | 統合元コレクション名 |
| _original_id | int | 統合元での元ID |

### 3.5 バッチアップサート処理

```python
# services/qdrant_service.py:592-603
def upsert_points_to_qdrant(
    client: QdrantClient,
    collection: str,
    points: List[models.PointStruct],
    batch_size: int = 128
) -> int:
    """ポイントをQdrantにアップサート"""
    count = 0
    for chunk in batched(points, batch_size):
        client.upsert(collection_name=collection, points=chunk)
        count += len(chunk)
    return count
```

**バッチ分割ユーティリティ:**

```python
# services/qdrant_service.py:75-84
def batched(seq: Iterable, size: int):
    """イテラブルをバッチに分割"""
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
```

**バッチサイズの選択:**

| batch_size | メモリ使用量 | スループット | 推奨用途 |
|-----------|------------|------------|---------|
| 64 | 低 | 中 | メモリ制限環境 |
| 128 | 中 | 高 | **推奨（デフォルト）** |
| 256 | 高 | 最高 | 大規模データ |

**upsertの動作:**
- 同一IDが存在 → 上書き更新
- 新規ID → 新規挿入
- トランザクション的な動作（バッチ単位）

### 3.6 ペイロードインデックス

検索効率化のため、domainフィールドにインデックスを作成する。

```python
# create_or_recreate_collection_for_qdrant() 内
try:
    client.create_payload_index(
        name,
        field_name="domain",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
except Exception:
    pass  # 既存の場合はスキップ
```

**インデックスの効果:**

```python
# インデックスなし
client.search(collection_name="qa_corpus", query_vector=qvec, limit=5)
# → 全ポイントをスキャン

# インデックスあり + フィルタ
client.search(
    collection_name="qa_corpus",
    query_vector=qvec,
    query_filter=models.Filter(
        must=[models.FieldCondition(
            key="domain",
            match=models.MatchValue(value="livedoor")
        )]
    ),
    limit=5
)
# → domainインデックスで高速フィルタリング
```

---

## 4. コレクション統合

### 4.1 統合機能の概要

複数のコレクションを1つの新しいコレクションに統合する機能。

**ユースケース:**
- 複数データセットの統合検索
- テスト用コレクションの本番統合
- バックアップからの復元

**統合フロー図:**

```
[コレクションA]     [コレクションB]     [コレクションC]
    │                   │                   │
    │ scroll()          │ scroll()          │ scroll()
    ▼                   ▼                   ▼
[ポイント取得]      [ポイント取得]      [ポイント取得]
    │                   │                   │
    └───────────────────┴───────────────────┘
                        │
                        │ ID再生成 + payload拡張
                        ▼
              [統合ポイントリスト]
                        │
                        │ upsert()
                        ▼
              [統合コレクション]
              integration_{name}
```

### 4.2 scroll_all_points_with_vectors()

コレクションから全ポイント（ベクトル含む）を取得する。

```python
# services/qdrant_service.py:626-672
def scroll_all_points_with_vectors(
    client: QdrantClient,
    collection_name: str,
    batch_size: int = 100,
    progress_callback: Optional[callable] = None,
) -> List[models.Record]:
    """コレクションから全ポイント（ベクトル含む）を取得"""
    all_points = []
    offset = None

    # 総件数を取得
    collection_info = client.get_collection(collection_name)
    total_points = collection_info.points_count

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,  # ベクトルも取得
        )

        if not points:
            break

        all_points.extend(points)

        if progress_callback:
            progress_callback(len(all_points), total_points)

        if next_offset is None:
            break

        offset = next_offset

    return all_points
```

**パラメータ:**

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|---------|------|
| client | QdrantClient | - | Qdrantクライアント |
| collection_name | str | - | コレクション名 |
| batch_size | int | 100 | 1回のスクロールで取得する件数 |
| progress_callback | callable | None | 進捗コールバック (取得済み, 総件数) |

### 4.3 merge_collections()

複数コレクションを統合して新コレクションに登録する。

```python
# services/qdrant_service.py:675-779
def merge_collections(
    client: QdrantClient,
    source_collections: List[str],
    target_collection: str,
    recreate: bool = True,
    vector_size: int = 1536,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """複数コレクションを統合して新コレクションに登録"""
```

**パラメータ:**

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|---------|------|
| source_collections | List[str] | - | 統合元コレクション名リスト |
| target_collection | str | - | 統合先コレクション名 |
| recreate | bool | True | 既存コレクションを削除して再作成 |
| vector_size | int | 1536 | ベクトルサイズ |
| progress_callback | callable | None | 進捗コールバック (メッセージ, 現在値, 最大値) |

**戻り値:**

```python
{
    "source_collections": ["qa_livedoor_a02", "qa_cc_news_a02"],
    "target_collection": "integration_qa_livedoor_a02",
    "points_per_collection": {
        "qa_livedoor_a02": 1500,
        "qa_cc_news_a02": 2000
    },
    "total_points": 3500,
    "success": True,
    "error": None
}
```

<details>
<summary>📝 merge_collections() 完全実装コード</summary>

```python
# services/qdrant_service.py:675-779

def merge_collections(
    client: QdrantClient,
    source_collections: List[str],
    target_collection: str,
    recreate: bool = True,
    vector_size: int = 1536,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """複数コレクションを統合して新コレクションに登録"""
    result = {
        "source_collections": source_collections,
        "target_collection": target_collection,
        "points_per_collection": {},
        "total_points": 0,
        "success": False,
        "error": None,
    }

    try:
        # ステップ1: 統合先コレクションを作成
        if progress_callback:
            progress_callback(f"コレクション '{target_collection}' を作成中...", 0, 100)

        create_or_recreate_collection_for_qdrant(
            client, target_collection, recreate, vector_size
        )

        # ステップ2: 各コレクションからポイントを取得して統合
        all_points = []
        collection_count = len(source_collections)

        for idx, src_collection in enumerate(source_collections):
            if progress_callback:
                progress_callback(
                    f"コレクション '{src_collection}' からデータ取得中...",
                    int((idx / collection_count) * 50),
                    100,
                )

            # ポイントを取得（ベクトル含む）
            points = scroll_all_points_with_vectors(client, src_collection)
            result["points_per_collection"][src_collection] = len(points)

            # ポイントIDを再生成（重複回避）+ ソース情報追加
            for i, point in enumerate(points):
                payload = dict(point.payload) if point.payload else {}
                payload["_source_collection"] = src_collection
                payload["_original_id"] = point.id

                new_id = abs(
                    hash(f"{target_collection}-{src_collection}-{point.id}-{i}")
                ) & 0x7FFFFFFFFFFFFFFF

                all_points.append(
                    models.PointStruct(
                        id=new_id,
                        vector=point.vector,
                        payload=payload,
                    )
                )

        result["total_points"] = len(all_points)

        # ステップ3: 統合先コレクションにアップサート
        if progress_callback:
            progress_callback("統合データをアップサート中...", 50, 100)

        if all_points:
            upserted = 0
            batch_size = 128
            for chunk in batched(all_points, batch_size):
                client.upsert(collection_name=target_collection, points=chunk)
                upserted += len(chunk)
                if progress_callback:
                    progress = 50 + int((upserted / len(all_points)) * 50)
                    progress_callback(
                        f"アップサート中... ({upserted}/{len(all_points)})",
                        progress,
                        100,
                    )

        result["success"] = True

        if progress_callback:
            progress_callback("統合完了", 100, 100)

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"コレクション統合エラー: {e}")

    return result
```

**ポイント:**
- `scroll_all_points_with_vectors()`: ベクトルを含む全ポイントを取得
- `_source_collection` / `_original_id`: トレーサビリティ確保（どのコレクション由来か追跡可能）
- 新IDハッシュ生成: `{target}-{src}-{original_id}-{index}` で衝突回避
- `progress_callback`: 3段階の進捗通知（コレクション作成→データ取得→アップサート）
- バッチアップサート: 128件ずつ分割してメモリ効率化

</details>

### 4.4 統合時のペイロード拡張

統合時、元のコレクション情報を保持するためペイロードを拡張する。

```python
# ポイントIDを再生成（重複回避）
for i, point in enumerate(points):
    # 元のpayloadにソースコレクション情報を追加
    payload = dict(point.payload) if point.payload else {}
    payload["_source_collection"] = src_collection
    payload["_original_id"] = point.id

    # 新しいIDを生成
    new_id = abs(
        hash(f"{target_collection}-{src_collection}-{point.id}-{i}")
    ) & 0x7FFFFFFFFFFFFFFF

    all_points.append(
        models.PointStruct(
            id=new_id,
            vector=point.vector,
            payload=payload,
        )
    )
```

**統合後のペイロード例:**

```python
{
    "domain": "livedoor",
    "question": "質問文",
    "answer": "回答文",
    "source": "a02_qa_pairs_livedoor.csv",
    "created_at": "2025-11-28T10:30:00+00:00",
    "schema": "qa:v1",
    "_source_collection": "qa_livedoor_a02",  # 追加
    "_original_id": 1234567890123456789       # 追加
}
```

---

## 5. 検索処理

### 5.1 クエリのベクトル化

検索クエリを埋め込みベクトルに変換する。

```python
# services/qdrant_service.py:610-619
def embed_query_for_search(
    query: str,
    model: str = "text-embedding-3-small",
    dims: Optional[int] = None
) -> List[float]:
    """検索クエリをベクトル化"""
    client = OpenAI()
    kwargs = {"model": model, "input": query}
    if dims:
        kwargs["dimensions"] = dims
    resp = client.embeddings.create(**kwargs)
    return resp.data[0].embedding
```

**embed_texts_for_qdrant() vs embed_query_for_search() の違い:**

| 関数 | 用途 | バッチ | 空文字処理 |
|-----|------|-------|----------|
| embed_texts_for_qdrant() | 大量データの登録時 | ✓ | ✓ |
| embed_query_for_search() | 検索クエリ1件 | × | × |

### 5.2 コサイン類似度検索

```python
# Qdrant検索の基本形
hits = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=limit
)

results = []
for h in hits:
    results.append({
        "score": h.score,
        "id": h.id,
        "payload": h.payload
    })
```

**コサイン類似度スコア:**

| スコア | 意味 | 解釈 |
|-------|------|------|
| 1.0 | 完全一致 | 同一または極めて類似 |
| 0.8〜0.99 | 高類似度 | 関連性が高い |
| 0.5〜0.79 | 中類似度 | ある程度関連 |
| 0.0〜0.49 | 低類似度 | 関連性が低い |
| < 0 | 負の相関 | 反対の意味 |

### 5.3 検索結果の構造

```python
# 検索結果の例
[
    {
        "score": 0.8923,
        "id": 1234567890123456789,
        "payload": {
            "domain": "livedoor",
            "question": "Pythonでリストをソートする方法は？",
            "answer": "sorted()関数またはlist.sort()メソッドを使用します...",
            "source": "a02_qa_pairs_livedoor.csv",
            "created_at": "2025-11-28T10:30:00+00:00",
            "schema": "qa:v1"
        }
    },
    {
        "score": 0.8456,
        "id": 9876543210987654321,
        "payload": {...}
    },
    ...
]
```

### 5.4 AI応答生成との連携

検索結果を基にAI応答を生成する（RAGパターン）。

```python
# ui/pages/qdrant_search_page.py
# 最高スコアの検索結果を使用
best_hit = hits[0]
question = best_hit.payload.get("question", "")
answer = best_hit.payload.get("answer", "")

# プロンプト構築
qa_prompt = (
    "以下の検索結果とユーザーの質問を踏まえて、"
    "日本語で簡潔かつ正確に回答してください。\n\n"
    f"ユーザーの質問:\n{query}\n\n"
    f"検索結果のスコア: {best_hit.score:.4f}\n"
    f"検索結果の質問: {question}\n"
    f"検索結果の回答: {answer}\n"
)

# OpenAI API呼び出し
oai_client = OpenAI()
oai_resp = oai_client.responses.create(
    model="gpt-4o-mini",
    input=qa_prompt
)
generated_answer = oai_resp.output_text
```

<details>
<summary>📝 検索・AI応答生成 完全実装コード</summary>

```python
# ui/pages/qdrant_search_page.py:160-267

# 検索実行
if do_search and query.strip():
    try:
        client = QdrantClient(url=qdrant_url)

        # コレクションに対応した埋め込み設定を取得
        collection_config = COLLECTION_EMBEDDINGS_SEARCH.get(
            collection, {"model": "text-embedding-3-small", "dims": 1536}
        )
        embedding_model = collection_config["model"]
        embedding_dims = collection_config.get("dims")

        # クエリを埋め込みベクトルに変換
        with st.spinner("埋め込みベクトルを生成中..."):
            qvec = embed_query_for_search(query, embedding_model, embedding_dims)

        # Qdrantで検索
        with st.spinner("検索中..."):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                hits = client.search(
                    collection_name=collection, query_vector=qvec, limit=topk
                )

        # 検索結果をDataFrameに変換
        rows = []
        for h in hits:
            row_data = {
                "スコア": f"{h.score:.4f}",
                "質問": h.payload.get("question", "N/A") if h.payload else "N/A",
                "回答": h.payload.get("answer", "N/A") if h.payload else "N/A",
                "ソース": h.payload.get("source", "N/A") if h.payload else "N/A",
            }
            rows.append(row_data)

        df_results = pd.DataFrame(rows)
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        # 最高スコアの結果でAI応答生成
        if hits:
            best_hit = hits[0]
            question = best_hit.payload.get("question", "")
            answer = best_hit.payload.get("answer", "")

            # AI応答生成プロンプト構築
            qa_prompt = (
                "以下の検索結果とユーザーの質問を踏まえて、日本語で簡潔かつ正確に回答してください。\n\n"
                f"ユーザーの質問:\n{query}\n\n"
                f"検索結果のスコア: {best_hit.score:.4f}\n"
                f"検索結果の質問: {question}\n"
                f"検索結果の回答: {answer}\n"
            )

            # OpenAI API呼び出し（responses.create）
            with st.spinner("AIが回答を生成中..."):
                oai_client = OpenAI()
                oai_resp = oai_client.responses.create(
                    model="gpt-4o-mini", input=qa_prompt
                )
                generated_answer = (
                    getattr(oai_resp, "output_text", None) or ""
                )

            if generated_answer.strip():
                st.markdown("**AI応答:**")
                st.write(generated_answer)

    except Exception as e:
        st.error(f"❌ エラーが発生しました: {str(e)}")
```

**ポイント:**
- `embed_query_for_search()`: クエリを1536次元ベクトルに変換
- `client.search()`: コサイン類似度でTop-K検索
- `responses.create()`: OpenAI Responses APIでAI応答生成
- `getattr(oai_resp, "output_text", None)`: 応答テキストの安全な取得
- エラーハンドリング: 接続エラー・コレクション未検出を個別処理

</details>

**RAGフロー図:**

```
[ユーザー質問]
    │
    ▼
[1. クエリベクトル化] ← embed_query_for_search()
    │
    ▼
[2. Qdrant検索] ← client.search()
    │
    │ Top-K結果 (question, answer, score)
    ▼
[3. コンテキスト構築]
    │
    │ プロンプト = 質問 + 検索結果
    ▼
[4. AI応答生成] ← OpenAI GPT-4o-mini
    │
    ▼
[最終回答]
```

---

## 6. 運用・設定

### 6.1 Qdrant設定（QDRANT_CONFIG）

```python
# services/qdrant_service.py:38-46
QDRANT_CONFIG = {
    "name": "Qdrant",
    "host": "localhost",
    "port": 6333,
    "icon": "🎯",
    "url": "http://localhost:6333",
    "health_check_endpoint": "/collections",
    "docker_image": "qdrant/qdrant",
}
```

**Qdrant起動方法:**

```bash
# Docker Compose
docker-compose -f docker-compose/docker-compose.yml up -d

# または直接Docker
docker run -p 6333:6333 qdrant/qdrant

# サーバー管理スクリプト
python server.py
```

### 6.2 コレクション管理（CRUD）

**全コレクション取得:**

```python
# services/qdrant_service.py:377-397
def get_all_collections(client: QdrantClient) -> List[Dict[str, Any]]:
    collections = client.get_collections()
    collection_list = []

    for collection in collections.collections:
        info = client.get_collection(collection.name)
        collection_list.append({
            "name": collection.name,
            "points_count": info.points_count,
            "status": info.status,
        })

    return collection_list
```

**コレクション統計:**

```python
# services/qdrant_service.py:336-374
def get_collection_stats(client, collection_name) -> Optional[Dict]:
    collection_info = client.get_collection(collection_name)

    return {
        "total_points": collection_info.points_count,
        "vector_config": {
            "size": vectors_config.size,      # 1536
            "distance": str(vectors_config.distance)  # "Cosine"
        },
        "status": collection_info.status  # "green"
    }
```

**全コレクション削除:**

```python
# services/qdrant_service.py:400-424
def delete_all_collections(client, excluded: List[str] = None) -> int:
    excluded = excluded or []
    collections = get_all_collections(client)

    deleted_count = 0
    for col in collections:
        if col["name"] not in excluded:
            client.delete_collection(collection_name=col["name"])
            deleted_count += 1

    return deleted_count
```

### 6.3 ヘルスチェック

```python
# services/qdrant_service.py:91-137
class QdrantHealthChecker:
    """Qdrantサーバーの接続状態をチェック"""

    def check_port(self, host: str, port: int, timeout: float = 2.0) -> bool:
        """ポートが開いているかチェック"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0

    def check_qdrant(self) -> Tuple[bool, str, Optional[Dict]]:
        """Qdrant接続チェック"""
        # ポートチェック
        if not self.check_port(QDRANT_CONFIG["host"], QDRANT_CONFIG["port"]):
            return False, "Connection refused (port closed)", None

        # API接続テスト
        self.client = QdrantClient(url=QDRANT_CONFIG["url"], timeout=5)
        collections = self.client.get_collections()

        metrics = {
            "collection_count": len(collections.collections),
            "collections": [c.name for c in collections.collections],
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
        }

        return True, "Connected", metrics
```

<details>
<summary>📝 QdrantHealthChecker 完全実装コード</summary>

```python
# services/qdrant_service.py:91-137

class QdrantHealthChecker:
    """Qdrantサーバーの接続状態をチェック"""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.client = None

    def check_port(self, host: str, port: int, timeout: float = 2.0) -> bool:
        """ポートが開いているかチェック"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            if self.debug_mode:
                logger.error(f"Port check failed for {host}:{port}: {e}")
            return False

    def check_qdrant(self) -> Tuple[bool, str, Optional[Dict]]:
        """Qdrant接続チェック"""
        start_time = time.time()

        # まずポートチェック
        if not self.check_port(QDRANT_CONFIG["host"], QDRANT_CONFIG["port"]):
            return False, "Connection refused (port closed)", None

        try:
            self.client = QdrantClient(url=QDRANT_CONFIG["url"], timeout=5)

            # コレクション取得
            collections = self.client.get_collections()

            metrics = {
                "collection_count": len(collections.collections),
                "collections": [c.name for c in collections.collections],
                "response_time_ms": round((time.time() - start_time) * 1000, 2),
            }

            return True, "Connected", metrics

        except Exception as e:
            error_msg = str(e)
            if self.debug_mode:
                error_msg = f"{error_msg}\n{traceback.format_exc()}"
            return False, error_msg, None
```

**ポイント:**
- `check_port()`: ソケットレベルでポート開放確認（高速な事前チェック）
- `timeout=2.0`: ポートチェックは2秒、API接続は5秒でタイムアウト
- `response_time_ms`: レスポンスタイム計測（パフォーマンス監視用）
- `debug_mode`: トラブルシューティング時にスタックトレース表示

</details>

**ヘルスチェック結果:**

```python
# 成功時
(True, "Connected", {
    "collection_count": 6,
    "collections": ["qa_corpus", "qa_livedoor_a02_20_llm", ...],
    "response_time_ms": 12.34
})

# 失敗時
(False, "Connection refused (port closed)", None)
```

### 6.4 統計情報取得

**QdrantDataFetcher クラス:**

```python
# services/qdrant_service.py:144-329
class QdrantDataFetcher:
    """Qdrantからデータを取得"""

    def fetch_collections(self) -> pd.DataFrame:
        """コレクション一覧をDataFrameで取得"""

    def fetch_collection_points(self, collection_name, limit=50) -> pd.DataFrame:
        """コレクションの詳細データを取得"""

    def fetch_collection_info(self, collection_name) -> Dict:
        """コレクションの詳細情報を取得"""

    def fetch_collection_source_info(self, collection_name, sample_size=200) -> Dict:
        """データソース情報を取得（ソース別の件数・割合）"""
```

**ソース情報の例:**

```python
{
    "total_points": 1500,
    "sources": {
        "a02_qa_pairs_livedoor.csv": {
            "sample_count": 150,
            "method": "llm",
            "domain": "livedoor",
            "estimated_total": 1500,
            "percentage": 100.0
        }
    },
    "sample_size": 150
}
```

<details>
<summary>📝 fetch_collection_source_info() 完全実装コード</summary>

```python
# services/qdrant_service.py:278-329

def fetch_collection_source_info(
    self, collection_name: str, sample_size: int = 200
) -> Dict[str, Any]:
    """コレクションのデータソース情報を取得（サンプリングベース推定）"""
    try:
        collection_info = self.client.get_collection(collection_name)
        total_points = collection_info.points_count

        # サンプルポイントを取得
        points_result = self.client.scroll(
            collection_name=collection_name,
            limit=min(sample_size, total_points),
            with_payload=True,
            with_vectors=False,
        )
        points = points_result[0]

        # sourceとgeneration_methodを集計
        source_stats = {}
        for point in points:
            if point.payload:
                source = point.payload.get("source", "unknown")
                method = point.payload.get("generation_method", "unknown")
                domain = point.payload.get("domain", "unknown")

                if source not in source_stats:
                    source_stats[source] = {
                        "sample_count": 0,
                        "method": method,
                        "domain": domain,
                    }
                source_stats[source]["sample_count"] += 1

        # 全体のデータ数を推定
        sample_total = len(points)
        for source, stats in source_stats.items():
            ratio = stats["sample_count"] / sample_total if sample_total > 0 else 0
            stats["estimated_total"] = int(total_points * ratio)
            stats["percentage"] = ratio * 100

        return {
            "total_points": total_points,
            "sources": source_stats,
            "sample_size": sample_total,
        }

    except Exception as e:
        logger.error(f"ソース情報取得エラー: {e}")
        return {"total_points": 0, "sources": {}, "sample_size": 0, "error": str(e)}
```

**ポイント:**
- `sample_size=200`: サンプリングで全体を推定（高速化）
- `with_vectors=False`: ペイロードのみ取得（帯域幅節約）
- 比率計算: `estimated_total = total_points * (sample_count / sample_total)`
- ソース別の統計: ファイル名、生成方式、ドメイン情報を集計

</details>

### 6.5 CSV→Qdrant登録フロー（UI）

Streamlit UIでの完全な登録フローを示す。

**登録フロー図:**

```
[CSVファイル選択]
    │
    │ qa_output/*.csv
    ▼
[1. データ読み込み]  ←── load_csv_for_qdrant()
    │
    │ DataFrame (question, answer)
    ▼
[2. コレクション作成]  ←── create_or_recreate_collection_for_qdrant()
    │
    │ recreate=True/False
    ▼
[3. 埋め込み入力構築]  ←── build_inputs_for_embedding()
    │
    │ Q+A連結テキスト
    ▼
[4. 埋め込み生成]  ←── embed_texts_for_qdrant()
    │
    │ 1536次元ベクトル
    ▼
[5. ポイント構築]  ←── build_points_for_qdrant()
    │
    │ PointStruct (id, vector, payload)
    ▼
[6. アップサート]  ←── upsert_points_to_qdrant()
    │
    ▼
[登録完了]
```

<details>
<summary>📝 CSV→Qdrant登録 完全実装コード</summary>

```python
# ui/pages/qdrant_registration_page.py:333-394

if run_registration:
    # CSVパスを構築
    csv_path = Path("qa_output") / selected_csv

    if not csv_path.exists():
        st.error(f"CSVファイルが見つかりません: {csv_path}")
        st.stop()

    try:
        client = QdrantClient(url=qdrant_url)

        # ステップ1: CSV読み込み
        with st.spinner("📂 CSVファイルを読み込み中..."):
            df = load_csv_for_qdrant(str(csv_path), limit=data_limit)
            st.info(f"読み込み完了: {len(df)} 件")

        # ステップ2: コレクション作成
        with st.spinner(f"📦 コレクション '{collection_name}' を準備中..."):
            create_or_recreate_collection_for_qdrant(
                client, collection_name, recreate_collection
            )

        # ステップ3: 埋め込み入力構築
        with st.spinner("📝 埋め込み入力を構築中..."):
            texts = build_inputs_for_embedding(df, include_answer=True)

        # ステップ4: 埋め込み生成
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("🧠 埋め込みベクトルを生成中..."):
            vectors = embed_texts_for_qdrant(
                texts, model="text-embedding-3-small"
            )
            progress_bar.progress(50)
            status_text.text(f"埋め込み完了: {len(vectors)} 件")

        # ステップ5: ポイント構築
        with st.spinner("🔧 Qdrantポイントを構築中..."):
            points = build_points_for_qdrant(df, vectors, domain, selected_csv)

        # ステップ6: アップサート
        with st.spinner("💾 Qdrantにデータを登録中..."):
            count = upsert_points_to_qdrant(client, collection_name, points)
            progress_bar.progress(100)
            status_text.text(f"登録完了: {count} 件")

        st.success(f"✅ {count} 件のデータを '{collection_name}' に登録しました")

        # 登録後の統計表示
        stats = get_collection_stats(client, collection_name)
        if stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("総ポイント数", stats["total_points"])
            with col2:
                st.metric("ベクトル次元", stats["vector_config"]["size"])
            with col3:
                st.metric("ステータス", stats["status"])

    except Exception as e:
        st.error(f"❌ 登録エラー: {str(e)}")
```

**ポイント:**
- 6ステップの登録フロー（CSVロード→コレクション作成→埋め込み入力構築→埋め込み生成→ポイント構築→アップサート）
- `include_answer=True`: Q+A連結で検索精度向上
- `recreate_collection`: 既存データを削除して再登録するか選択可能
- 進捗バー: 埋め込み生成→アップサートで50%→100%
- 登録後の統計表示: ポイント数、ベクトル次元、ステータス

</details>

---

## 7. 付録

### 7.1 コレクション名とCSVファイルの対応表

```python
# services/qdrant_service.py:60-68
COLLECTION_CSV_MAPPING = {
    "qa_corpus": "qa_pairs_corpus.csv",
    "qa_cc_news_a02_llm": "a02_qa_pairs_cc_news.csv",
    "qa_cc_news_a03_rule": "a03_qa_pairs_cc_news.csv",
    "qa_cc_news_a10_hybrid": "a10_qa_pairs_cc_news.csv",
    "qa_livedoor_a02_20_llm": "a02_qa_pairs_livedoor.csv",
    "qa_livedoor_a03_rule": "a03_qa_pairs_livedoor.csv",
    "qa_livedoor_a10_hybrid": "a10_qa_pairs_livedoor.csv",
}

# 埋め込み設定
# services/qdrant_service.py:49-57
COLLECTION_EMBEDDINGS_SEARCH = {
    "qa_corpus": {"model": "text-embedding-3-small", "dims": 1536},
    "qa_cc_news_a02_llm": {"model": "text-embedding-3-small", "dims": 1536},
    "qa_cc_news_a03_rule": {"model": "text-embedding-3-small", "dims": 1536},
    "qa_cc_news_a10_hybrid": {"model": "text-embedding-3-small", "dims": 1536},
    "qa_livedoor_a02_20_llm": {"model": "text-embedding-3-small", "dims": 1536},
    "qa_livedoor_a03_rule": {"model": "text-embedding-3-small", "dims": 1536},
    "qa_livedoor_a10_hybrid": {"model": "text-embedding-3-small", "dims": 1536},
}
```

### 7.2 コード参照一覧

| 機能 | ファイル | 関数/クラス | 行番号 |
|-----|---------|------------|-------|
| Qdrant設定 | services/qdrant_service.py | QDRANT_CONFIG | 38-46 |
| バッチ分割 | services/qdrant_service.py | batched() | 75-84 |
| ヘルスチェック | services/qdrant_service.py | QdrantHealthChecker | 91-137 |
| データ取得 | services/qdrant_service.py | QdrantDataFetcher | 144-329 |
| コレクション統計 | services/qdrant_service.py | get_collection_stats() | 336-374 |
| 全コレクション取得 | services/qdrant_service.py | get_all_collections() | 377-397 |
| 全コレクション削除 | services/qdrant_service.py | delete_all_collections() | 400-424 |
| CSV読み込み | services/qdrant_service.py | load_csv_for_qdrant() | 431-459 |
| 埋め込み入力構築 | services/qdrant_service.py | build_inputs_for_embedding() | 462-466 |
| 埋め込み生成（バッチ） | services/qdrant_service.py | embed_texts_for_qdrant() | 469-531 |
| コレクション作成 | services/qdrant_service.py | create_or_recreate_collection_for_qdrant() | 534-562 |
| ポイント構築 | services/qdrant_service.py | build_points_for_qdrant() | 565-589 |
| アップサート | services/qdrant_service.py | upsert_points_to_qdrant() | 592-603 |
| 検索クエリベクトル化 | services/qdrant_service.py | embed_query_for_search() | 610-619 |
| 全ポイント取得 | services/qdrant_service.py | scroll_all_points_with_vectors() | 626-672 |
| コレクション統合 | services/qdrant_service.py | merge_collections() | 675-779 |
| 登録UI | ui/pages/qdrant_registration_page.py | show_qdrant_registration_page() | 39-600 |
| 検索UI | ui/pages/qdrant_search_page.py | show_qdrant_search_page() | - |
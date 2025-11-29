# python a42_qdrant_registration.py --recreate --include-answer

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
a42_qdrant_registration.py — 3つの異なるコレクションへの分離登録版
--------------------------------------------------------------------------------
cc_newsドメインのQ&Aデータを、生成方法ごとに別々のコレクションに登録します。

【コレクション構成】
- qa_cc_news_a02_llm    : a02_qa_pairs_cc_news.csv (LLM生成方式)
- qa_cc_news_a03_rule   : a03_qa_pairs_cc_news.csv (ルールベース生成方式)
- qa_cc_news_a10_hybrid : a10_qa_pairs_cc_news.csv (ハイブリッド生成方式)

【主な変更点（a30との違い）】
- 単一コレクション → 3つの独立したコレクション
- 各CSVファイルが専用のコレクションを持つ
- domainとgeneration_methodフィールドを保持（互換性のため）

使い方：
  # 1. CSVファイルを生成（a20_output_qa_csv.pyを実行）
  python a20_output_qa_csv.py

  # 2. Qdrantサーバーを起動
  export OPENAI_API_KEY=sk-...
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

  # -------------------------------------------------------------
  # 3. 3つのコレクションにデータを登録
  python a42_qdrant_registration.py --recreate --include-answer
  # -------------------------------------------------------------
# ローカルデータの登録：
python a42_qdrant_registration.py \
    --input-file qa_output/qa_pairs_upload_20251122_182355.csv \
    --recreate \
    --include-answer
  # -------------------------------------------------------------

  # 4. 特定のコレクションのみ登録
  python a42_qdrant_registration.py --collection qa_cc_news_a02_llm --recreate

  # 5. 検索テスト（特定コレクション）
  python a42_qdrant_registration.py --search "気候変動" --collection qa_cc_news_a02_llm

  # 6. ローカルファイル登録（カスタムアップロード）
  python a42_qdrant_registration.py --input-file qa_output/qa_pairs_upload_20251122_182355.csv --recreate --include-answer

  # 7. ローカルファイル登録（データ件数制限付き）
  python a42_qdrant_registration.py --input-file data/my_qa_data.csv --limit 100 --include-answer

主要引数：
  --recreate          : コレクション削除→新規作成
  --collection        : 特定コレクションのみ処理（指定なしで全コレクション）
  --input-file        : ローカルファイルを登録（CSV/TXT/JSON/JSONL）
  --qdrant-url        : 既定は http://localhost:6333
  --batch-size        : Embeddings/Upsert バッチサイズ（既定 32）
  --limit             : データ件数上限（開発用、0=無制限）
  --include-answer    : 埋め込み入力に answer も結合（question + "\\n" + answer）
  --search            : クエリ指定で検索のみ実行
  --topk              : 上位件数（既定5）
"""
import argparse
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Any
from pathlib import Path

import pandas as pd

try:
    import yaml  # PyYAML
except Exception:
    yaml = None

try:
    import helper_api as hapi
except Exception:
    hapi = None

try:
    import helper_rag as hrag
except Exception:
    hrag = None

from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI

# ------------------ デフォルト設定 ------------------
DEFAULTS = {
    "rag": {
        "include_answer_in_embedding": False,
    },
    "embeddings": {
        "primary": {"provider": "openai", "model": "text-embedding-3-small", "dims": 1536},
    },
    "qdrant": {"url": "http://localhost:6333"},
}

# ------------------ コレクション定義 ------------------
# CSVファイル/TXTファイル → コレクション名 → 生成方法のマッピング
COLLECTION_MAPPINGS = [
    {
        "csv_file": "qa_output/a02_qa_pairs_cc_news.csv",
        "collection": "qa_cc_news_a02_llm",
        "generation_method": "a02_make_qa",
        "domain": "cc_news",
        "description": "LLM生成方式（a02_make_qa_para.py）",
        "type": "qa"
    },
    {
        "csv_file": "qa_output/a03_qa_pairs_cc_news.csv",
        "collection": "qa_cc_news_a03_rule",
        "generation_method": "a03_coverage",
        "domain": "cc_news",
        "description": "ルールベース生成方式（a03_rag_qa_coverage_improved.py）",
        "type": "qa"
    },
    {
        "csv_file": "qa_output/a10_qa_pairs_cc_news.csv",
        "collection": "qa_cc_news_a10_hybrid",
        "generation_method": "a10_hybrid",
        "domain": "cc_news",
        "description": "ハイブリッド生成方式（a10_qa_optimized_hybrid_batch.py）",
        "type": "qa"
    },
    {
        "csv_file": "qa_output/a02_qa_pairs_livedoor.csv",
        "collection": "qa_livedoor_a02_20_llm",
        "generation_method": "a02_make_qa",
        "domain": "livedoor",
        "description": "LLM生成方式（a02_make_qa_para.py）- livedoorデータ",
        "type": "qa"
    },
    {
        "csv_file": "qa_output/a03_qa_pairs_livedoor.csv",
        "collection": "qa_livedoor_a03_rule",
        "generation_method": "a03_coverage",
        "domain": "livedoor",
        "description": "ルールベース生成方式（a03_rag_qa_coverage_improved.py）- livedoorデータ",
        "type": "qa"
    },
    {
        "csv_file": "qa_output/a10_qa_pairs_livedoor.csv",
        "collection": "qa_livedoor_a10_hybrid",
        "generation_method": "a10_hybrid",
        "domain": "livedoor",
        "description": "ハイブリッド生成方式（a10_qa_optimized_hybrid_batch.py）- livedoorデータ",
        "type": "qa"
    },
    {
        "csv_file": "OUTPUT/cc_news.txt",
        "collection": "raw_cc_news",
        "generation_method": "raw_text",
        "domain": "cc_news",
        "description": "生テキストデータ（cc_news）",
        "type": "raw"
    },
    {
        "csv_file": "OUTPUT/livedoor.txt",
        "collection": "raw_livedoor",
        "generation_method": "raw_text",
        "domain": "livedoor",
        "description": "生テキストデータ（livedoor）",
        "type": "raw"
    }
]


# ------------------ 設定ロード ------------------
def load_config(path: str = "config.yml") -> Dict[str, Any]:
    cfg = {}
    if yaml and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    # マージ（浅いマージで十分。必要なら深いマージに差し替え）
    def merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst.setdefault(k, v)
    full = {}
    merge(full, DEFAULTS)
    merge(full, cfg)
    return full

# ------------------ 小道具 ------------------
def batched(seq: Iterable, size: int):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

# ------------------ OpenAIクライアント ------------------
def get_openai_client():
    if hapi and hasattr(hapi, "get_openai_client"):
        return hapi.get_openai_client()
    return OpenAI()

# ------------------ 埋め込み実装（helper優先） ------------------
def embed_texts_openai(texts: List[str], model: str, client: Optional[OpenAI] = None) -> List[List[float]]:
    client = client or get_openai_client()
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def embed_texts(texts: List[str], model: str, batch_size: int = 128) -> List[List[float]]:
    """
    テキストをバッチ処理でEmbeddingに変換
    OpenAIの8192トークン制限を考慮して動的にバッチサイズを調整

    Note: batch_size引数は後方互換性のために残しているが、実際にはトークン数のみで制御
    """
    if hrag and hasattr(hrag, "embed_texts"):
        return hrag.embed_texts(texts, model=model, batch_size=batch_size)

    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    client = get_openai_client()

    # OpenAI Embeddingの最大トークン数（安全マージン込み）
    MAX_TOKENS_PER_REQUEST = 8000  # 8192から余裕を持たせて8000

    # ★ 空文字列・空白のみの文字列を除外し、インデックスマッピングを保持
    valid_texts = []
    valid_indices = []
    for i, text in enumerate(texts):
        if text and text.strip():  # 空文字列と空白のみを除外
            valid_texts.append(text)
            valid_indices.append(i)

    # 全て空文字列の場合はダミーベクトルを返す
    if not valid_texts:
        print("\r   [WARN] 全てのテキストが空文字列です。ダミーベクトルを返します。", flush=True)
        return [[0.0] * 1536] * len(texts)

    # 有効なテキストのみで埋め込み生成
    valid_vecs: List[List[float]] = []
    current_batch = []
    current_tokens = 0
    batch_count = 0

    for i, text in enumerate(valid_texts):
        text_tokens = len(enc.encode(text))

        # 単一チャンクが制限を超える場合はエラー
        if text_tokens > MAX_TOKENS_PER_REQUEST:
            raise ValueError(
                f"Single text at index {valid_indices[i]} has {text_tokens} tokens, "
                f"which exceeds MAX_TOKENS_PER_REQUEST ({MAX_TOKENS_PER_REQUEST}). "
                f"Please reduce chunk_size when creating chunks."
            )

        # 追加前にチェック：制限を超える場合は先にバッチを送信
        if current_tokens + text_tokens > MAX_TOKENS_PER_REQUEST:
            # 現在のバッチがあれば送信
            if current_batch:
                batch_count += 1
                print(f"\r   埋め込み生成中: バッチ{batch_count} ({len(current_batch)}件, {current_tokens}トークン)... ", end="", flush=True)
                valid_vecs.extend(embed_texts_openai(current_batch, model=model, client=client))
                current_batch = []
                current_tokens = 0

        # バッチに追加
        current_batch.append(text)
        current_tokens += text_tokens

    # 残りのバッチを処理
    if current_batch:
        batch_count += 1
        print(f"\r   埋め込み生成中: バッチ{batch_count} ({len(current_batch)}件, {current_tokens}トークン)... ", end="", flush=True)
        valid_vecs.extend(embed_texts_openai(current_batch, model=model, client=client))

    # 元のインデックスに合わせてベクトルを再配置（空文字列はゼロベクトル）
    vecs: List[List[float]] = []
    valid_vec_idx = 0
    for i in range(len(texts)):
        if i in valid_indices:
            vecs.append(valid_vecs[valid_vec_idx])
            valid_vec_idx += 1
        else:
            # 空文字列の場合はゼロベクトル
            vecs.append([0.0] * 1536)

    return vecs

# ------------------ 入力テキスト構築 ------------------
def build_inputs(df: pd.DataFrame, include_answer: bool) -> List[str]:
    if include_answer:
        return (df["question"].astype(str) + "\n" + df["answer"].astype(str)).tolist()
    return df["question"].astype(str).tolist()

# ------------------ CSVロード ------------------
def load_csv(path: str, required=("question", "answer"), limit: int = 0) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    # 列名マッピングが必要ならここで調整（例: 'Question'->'question'）
    column_mappings = {
        'Question': 'question',
        'Response': 'answer',
        'Answer': 'answer',
        'correct_answer': 'answer'
    }
    df = df.rename(columns=column_mappings)
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{path} には '{col}' 列が必要です（列: {list(df.columns)}）")
    df = df.fillna("").drop_duplicates(subset=list(required)).reset_index(drop=True)
    if limit and limit > 0:
        df = df.head(limit).copy()
    return df

# ------------------ TXTロード（生テキスト） ------------------
def detect_language(text: str) -> str:
    """
    テキストの言語を判定（日本語 or 英語）

    Args:
        text: 判定するテキスト

    Returns:
        "ja" (日本語) or "en" (英語)
    """
    # 日本語文字（ひらがな、カタカナ、漢字）の検出
    japanese_chars = len([c for c in text if '\u3040' <= c <= '\u309F' or  # ひらがな
                                            '\u30A0' <= c <= '\u30FF' or  # カタカナ
                                            '\u4E00' <= c <= '\u9FFF'])    # 漢字

    total_chars = len([c for c in text if not c.isspace()])

    # 日本語文字が30%以上なら日本語と判定
    if total_chars > 0 and japanese_chars / total_chars > 0.3:
        return "ja"
    return "en"

def chunk_japanese_text(text: str, max_tokens: int = 500) -> List[str]:
    """
    日本語テキストをセマンティックにチャンク分割
    段落→文単位で意味を保持しながら分割
    単一文が大きすぎる場合は強制的に固定長分割

    Args:
        text: 分割する日本語テキスト
        max_tokens: チャンクの最大トークン数

    Returns:
        チャンクのリスト
    """
    import tiktoken
    import re

    enc = tiktoken.get_encoding("cl100k_base")

    # 段落で分割（空行区切り）
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = len(enc.encode(para))

        # 現在のチャンクに追加できるか確認
        if current_tokens + para_tokens <= max_tokens:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += para
            current_tokens += para_tokens
        else:
            # 現在のチャンクを保存
            if current_chunk:
                chunks.append(current_chunk)

            # 段落が大きすぎる場合は文単位で分割
            if para_tokens > max_tokens:
                # 日本語の句点で文分割
                sentences = re.split(r'([。！？\n])', para)
                temp_chunk = ""
                temp_tokens = 0

                for i in range(0, len(sentences), 2):
                    sent = sentences[i]
                    if i + 1 < len(sentences):
                        sent += sentences[i + 1]

                    sent_tokens = len(enc.encode(sent))

                    # 単一文が大きすぎる場合は強制分割
                    if sent_tokens > max_tokens:
                        # 現在のtemp_chunkを保存
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = ""
                            temp_tokens = 0

                        # 長文を固定長で強制分割
                        sent_enc = enc.encode(sent)
                        for chunk_start in range(0, len(sent_enc), max_tokens):
                            chunk_tokens = sent_enc[chunk_start:chunk_start + max_tokens]
                            chunk_text = enc.decode(chunk_tokens).strip()
                            # 空文字列チェック
                            if chunk_text:
                                chunks.append(chunk_text)
                    elif temp_tokens + sent_tokens <= max_tokens:
                        temp_chunk += sent
                        temp_tokens += sent_tokens
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sent
                        temp_tokens = sent_tokens

                if temp_chunk:
                    current_chunk = temp_chunk
                    current_tokens = temp_tokens
                else:
                    current_chunk = ""
                    current_tokens = 0
            else:
                current_chunk = para
                current_tokens = para_tokens

    # 最後のチャンク
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def chunk_english_text(text: str, max_tokens: int = 500) -> List[str]:
    """
    英語テキストをセマンティックにチャンク分割
    段落→文単位で意味を保持しながら分割
    単一文が大きすぎる場合は強制的に固定長分割

    Args:
        text: 分割する英語テキスト
        max_tokens: チャンクの最大トークン数

    Returns:
        チャンクのリスト
    """
    import tiktoken
    import re

    enc = tiktoken.get_encoding("cl100k_base")

    # 段落で分割（空行区切り）
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = len(enc.encode(para))

        # 現在のチャンクに追加できるか確認
        if current_tokens + para_tokens <= max_tokens:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += para
            current_tokens += para_tokens
        else:
            # 現在のチャンクを保存
            if current_chunk:
                chunks.append(current_chunk)

            # 段落が大きすぎる場合は文単位で分割
            if para_tokens > max_tokens:
                # 英語の文分割（ピリオド+スペース+大文字で分割、略語を考慮）
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', para)
                temp_chunk = ""
                temp_tokens = 0

                for sent in sentences:
                    sent_tokens = len(enc.encode(sent))

                    # 単一文が大きすぎる場合は強制分割
                    if sent_tokens > max_tokens:
                        # 現在のtemp_chunkを保存
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = ""
                            temp_tokens = 0

                        # 長文を固定長で強制分割
                        sent_enc = enc.encode(sent)
                        for chunk_start in range(0, len(sent_enc), max_tokens):
                            chunk_tokens = sent_enc[chunk_start:chunk_start + max_tokens]
                            chunk_text = enc.decode(chunk_tokens).strip()
                            # 空文字列チェック
                            if chunk_text:
                                chunks.append(chunk_text)
                    elif temp_tokens + sent_tokens <= max_tokens:
                        if temp_chunk:
                            temp_chunk += " "
                        temp_chunk += sent
                        temp_tokens += sent_tokens
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sent
                        temp_tokens = sent_tokens

                if temp_chunk:
                    current_chunk = temp_chunk
                    current_tokens = temp_tokens
                else:
                    current_chunk = ""
                    current_tokens = 0
            else:
                current_chunk = para
                current_tokens = para_tokens

    # 最後のチャンク
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def load_txt(path: str, limit: int = 0, chunk_size: int = 500) -> pd.DataFrame:
    """
    テキストファイルを読み込み、セマンティックチャンクに分割してDataFrameに変換
    言語を自動判定し、適切な方法でチャンク分割する

    Args:
        path: テキストファイルパス
        limit: チャンク数の上限（0=無制限）
        chunk_size: チャンクの最大トークン数（デフォルト500）

    Returns:
        チャンクを含むDataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"TXT not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # 言語判定
    lang = detect_language(full_text[:1000])  # 先頭1000文字で判定
    print(f"   言語判定: {'日本語' if lang == 'ja' else '英語'}")

    # 言語に応じたチャンク分割
    if lang == "ja":
        chunks = chunk_japanese_text(full_text, max_tokens=chunk_size)
    else:
        chunks = chunk_english_text(full_text, max_tokens=chunk_size)

    print(f"   チャンク数: {len(chunks):,}件（最大{chunk_size}トークン/チャンク）")

    if limit and limit > 0:
        chunks = chunks[:limit]
        print(f"   制限適用後: {len(chunks):,}件")

    # DataFrameに変換（textカラムのみ）
    df = pd.DataFrame({"text": chunks})
    return df

# ------------------ Qdrant: コレクション作成（Named Vectors対応） ------------------
def create_or_recreate_collection(client: QdrantClient, name: str, recreate: bool,
                                  embeddings_cfg: Dict[str, Dict[str, Any]]):
    # embeddings_cfg: dict[name] = {"model": "...", "dims": int}
    # Named Vectors：複数キーなら dict を、単一なら VectorParams を使う
    if len(embeddings_cfg) == 1:
        dims = list(embeddings_cfg.values())[0]["dims"]
        vectors_config = models.VectorParams(size=dims, distance=models.Distance.COSINE)
    else:
        # Named vectors
        vectors_config = {
            k: models.VectorParams(size=v["dims"], distance=models.Distance.COSINE)
            for k, v in embeddings_cfg.items()
        }
    if recreate:
        client.recreate_collection(collection_name=name, vectors_config=vectors_config)
    else:
        # 無ければ作成
        try:
            client.get_collection(name)
        except Exception:
            client.create_collection(collection_name=name, vectors_config=vectors_config)
    # よく使うpayloadの索引（任意）
    try:
        client.create_payload_index(name, field_name="domain", field_schema=models.PayloadSchemaType.KEYWORD)
    except Exception:
        pass
    try:
        client.create_payload_index(name, field_name="generation_method", field_schema=models.PayloadSchemaType.KEYWORD)
    except Exception:
        pass

# ------------------ ポイント構築（Named Vectors対応） ------------------
def build_points(df: pd.DataFrame, vectors_by_name: Dict[str, List[List[float]]], domain: str, source_file: str,
                 generation_method: str = None, data_type: str = "qa") -> List[models.PointStruct]:
    # vectors_by_name: name -> list[vec]
    n = len(df)
    for name, vecs in vectors_by_name.items():
        if len(vecs) != n:
            raise ValueError(f"vectors length mismatch for '{name}': df={n}, vecs={len(vecs)}")
    now_iso = datetime.now(timezone.utc).isoformat()
    points: List[models.PointStruct] = []

    for i, row in enumerate(df.itertuples(index=False)):
        if data_type == "qa":
            # Q&Aペア用のペイロード
            payload = {
                "domain": domain,
                "generation_method": generation_method or "unknown",
                "question": getattr(row, "question"),
                "answer": getattr(row, "answer"),
                "source": os.path.basename(source_file),
                "created_at": now_iso,
                "schema": "qa:v1",
            }
        else:  # raw text
            # 生テキスト用のペイロード
            payload = {
                "domain": domain,
                "generation_method": generation_method or "unknown",
                "text": getattr(row, "text"),
                "source": os.path.basename(source_file),
                "created_at": now_iso,
                "schema": "raw:v1",
            }

        # Qdrant requires point IDs to be UUID or unsigned integer
        # ソースファイル名も含めてユニークなIDを生成
        pid = abs(hash(f"{domain}-{generation_method}-{source_file}-{i}")) & 0x7FFFFFFFFFFFFFFF  # Convert to positive 64-bit integer
        if len(vectors_by_name) == 1:
            # 単一ベクトル
            vec = list(vectors_by_name.values())[0][i]
            points.append(models.PointStruct(id=pid, vector=vec, payload=payload))
        else:
            # Named Vectors（dict渡し）
            vecs_dict = {name: vecs[i] for name, vecs in vectors_by_name.items()}
            points.append(models.PointStruct(id=pid, vector=vecs_dict, payload=payload))
    return points

def upsert_points(client: QdrantClient, collection: str, points: List[models.PointStruct], batch_size: int = 128) -> int:
    count = 0
    for chunk in batched(points, batch_size):
        client.upsert(collection_name=collection, points=chunk)
        count += len(chunk)
    return count

# ------------------ 検索（Named Vectors対応） ------------------
def embed_one(text: str, model: str) -> List[float]:
    return embed_texts([text], model=model, batch_size=1)[0]

def search(client: QdrantClient, collection: str, query: str, using_vec: str, model_for_using: str,
           topk: int = 5, domain: Optional[str] = None, generation_method: Optional[str] = None):
    qvec = embed_one(query, model=model_for_using)
    qfilter = None
    filter_conditions = []
    if domain:
        filter_conditions.append(models.FieldCondition(key="domain", match=models.MatchValue(value=domain)))
    if generation_method:
        filter_conditions.append(models.FieldCondition(key="generation_method", match=models.MatchValue(value=generation_method)))
    if filter_conditions:
        qfilter = models.Filter(must=filter_conditions)
    
    # ベクトル設定を確認して適切な検索方法を選択
    try:
        collection_info = client.get_collection(collection)
        # Named Vectorsかどうか確認
        has_named_vectors = hasattr(collection_info.config.params.vectors, '__iter__') and not isinstance(
            collection_info.config.params.vectors, (str, models.VectorParams))
        
        if has_named_vectors and using_vec:
            # Named Vectorsの場合、using引数を試す
            try:
                hits = client.search(collection_name=collection, query_vector=qvec, limit=topk,
                                   query_filter=qfilter, using=using_vec)
            except (TypeError, Exception):
                # using引数がサポートされていない、または他のエラーの場合
                hits = client.search(collection_name=collection, query_vector=qvec, limit=topk,
                                   query_filter=qfilter)
        else:
            # 単一ベクトルの場合、using引数なしで検索
            hits = client.search(collection_name=collection, query_vector=qvec, limit=topk,
                               query_filter=qfilter)
    except Exception:
        # コレクション情報が取得できない場合、using引数なしで検索
        hits = client.search(collection_name=collection, query_vector=qvec, limit=topk,
                           query_filter=qfilter)
    
    return hits

# ------------------ メイン ------------------
def main():
    cfg = load_config("config.yml")
    rag_cfg = cfg.get("rag", {})
    embeddings_cfg: Dict[str, Dict[str, Any]] = cfg.get("embeddings", {})
    qdrant_url = (cfg.get("qdrant", {}) or {}).get("url", "http://localhost:6333")

    ap = argparse.ArgumentParser(
        description="3つのQ&Aデータセットをそれぞれ独立したQdrantコレクションに登録"
    )
    ap.add_argument("--recreate", action="store_true",
                    help="Drop & create collection before upsert.")
    ap.add_argument("--collection", default=None,
                    help="特定コレクションのみ処理（指定なしで全コレクション）")
    ap.add_argument("--input-file", default=None,
                    help="ローカルファイルを登録（CSV/TXT/JSON/JSONL）")
    ap.add_argument("--qdrant-url", default=qdrant_url)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0,
                    help="Row limit per CSV for development (0=all)")
    ap.add_argument("--include-answer", action="store_true",
                    default=rag_cfg.get("include_answer_in_embedding", False),
                    help="Use 'question\\nanswer' as embedding input.")
    ap.add_argument("--search", default=None, help="Run search only.")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    if not embeddings_cfg:
        embeddings_cfg = DEFAULTS["embeddings"]

    # Qdrant client
    client = QdrantClient(url=args.qdrant_url, timeout=300)

    # 検索のみ
    if args.search:
        if not args.collection:
            print("[ERROR] 検索には --collection の指定が必要です")
            return

        model = embeddings_cfg["primary"]["model"]
        hits = search(client, args.collection, args.search, "primary", model, topk=args.topk)

        print(f"\n[Search] collection={args.collection} query={args.search!r}")
        for h in hits:
            method = h.payload.get('generation_method', 'unknown')
            question = h.payload.get('question', '')[:80]
            answer = h.payload.get('answer', '')[:80]
            print(f"score={h.score:.4f}  method={method}  Q: {question}  A: {answer}...")
        return

    # 処理対象のコレクションを決定
    if args.input_file:
        # ローカルファイル登録
        if not os.path.exists(args.input_file):
            print(f"[ERROR] ファイルが見つかりません: {args.input_file}")
            return

        # ファイル名からメタデータを自動生成
        file_path = Path(args.input_file)
        file_basename = file_path.stem  # 拡張子なしのファイル名
        file_ext = file_path.suffix.lower().lstrip('.')  # 拡張子

        # コレクション名を生成（例: qa_upload_20251122_182355）
        # ファイル名から"qa_pairs_"などのプレフィックスを除去
        clean_name = file_basename.replace('qa_pairs_', '').replace('a02_qa_pairs_', '').replace('a03_qa_pairs_', '').replace('a10_qa_pairs_', '')
        collection_name = f"qa_{clean_name}"

        # domainを推定（ファイル名から）
        if 'cc_news' in file_basename.lower():
            domain = 'cc_news'
        elif 'livedoor' in file_basename.lower():
            domain = 'livedoor'
        elif 'upload' in file_basename.lower():
            domain = 'custom_upload'
        else:
            domain = 'custom'

        # データタイプを判定
        if file_ext in ['txt', 'text']:
            data_type = 'raw'
            generation_method = 'raw_text'
        else:
            data_type = 'qa'
            generation_method = 'custom_upload'

        # 動的にマッピングを生成
        target_mappings = [{
            "csv_file": args.input_file,
            "collection": collection_name,
            "generation_method": generation_method,
            "domain": domain,
            "description": f"カスタムアップロード ({file_basename})",
            "type": data_type
        }]

        print("\n[INFO] ローカルファイル登録モード")
        print(f"  ファイル: {args.input_file}")
        print(f"  コレクション: {collection_name}")
        print(f"  ドメイン: {domain}")
        print(f"  タイプ: {data_type}")

    elif args.collection:
        # 特定コレクションのみ
        target_mappings = [m for m in COLLECTION_MAPPINGS if m["collection"] == args.collection]
        if not target_mappings:
            print(f"[ERROR] コレクション '{args.collection}' は定義されていません")
            print(f"利用可能なコレクション: {[m['collection'] for m in COLLECTION_MAPPINGS]}")
            return
    else:
        # 全コレクション
        target_mappings = COLLECTION_MAPPINGS

    # インジェスト処理
    print(f"\n[INFO] 処理対象: {len(target_mappings)} コレクション")
    print("=" * 80)

    total = 0
    for mapping in target_mappings:
        csv_file = mapping["csv_file"]
        collection_name = mapping["collection"]
        generation_method = mapping["generation_method"]
        domain = mapping["domain"]
        description = mapping["description"]

        print(f"\n📦 コレクション: {collection_name}")
        print(f"   説明: {description}")
        print(f"   ソース: {csv_file}")
        print("-" * 80)

        if not os.path.exists(csv_file):
            print(f"[WARN] ファイルが見つかりません: {csv_file} (スキップ)")
            continue

        # コレクション作成
        create_or_recreate_collection(client, collection_name, args.recreate, embeddings_cfg)

        # データタイプに応じてロード
        data_type = mapping.get("type", "qa")
        if data_type == "raw":
            # 生テキストファイルをロード
            df = load_txt(csv_file, limit=args.limit)
            print(f"   データ件数: {len(df):,}件")
            texts = df["text"].tolist()
        else:
            # Q&A CSVをロード
            df = load_csv(csv_file, limit=args.limit)
            print(f"   データ件数: {len(df):,}件")
            texts = build_inputs(df, include_answer=args.include_answer)

        # 埋め込み生成
        vectors_by_name: Dict[str, List[List[float]]] = {}
        for name, vcfg in embeddings_cfg.items():
            print(f"   埋め込み生成中: {name} (model={vcfg['model']})... ", end="", flush=True)
            vectors_by_name[name] = embed_texts(texts, model=vcfg["model"], batch_size=args.batch_size)
            print("✓")

        # ポイント構築とアップサート
        points = build_points(df, vectors_by_name, domain, csv_file, generation_method, data_type=data_type)
        print("   アップサート中... ", end="", flush=True)
        n = upsert_points(client, collection_name, points, batch_size=args.batch_size)
        print(f"✓ {n:,}件")

        total += n

    print("\n" + "=" * 80)
    print(f"✅ 完了: 総登録件数 {total:,}件")

    # 登録されたコレクション一覧を表示
    print("\n[INFO] 登録されたコレクション一覧:")
    print("-" * 80)
    all_collections = client.get_collections()
    for col in all_collections.collections:
        print(f"  • {col.name}")
    print("-" * 80)

    # 検証検索
    print("\n[INFO] 検証検索を実行中...")
    model = embeddings_cfg["primary"]["model"]

    for mapping in target_mappings:
        collection_name = mapping["collection"]
        try:
            # コレクション情報を取得
            info = client.get_collection(collection_name)
            print(f"\n  {collection_name}: {info.points_count:,}件登録済み")

            # サンプル検索
            hits = search(client, collection_name, "気候変動", "primary", model, topk=2)
            if hits:
                for h in hits[:1]:
                    q = h.payload.get('question', '')[:50]
                    print(f"    サンプル検索結果: score={h.score:.4f}  Q: {q}...")
        except Exception as e:
            print(f"    検証エラー: {e}")

if __name__ == "__main__":
    main()

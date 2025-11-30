#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qdrant_service.py - Qdrant操作サービス
======================================
Qdrantベクトルデータベースの操作を担当

機能:
- ヘルスチェック（QdrantHealthChecker）
- データ取得（QdrantDataFetcher）
- コレクション管理（CRUD）
- 埋め込み生成・登録
- 検索機能
"""

import os
import socket
import time
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Iterable

import pandas as pd
import tiktoken
from helper_embedding import create_embedding_client, get_embedding_dimensions
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


# ===================================================================
# Qdrant設定
# ===================================================================

QDRANT_CONFIG = {
    "name": "Qdrant",
    "host": "localhost",
    "port": 6333,
    "icon": "🎯",
    "url": "http://localhost:6333",
    "health_check_endpoint": "/collections",
    "docker_image": "qdrant/qdrant",
}

# コレクション固有の埋め込み設定
COLLECTION_EMBEDDINGS_SEARCH = {
    "qa_corpus": {"model": "gemini-embedding-001", "dims": 3072},
    "qa_cc_news_a02_llm": {"model": "gemini-embedding-001", "dims": 3072},
    "qa_cc_news_a03_rule": {"model": "gemini-embedding-001", "dims": 3072},
    "qa_cc_news_a10_hybrid": {"model": "gemini-embedding-001", "dims": 3072},
    "qa_livedoor_a02_20_llm": {"model": "gemini-embedding-001", "dims": 3072},
    "qa_livedoor_a03_rule": {"model": "gemini-embedding-001", "dims": 3072},
    "qa_livedoor_a10_hybrid": {"model": "gemini-embedding-001", "dims": 3072},
}

# コレクションとCSVファイルの対応表
COLLECTION_CSV_MAPPING = {
    "qa_corpus": "qa_pairs_corpus.csv",
    "qa_cc_news_a02_llm": "a02_qa_pairs_cc_news.csv",
    "qa_cc_news_a03_rule": "a03_qa_pairs_cc_news.csv",
    "qa_cc_news_a10_hybrid": "a10_qa_pairs_cc_news.csv",
    "qa_livedoor_a02_20_llm": "a02_qa_pairs_livedoor.csv",
    "qa_livedoor_a03_rule": "a03_qa_pairs_livedoor.csv",
    "qa_livedoor_a10_hybrid": "a10_qa_pairs_livedoor.csv",
}


# ===================================================================
# ユーティリティ関数
# ===================================================================

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


# ===================================================================
# Qdrantヘルスチェッカー
# ===================================================================

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


# ===================================================================
# Qdrantデータフェッチャー
# ===================================================================

class QdrantDataFetcher:
    """Qdrantからデータを取得"""

    def __init__(self, client: QdrantClient):
        self.client = client

    def fetch_collections(self) -> pd.DataFrame:
        """コレクション一覧を取得"""
        try:
            collections = self.client.get_collections()

            data = []
            for collection in collections.collections:
                try:
                    info = self.client.get_collection(collection.name)
                    data.append(
                        {
                            "Collection": collection.name,
                            "Vectors Count": info.vectors_count,
                            "Points Count": info.points_count,
                            "Indexed Vectors": info.indexed_vectors_count,
                            "Status": info.status,
                        }
                    )
                except Exception:
                    data.append(
                        {
                            "Collection": collection.name,
                            "Vectors Count": "N/A",
                            "Points Count": "N/A",
                            "Indexed Vectors": "N/A",
                            "Status": "Error",
                        }
                    )

            return (
                pd.DataFrame(data)
                if data
                else pd.DataFrame({"Info": ["No collections found"]})
            )

        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})

    def fetch_collection_points(
        self, collection_name: str, limit: int = 50
    ) -> pd.DataFrame:
        """コレクションの詳細データを取得"""
        try:
            # スクロールを使ってポイントを取得
            points_result = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            points = points_result[0]  # scrollは (points, next_offset) のタプルを返す

            if not points:
                return pd.DataFrame({"Info": ["No points found in collection"]})

            # ポイントをDataFrameに変換
            data = []
            for point in points:
                row = {"ID": point.id}

                # payloadの各フィールドを列として追加
                if point.payload:
                    for key, value in point.payload.items():
                        # 長すぎる文字列は切り詰め
                        if isinstance(value, str) and len(value) > 200:
                            row[key] = value[:200] + "..."
                        elif isinstance(value, (list, dict)):
                            row[key] = (
                                str(value)[:200] + "..."
                                if len(str(value)) > 200
                                else str(value)
                            )
                        else:
                            row[key] = value

                data.append(row)

            return pd.DataFrame(data)

        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})

    def fetch_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """コレクションの詳細情報を取得"""
        try:
            collection_info = self.client.get_collection(collection_name)

            # configの構造を安全にアクセス
            vector_config = collection_info.config.params.vectors

            # vector_configの型を判定して適切に処理
            if hasattr(vector_config, "size"):
                # 単一ベクトル設定
                vector_size = vector_config.size
                distance = vector_config.distance
            elif hasattr(vector_config, "__iter__"):
                # Named vectors設定の場合
                vector_sizes = {}
                distances = {}
                for name, config in (
                    vector_config.items() if isinstance(vector_config, dict) else []
                ):
                    vector_sizes[name] = (
                        config.size if hasattr(config, "size") else "N/A"
                    )
                    distances[name] = (
                        config.distance if hasattr(config, "distance") else "N/A"
                    )
                vector_size = vector_sizes if vector_sizes else "N/A"
                distance = distances if distances else "N/A"
            else:
                vector_size = "N/A"
                distance = "N/A"

            return {
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "indexed_vectors": collection_info.indexed_vectors_count,
                "status": collection_info.status,
                "config": {
                    "vector_size": vector_size,
                    "distance": distance,
                },
            }
        except Exception as e:
            return {"error": str(e)}

    def fetch_collection_source_info(
        self, collection_name: str, sample_size: int = 200
    ) -> Dict[str, Any]:
        """コレクションのデータソース情報を取得"""
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

            if not points:
                return {"total_points": total_points, "sources": {}, "sample_size": 0}

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
                ratio = stats["sample_count"] / sample_total
                stats["estimated_total"] = int(total_points * ratio)
                stats["percentage"] = ratio * 100

            return {
                "total_points": total_points,
                "sources": source_stats,
                "sample_size": sample_total,
            }

        except Exception as e:
            return {"error": str(e)}


# ===================================================================
# コレクション管理関数
# ===================================================================

def get_collection_stats(
    client: QdrantClient, collection_name: str
) -> Optional[Dict[str, Any]]:
    """コレクションの統計情報を取得"""
    try:
        collection_info = client.get_collection(collection_name)
        total_points = collection_info.points_count

        # ベクトル設定情報を取得
        vectors_config = collection_info.config.params.vectors
        vector_info = {}

        if isinstance(vectors_config, dict):
            # Named Vectors
            for name, config in vectors_config.items():
                vector_info[name] = {
                    "size": config.size,
                    "distance": str(config.distance),
                }
        elif hasattr(vectors_config, "size"):
            # Single Vector
            vector_info["default"] = {
                "size": vectors_config.size,
                "distance": str(vectors_config.distance),
            }

        return {
            "total_points": total_points,
            "vector_config": vector_info,
            "status": collection_info.status,
        }

    except UnexpectedResponse as e:
        if "doesn't exist" in str(e) or "not found" in str(e).lower():
            return None
        raise
    except Exception as e:
        logger.error(f"統計情報取得エラー: {e}")
        return None


def get_all_collections(client: QdrantClient) -> List[Dict[str, Any]]:
    """全コレクションの情報を取得"""
    collections = client.get_collections()
    collection_list = []

    for collection in collections.collections:
        try:
            info = client.get_collection(collection.name)
            collection_list.append(
                {
                    "name": collection.name,
                    "points_count": info.points_count,
                    "status": info.status,
                }
            )
        except Exception:
            collection_list.append(
                {"name": collection.name, "points_count": 0, "status": "unknown"}
            )

    return collection_list


def delete_all_collections(client: QdrantClient, excluded: List[str] = None) -> int:
    """全コレクションを削除"""
    excluded = excluded or []
    collections = get_all_collections(client)

    if not collections:
        return 0

    to_delete = [c for c in collections if c["name"] not in excluded]

    if not to_delete:
        return 0

    deleted_count = 0
    failed_count = 0

    for col in to_delete:
        try:
            client.delete_collection(collection_name=col["name"])
            deleted_count += 1
        except Exception as e:
            logger.error(f"コレクション削除エラー {col['name']}: {e}")
            failed_count += 1

    return deleted_count


# ===================================================================
# データ処理・登録関数
# ===================================================================

def load_csv_for_qdrant(
    path: str, required=("question", "answer"), limit: int = 0
) -> pd.DataFrame:
    """CSVをロード（Qdrant登録用）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)

    # 列名マッピング
    column_mappings = {
        "Question": "question",
        "Response": "answer",
        "Answer": "answer",
        "correct_answer": "answer",
    }
    df = df.rename(columns=column_mappings)

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"{path} には '{col}' 列が必要です（列: {list(df.columns)}）"
            )

    df = df.fillna("").drop_duplicates(subset=list(required)).reset_index(drop=True)

    if limit and limit > 0:
        df = df.head(limit).copy()

    return df


def build_inputs_for_embedding(df: pd.DataFrame, include_answer: bool) -> List[str]:
    """埋め込み用入力テキストを構築"""
    if include_answer:
        return (df["question"].astype(str) + "\n" + df["answer"].astype(str)).tolist()
    return df["question"].astype(str).tolist()


def embed_texts_for_qdrant(
    texts: List[str], model: str = "gemini-embedding-001", batch_size: int = 128
) -> List[List[float]]:
    """テキストをバッチ処理でEmbeddingに変換（Gemini API使用）"""
    # Gemini Embeddingクライアントを使用
    embedding_client = create_embedding_client(provider="gemini")
    dims = get_embedding_dimensions("gemini")  # 3072

    # 空文字列・空白のみの文字列を除外
    valid_texts = []
    valid_indices = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_texts.append(text)
            valid_indices.append(i)

    if not valid_texts:
        logger.warning("全てのテキストが空文字列です。ダミーベクトルを返します。")
        return [[0.0] * dims] * len(texts)

    # Gemini Embeddingでバッチ処理
    valid_vecs = embedding_client.embed_texts(valid_texts, batch_size=batch_size)

    # 元のインデックスに合わせてベクトルを再配置
    vecs: List[List[float]] = []
    valid_vec_idx = 0
    for i in range(len(texts)):
        if i in valid_indices:
            vecs.append(valid_vecs[valid_vec_idx])
            valid_vec_idx += 1
        else:
            vecs.append([0.0] * dims)

    return vecs


def create_or_recreate_collection_for_qdrant(
    client: QdrantClient, name: str, recreate: bool, vector_size: int = 3072
):
    """コレクション作成または再作成"""
    vectors_config = models.VectorParams(
        size=vector_size, distance=models.Distance.COSINE
    )

    if recreate:
        try:
            client.delete_collection(collection_name=name)
        except Exception:
            pass
        client.create_collection(collection_name=name, vectors_config=vectors_config)
    else:
        try:
            client.get_collection(name)
        except Exception:
            client.create_collection(
                collection_name=name, vectors_config=vectors_config
            )

    # ペイロード索引を作成
    try:
        client.create_payload_index(
            name, field_name="domain", field_schema=models.PayloadSchemaType.KEYWORD
        )
    except Exception:
        pass


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

        pid = abs(hash(f"{domain}-{source_file}-{i}")) & 0x7FFFFFFFFFFFFFFF
        points.append(models.PointStruct(id=pid, vector=vectors[i], payload=payload))

    return points


def upsert_points_to_qdrant(
    client: QdrantClient,
    collection: str,
    points: List[models.PointStruct],
    batch_size: int = 128,
) -> int:
    """ポイントをQdrantにアップサート"""
    count = 0
    for chunk in batched(points, batch_size):
        client.upsert(collection_name=collection, points=chunk)
        count += len(chunk)
    return count


# ===================================================================
# 検索関数
# ===================================================================

def embed_query_for_search(
    query: str, model: str = "gemini-embedding-001", dims: Optional[int] = None
) -> List[float]:
    """検索クエリをベクトル化（Gemini API使用）"""
    embedding_client = create_embedding_client(provider="gemini")
    return embedding_client.embed_text(query)


# ===================================================================
# コレクション統合関数
# ===================================================================

def scroll_all_points_with_vectors(
    client: QdrantClient,
    collection_name: str,
    batch_size: int = 100,
    progress_callback: Optional[callable] = None,
) -> List[models.Record]:
    """コレクションから全ポイント（ベクトル含む）を取得

    Args:
        client: QdrantClient
        collection_name: コレクション名
        batch_size: 1回のスクロールで取得する件数
        progress_callback: 進捗コールバック (取得済み件数, 総件数)

    Returns:
        全ポイントのリスト
    """
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
            with_vectors=True,
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


def merge_collections(
    client: QdrantClient,
    source_collections: List[str],
    target_collection: str,
    recreate: bool = True,
    vector_size: int = 3072,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """複数コレクションを統合して新コレクションに登録

    Args:
        client: QdrantClient
        source_collections: 統合元コレクション名のリスト
        target_collection: 統合先コレクション名
        recreate: 既存コレクションを削除して再作成するか
        vector_size: ベクトルサイズ
        progress_callback: 進捗コールバック (メッセージ, 現在値, 最大値)

    Returns:
        統合結果の辞書
    """
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

            # ポイントを取得
            points = scroll_all_points_with_vectors(client, src_collection)
            result["points_per_collection"][src_collection] = len(points)

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
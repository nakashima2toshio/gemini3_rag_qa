#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
celery_tasks.py - Celery非同期タスク定義
=========================================
Q/Aペア生成の並列処理のためのCeleryタスク定義
"""

import os
import json
import logging
from typing import List, Dict, Optional
from celery import Celery
from dotenv import load_dotenv
# 環境変数読み込み
load_dotenv()

# 共通モジュールからインポート
from models import QAPairsResponse
from config import ModelConfig, CeleryConfig

# =====================================================
# Gemini 3 Migration: 抽象化レイヤー
# =====================================================
from helper_llm import create_llm_client, LLMClient

# デフォルトプロバイダー（環境変数で設定可能）
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "openai"

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Celeryアプリケーション設定
app = Celery(
    'qa_generation',
    broker=os.getenv('CELERY_BROKER_URL', CeleryConfig.BROKER_URL),
    backend=os.getenv('CELERY_RESULT_BACKEND', CeleryConfig.RESULT_BACKEND)
)

# Celery設定
app.conf.update(
    task_serializer=CeleryConfig.TASK_SERIALIZER,
    accept_content=CeleryConfig.ACCEPT_CONTENT,
    result_serializer=CeleryConfig.RESULT_SERIALIZER,
    timezone=CeleryConfig.TIMEZONE,
    enable_utc=CeleryConfig.ENABLE_UTC,
    # タスクのタイムアウト設定
    task_time_limit=CeleryConfig.TASK_TIME_LIMIT,
    task_soft_time_limit=CeleryConfig.TASK_SOFT_TIME_LIMIT,
    # 並列度の制御
    worker_concurrency=CeleryConfig.WORKER_CONCURRENCY,
    worker_prefetch_multiplier=CeleryConfig.WORKER_PREFETCH_MULTIPLIER,
    # リトライ設定
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)


# ===========================================
# モデル別パラメータ制約（config.pyから参照）
# ===========================================
def supports_temperature(model: str) -> bool:
    """モデルがtemperatureパラメータをサポートするかチェック"""
    return ModelConfig.supports_temperature(model)


def determine_qa_count(chunk_data: Dict, config: Dict) -> int:
    """
    チャンクのトークン数に基づいてQ/A数を決定

    Args:
        chunk_data: チャンクデータ
        config: 設定情報

    Returns:
        生成するQ/A数
    """
    tokens = chunk_data.get('tokens', 100)
    base_qa_count = config.get('qa_per_chunk', 2)

    # トークン数に応じて調整
    if tokens < 50:
        return max(1, base_qa_count - 1)
    elif tokens > 150:
        return base_qa_count + 1
    else:
        return base_qa_count


def _extract_parsed_response(response, model: str) -> QAPairsResponse:
    """
    responses.parse() API のレスポンスから解析済みデータを抽出

    Args:
        response: OpenAI API レスポンス
        model: 使用したモデル名

    Returns:
        QAPairsResponse オブジェクト

    Raises:
        ValueError: 解析可能なレスポンスがない場合
    """
    parsed_response = None

    # デバッグ: レスポンスの構造をログ出力
    logger.info(f"レスポンスタイプ: {type(response).__name__}")
    response_attrs = [attr for attr in dir(response) if not attr.startswith('_')]
    logger.info(f"レスポンス属性: {response_attrs}")

    # 方法1: output_parsed属性を直接確認（GPT-5シリーズ対応）
    if hasattr(response, 'output_parsed') and response.output_parsed:
        parsed_response = response.output_parsed
        logger.info("GPT-5形式のレスポンス（output_parsed）を使用")
        return parsed_response

    # 方法2: output配列から探索（GPT-4o対応）
    if hasattr(response, 'output'):
        logger.info(f"output配列の長さ: {len(response.output)}")
        for idx, output in enumerate(response.output):
            output_type = getattr(output, 'type', 'N/A')
            logger.info(f"output[{idx}] タイプ: {type(output).__name__}, type属性: {output_type}")

            # ReasoningItemはスキップ
            if hasattr(output, 'type'):
                if output.type == "reasoning":
                    logger.info("  -> reasoning タイプをスキップ")
                    continue
                elif output.type == "message" and hasattr(output, 'content') and output.content:
                    logger.info(f"  -> message タイプ、content長さ: {len(output.content)}")
                    for item_idx, item in enumerate(output.content):
                        item_type = getattr(item, 'type', 'N/A')
                        logger.info(f"    content[{item_idx}] タイプ: {type(item).__name__}, type属性: {item_type}")

                        # parsed属性をチェック
                        if hasattr(item, 'type') and item.type == "output_text" and hasattr(item, 'parsed') and item.parsed:
                            parsed_response = item.parsed
                            logger.info("GPT-4o形式のレスポンス（output配列 -> parsed）を使用")
                            return parsed_response

                        # text属性がある場合もログ出力
                        if hasattr(item, 'text') and item.text:
                            text_preview = item.text[:100] if len(item.text) > 100 else item.text
                            logger.info(f"    text属性あり（最初の100文字）: {text_preview}...")

            if parsed_response:
                break

    # 方法3: 直接text属性からJSON解析を試みる
    if not parsed_response and hasattr(response, 'output'):
        logger.info("方法3: text属性からJSON直接解析を試行")
        for output in response.output:
            if hasattr(output, 'type') and output.type == "message" and hasattr(output, 'content'):
                for item in output.content:
                    if hasattr(item, 'text') and item.text:
                        try:
                            json_data = json.loads(item.text)
                            if 'qa_pairs' in json_data:
                                # JSONからQAPairsResponseを構築
                                parsed_response = QAPairsResponse(**json_data)
                                logger.info("JSONテキストから直接解析成功")
                                return parsed_response
                        except json.JSONDecodeError as e:
                            logger.debug(f"JSON解析失敗（JSONDecodeError）: {str(e)[:100]}")
                        except Exception as e:
                            logger.debug(f"JSON解析失敗（その他）: {str(e)[:100]}")
            if parsed_response:
                break

    # 方法4: output_text属性を直接確認
    if not parsed_response and hasattr(response, 'output_text') and response.output_text:
        logger.info("方法4: output_text属性からJSON解析を試行")
        try:
            json_data = json.loads(response.output_text)
            if 'qa_pairs' in json_data:
                parsed_response = QAPairsResponse(**json_data)
                logger.info("output_textから直接解析成功")
                return parsed_response
        except Exception as e:
            logger.debug(f"output_text解析失敗: {str(e)[:100]}")

    if not parsed_response:
        logger.error("OpenAI APIが解析可能なレスポンスを返しませんでした")
        # レスポンスの詳細をログ出力（デバッグ用）
        try:
            response_str = str(response)[:1000]
            logger.error(f"レスポンス全体（最初の1000文字）: {response_str}")
        except Exception:
            pass
        raise ValueError("No parsable response from OpenAI API")

    return parsed_response


@app.task(bind=True, max_retries=3)
def generate_qa_for_chunk_async(self, chunk_data: Dict, config: Dict, model: str = "gemini-2.0-flash") -> Dict:
    """
    単一チャンクからQ/Aペアを非同期生成（Celeryタスク）

    ※ 後方互換性のためのラッパー関数。内部でgenerate_qa_unified_asyncを呼び出す。

    Args:
        chunk_data: チャンクデータ
        config: データセット設定
        model: 使用するモデル（デフォルト: gemini-2.0-flash）

    Returns:
        生成されたQ/Aペアと関連情報を含む辞書
    """
    logger.info(f"[後方互換ラッパー] generate_qa_for_chunk_async -> generate_qa_unified_async")
    # 統合タスクに委譲（Geminiを使用）
    return generate_qa_unified_async(chunk_data, config, model=None, provider="gemini")


@app.task(bind=True, max_retries=3)
def generate_qa_for_batch_async(self, chunks: List[Dict], config: Dict, model: str = "gemini-2.0-flash") -> Dict:
    """
    複数チャンクからQ/Aペアを非同期バッチ生成（Celeryタスク）

    ※ 後方互換性のためのラッパー関数。各チャンクに対してgenerate_qa_unified_asyncを呼び出す。

    Args:
        chunks: チャンクデータのリスト（1-5個）
        config: データセット設定
        model: 使用するモデル（デフォルト: gemini-2.0-flash）

    Returns:
        生成されたQ/Aペアと関連情報を含む辞書
    """
    logger.info(f"[後方互換ラッパー] generate_qa_for_batch_async -> generate_qa_unified_async (複数チャンク)")

    chunk_ids = [c.get('id', 'unknown') for c in chunks]
    all_qa_pairs = []

    try:
        # 各チャンクに対して統合タスクを実行
        for chunk in chunks:
            result = generate_qa_unified_async(chunk, config, model=None, provider="gemini")
            if result.get('success'):
                all_qa_pairs.extend(result.get('qa_pairs', []))

        logger.info(f"[後方互換バッチ] 完了: {len(chunks)}チャンク - {len(all_qa_pairs)}個のQ/A生成")

        return {
            "success": True,
            "chunk_ids": chunk_ids,
            "qa_pairs": all_qa_pairs,
            "error": None
        }

    except Exception as e:
        logger.error(f"[後方互換バッチ] エラー: {str(e)}")

        # リトライ処理
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=5 * (self.request.retries + 1))

        return {
            "success": False,
            "chunk_ids": chunk_ids,
            "qa_pairs": all_qa_pairs,
            "error": str(e)
        }


# =====================================================
# Gemini 3 Migration: 統合Q/A生成タスク
# =====================================================

@app.task(bind=True, max_retries=3)
def generate_qa_unified_async(
    self,
    chunk_data: Dict,
    config: Dict,
    model: str = None,
    provider: str = None
) -> Dict:
    """
    単一チャンクからQ/Aペアを非同期生成（統合版: Gemini/OpenAI対応）

    Gemini 3 Migration: プロバイダーに応じてGeminiまたはOpenAIを使用

    Args:
        chunk_data: チャンクデータ
        config: データセット設定
        model: 使用するモデル（Noneの場合はプロバイダーのデフォルト）
        provider: "gemini" or "openai"（Noneの場合はDEFAULT_LLM_PROVIDER）

    Returns:
        生成されたQ/Aペアと関連情報を含む辞書
    """
    try:
        provider = provider or DEFAULT_LLM_PROVIDER

        # Geminiプロバイダー使用時にOpenAIモデル名が渡された場合はデフォルトモデルを使用
        if provider == "gemini" and model and ("gpt" in model.lower() or "o1" in model.lower() or "o3" in model.lower() or "o4" in model.lower()):
            logger.warning(f"[統合タスク] OpenAIモデル '{model}' はGeminiプロバイダーで使用できません。デフォルトモデルを使用します。")
            model = None  # Noneにすることでデフォルトのgemini-2.0-flashを使用

        logger.info(f"[統合タスク] チャンク {chunk_data.get('id', 'unknown')}, プロバイダー: {provider}, モデル: {model or 'default'}")

        # Q/A数の決定
        num_pairs = determine_qa_count(chunk_data, config)
        lang = config["lang"]

        # プロンプト設定（言語別）
        if lang == "ja":
            system_instruction = """あなたは教育コンテンツ作成の専門家です。
与えられた日本語テキストから、学習効果の高いQ&Aペアを生成してください。
質問は明確で具体的に、回答は簡潔で正確に（1-2文程度）。"""

            prompt = f"""以下のテキストから{num_pairs}個のQ&Aペアを生成してください。

質問タイプ: fact（事実確認）, reason（理由説明）, comparison（比較）, application（応用）

テキスト:
{chunk_data['text']}

JSON形式で出力:
{{"qa_pairs": [{{"question": "質問文", "answer": "回答文", "question_type": "fact/reason/comparison/application"}}]}}"""
        else:
            system_instruction = """You are an expert in educational content creation.
Generate high-quality Q&A pairs from the given English text.
Questions should be clear and specific, answers concise and accurate (1-2 sentences)."""

            prompt = f"""Generate {num_pairs} Q&A pairs from the following text.

Question types: fact, reason, comparison, application

Text:
{chunk_data['text']}

Output in JSON format:
{{"qa_pairs": [{{"question": "question text", "answer": "answer text", "question_type": "fact/reason/comparison/application"}}]}}"""

        # 統合LLMクライアントを使用
        llm_client = create_llm_client(provider=provider)

        # 構造化出力を試行
        try:
            result = llm_client.generate_structured(
                prompt=f"{system_instruction}\n\n{prompt}",
                response_schema=QAPairsResponse,
                model=model
            )

            qa_pairs = []
            for qa_data in result.qa_pairs:
                qa = {
                    "question": qa_data.question,
                    "answer": qa_data.answer,
                    "question_type": qa_data.question_type,
                    "source_chunk_id": chunk_data.get('id', ''),
                    "doc_id": chunk_data.get('doc_id', ''),
                    "dataset_type": chunk_data.get('dataset_type', ''),
                    "chunk_idx": chunk_data.get('chunk_idx', 0),
                    "provider": provider  # 使用プロバイダーを記録
                }
                qa_pairs.append(qa)

        except Exception as e:
            logger.warning(f"構造化出力失敗、テキスト生成にフォールバック: {str(e)[:100]}")

            # フォールバック: テキスト生成してJSON解析
            response_text = llm_client.generate_content(
                prompt=f"{system_instruction}\n\n{prompt}",
                model=model
            )

            # JSONを抽出して解析
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                qa_pairs = []
                for qa_data in parsed_data.get('qa_pairs', []):
                    qa = {
                        "question": qa_data.get('question', ''),
                        "answer": qa_data.get('answer', ''),
                        "question_type": qa_data.get('question_type', 'fact'),
                        "source_chunk_id": chunk_data.get('id', ''),
                        "doc_id": chunk_data.get('doc_id', ''),
                        "dataset_type": chunk_data.get('dataset_type', ''),
                        "chunk_idx": chunk_data.get('chunk_idx', 0),
                        "provider": provider
                    }
                    qa_pairs.append(qa)
            else:
                raise ValueError("JSON not found in response")

        logger.info(f"[統合タスク] 完了: {len(qa_pairs)}個のQ/A生成")

        return {
            "success": True,
            "chunk_id": chunk_data.get('id'),
            "qa_pairs": qa_pairs,
            "provider": provider,
            "error": None
        }

    except Exception as e:
        logger.error(f"[統合タスク] エラー: {str(e)}")

        # リトライ処理
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=5 * (self.request.retries + 1))

        return {
            "success": False,
            "chunk_id": chunk_data.get('id'),
            "qa_pairs": [],
            "provider": provider or DEFAULT_LLM_PROVIDER,
            "error": str(e)
        }


def submit_unified_qa_generation(
    chunks: List[Dict],
    config: Dict,
    model: str = None,
    provider: str = None
) -> List:
    """
    統合Q/A生成ジョブを投入（Gemini/OpenAI対応）

    Gemini 3 Migration: プロバイダーを選択可能

    Args:
        chunks: チャンクのリスト
        config: データセット設定
        model: 使用するモデル（Noneの場合はプロバイダーのデフォルト）
        provider: "gemini" or "openai"（Noneの場合はDEFAULT_LLM_PROVIDER）

    Returns:
        Celeryタスクのリスト

    Example:
        # Gemini使用（デフォルト）
        tasks = submit_unified_qa_generation(chunks, config)

        # OpenAI使用
        tasks = submit_unified_qa_generation(chunks, config, provider="openai")
    """
    provider = provider or DEFAULT_LLM_PROVIDER
    tasks = []

    for chunk in chunks:
        task = generate_qa_unified_async.apply_async(
            args=[chunk, config, model, provider],
            queue='qa_generation'
        )
        tasks.append(task)

    logger.info(f"[統合Q/A生成] 投入タスク数: {len(tasks)}, プロバイダー: {provider}")
    return tasks


def submit_parallel_qa_generation(chunks: List[Dict], config: Dict, model: str = "gemini-2.0-flash",
                                 batch_size: int = 3) -> List:
    """
    並列Q/A生成ジョブを投入

    Args:
        chunks: チャンクのリスト
        config: データセット設定
        model: 使用するモデル
        batch_size: バッチサイズ（1-5）

    Returns:
        Celeryタスクのリスト
    """
    tasks = []

    # バッチ処理の場合
    if batch_size > 1:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            task = generate_qa_for_batch_async.apply_async(
                args=[batch, config, model],
                queue='qa_generation'  # ワーカーが監視しているキューを指定
            )
            tasks.append(task)
            logger.debug(f"タスク投入: {task.id} - {len(batch)}チャンク")
    else:
        # 個別処理の場合
        for chunk in chunks:
            task = generate_qa_for_chunk_async.apply_async(
                args=[chunk, config, model],
                queue='qa_generation'  # ワーカーが監視しているキューを指定
            )
            tasks.append(task)

    logger.info(f"投入されたタスク数: {len(tasks)}")
    return tasks


def collect_results(tasks: List, timeout: int = 300) -> List[Dict]:
    """
    並列処理の結果を収集（改良版：Redis直接アクセスで確実な結果取得）

    Args:
        tasks: Celeryタスクのリスト
        timeout: タイムアウト（秒）

    Returns:
        Q/Aペアのリスト
    """
    import time
    import redis
    import json
    from celery.result import AsyncResult

    all_qa_pairs = []
    failed_chunks = []
    total_tasks = len(tasks)
    completed_tasks = set()  # 完了済みタスクのインデックス
    failed_tasks = set()     # 失敗タスクのインデックス

    logger.info(f"結果収集開始: {total_tasks}個のタスク")

    # Redis接続を確立
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=int(os.getenv('REDIS_DB', 0)),
        decode_responses=True
    )

    # タスクIDをログ出力
    for i, task in enumerate(tasks):
        logger.debug(f"タスク {i+1}: ID={task.id}")

    start_time = time.time()
    last_log_time = start_time
    stall_check_time = start_time
    last_completed_count = 0
    stall_counter = 0

    # シンプルなループで結果を収集
    while len(completed_tasks) + len(failed_tasks) < total_tasks:
        current_time = time.time()
        elapsed = current_time - start_time

        # タイムアウトチェック
        if elapsed > timeout:
            logger.error(f"タイムアウト: {elapsed:.1f}秒経過")
            logger.info(f"収集済み: {len(completed_tasks)}個, 未収集: {total_tasks - len(completed_tasks) - len(failed_tasks)}個")
            break

        # 5秒ごとに進捗表示
        if current_time - last_log_time >= 5:
            logger.info(f"進捗: 完了={len(completed_tasks)}/{total_tasks}, "
                       f"失敗={len(failed_tasks)}, "
                       f"処理中={total_tasks - len(completed_tasks) - len(failed_tasks)}, "
                       f"経過時間={elapsed:.1f}秒")
            last_log_time = current_time

        # 30秒ごとに停滞チェック
        if current_time - stall_check_time >= 30:
            current_completed = len(completed_tasks)
            if current_completed == last_completed_count:
                stall_counter += 1
                logger.warning(f"⚠️ 処理が停滞している可能性があります（{stall_counter * 30}秒間進捗なし）")

                # 3分間進捗がない場合、状態を強制リフレッシュ
                if stall_counter >= 6:
                    logger.warning("📊 タスク状態を強制的に再取得...")
                    for i, task in enumerate(tasks):
                        if i not in completed_tasks and i not in failed_tasks:
                            try:
                                # AsyncResultで再取得
                                refreshed_task = AsyncResult(task.id)
                                if refreshed_task.state in ['SUCCESS', 'FAILURE']:
                                    logger.info(f"タスク {i+1} の状態: {refreshed_task.state}")
                            except Exception as e:
                                logger.debug(f"タスク {i+1} の状態取得エラー: {str(e)}")
                    stall_counter = 0
            else:
                stall_counter = 0
                last_completed_count = current_completed
            stall_check_time = current_time

        # 各タスクをチェック
        pending_indices = []
        for i, task in enumerate(tasks):
            # 既に処理済みのタスクはスキップ
            if i in completed_tasks or i in failed_tasks:
                continue

            try:
                # まずRedisから直接状態を確認（最も確実な方法）
                redis_key = f"celery-task-meta-{task.id}"
                redis_data = redis_client.get(redis_key)

                result = None
                redis_state = None

                if redis_data:
                    try:
                        redis_result = json.loads(redis_data)
                        redis_state = redis_result.get('status')

                        # Redisで完了している場合は結果を直接取得
                        if redis_state == 'SUCCESS':
                            result = redis_result.get('result')
                            if result and isinstance(result, dict):
                                if result.get('success'):
                                    # Redis直接取得成功
                                    logger.debug(f"タスク {i+1} Redis直接取得成功")
                                else:
                                    # タスクは完了したが、結果がsuccess=False（エラー）
                                    logger.debug(f"タスク {i+1} Redis直接取得（失敗結果）")
                        elif redis_state == 'FAILURE':
                            # Celeryタスク自体が失敗
                            failed_tasks.add(i)
                            logger.error(f"✗ タスク {i+1}/{total_tasks} 失敗（Redis FAILURE状態）")
                            continue
                    except json.JSONDecodeError:
                        logger.warning(f"タスク {i+1} Redis JSONデコードエラー")

                # Redisから結果を取得できた場合は処理
                if result and isinstance(result, dict):
                    if result.get('success'):
                        # 成功タスク
                        qa_pairs = result.get('qa_pairs', [])
                        all_qa_pairs.extend(qa_pairs)
                        completed_tasks.add(i)

                        # チャンク情報をログ
                        if 'chunk_id' in result:
                            logger.info(f"✓ タスク {i+1}/{total_tasks} 完了（Redis直接）: チャンク {result['chunk_id']} - {len(qa_pairs)}個のQ/A")
                        elif 'chunk_ids' in result:
                            logger.info(f"✓ タスク {i+1}/{total_tasks} 完了（Redis直接）: バッチ {len(result['chunk_ids'])}チャンク - {len(qa_pairs)}個のQ/A")
                    else:
                        # 失敗タスク（success=False）- Redisから直接検出
                        failed_tasks.add(i)
                        error_msg = result.get('error', 'Unknown error')
                        logger.error(f"✗ タスク {i+1}/{total_tasks} 失敗（Redis直接）: {error_msg[:200]}")

                        if 'chunk_id' in result:
                            failed_chunks.append(result['chunk_id'])
                        elif 'chunk_ids' in result:
                            failed_chunks.extend(result['chunk_ids'])
                else:
                    # Redisから結果を取得できなかった場合は従来の方法を試す
                    # (redis_dataがない、redis_stateがPENDING/STARTED、resultがNone等)
                    task_state = task.state

                    # デバッグ: 最後のタスクの状態を詳細ログ
                    if i == total_tasks - 1:
                        logger.debug(f"🔍 最後のタスク[{i}] Redis状態={redis_state}, Celery状態={task_state}, id={task.id[:8]}...")

                    # Celery経由で結果を取得
                    if task_state == 'SUCCESS' or task.ready():
                        # 結果を取得（短いタイムアウト）
                        try:
                            result = task.get(timeout=1, propagate=False)
                        except Exception:
                            pass

                        if result and isinstance(result, dict) and result.get('success'):
                            qa_pairs = result.get('qa_pairs', [])
                            all_qa_pairs.extend(qa_pairs)
                            completed_tasks.add(i)

                            # チャンク情報をログ
                            if 'chunk_id' in result:
                                logger.info(f"✓ タスク {i+1}/{total_tasks} 完了: チャンク {result['chunk_id']} - {len(qa_pairs)}個のQ/A")
                            elif 'chunk_ids' in result:
                                logger.info(f"✓ タスク {i+1}/{total_tasks} 完了: バッチ {len(result['chunk_ids'])}チャンク - {len(qa_pairs)}個のQ/A")
                        elif result is not None:
                            failed_tasks.add(i)
                            error_msg = result.get('error', 'Unknown error') if result and isinstance(result, dict) else str(result)
                            logger.error(f"✗ タスク {i+1}/{total_tasks} 失敗: {error_msg[:200]}")

                            if result and isinstance(result, dict):
                                if 'chunk_id' in result:
                                    failed_chunks.append(result['chunk_id'])
                                elif 'chunk_ids' in result:
                                    failed_chunks.extend(result['chunk_ids'])

                    elif task_state == 'FAILURE':
                        failed_tasks.add(i)
                        logger.error(f"✗ タスク {i+1}/{total_tasks} 失敗（状態: FAILURE）")

                    elif task_state == 'PENDING':
                        pending_indices.append(i)

            except TimeoutError:
                # タイムアウトは正常（まだ処理中）
                pass
            except Exception as e:
                # その他のエラーは1回だけログ
                if i not in failed_tasks:
                    logger.debug(f"タスク {i+1} チェック中にエラー（処理継続）: {str(e)[:100]}")

        # デバッグ: ペンディング状態のタスクが多い場合警告
        if len(pending_indices) > 10 and current_time - start_time > 60:
            logger.debug(f"ペンディング状態のタスク数: {len(pending_indices)}")

        # 短い待機（処理済みタスクが多い場合は待機時間を長くする）
        completion_ratio = len(completed_tasks) / total_tasks if total_tasks > 0 else 0
        if completion_ratio > 0.9:
            time.sleep(1.0)  # 90%以上完了したら待機時間を長く
        else:
            time.sleep(0.5)

    # 最終的な未完了タスクの処理（より積極的に取得）
    logger.info(f"ループ終了後の最終確認: 未処理タスク {total_tasks - len(completed_tasks) - len(failed_tasks)}個")
    for i, task in enumerate(tasks):
        if i not in completed_tasks and i not in failed_tasks:
            try:
                # AsyncResultで強制的に最新状態を取得
                from celery.result import AsyncResult
                refreshed = AsyncResult(task.id)
                final_state = refreshed.state
                logger.info(f"タスク {i+1} 最終状態: {final_state}")

                # 状態に関わらず結果取得を試みる
                if final_state in ['SUCCESS', 'FAILURE'] or refreshed.ready():
                    result = refreshed.get(timeout=3, propagate=False)
                    if result and isinstance(result, dict) and result.get('success'):
                        qa_pairs = result.get('qa_pairs', [])
                        all_qa_pairs.extend(qa_pairs)
                        completed_tasks.add(i)
                        logger.info(f"✓ タスク {i+1}/{total_tasks} 最終収集で完了: {len(qa_pairs)}個のQ/A")
                    else:
                        failed_tasks.add(i)
                        logger.warning(f"タスク {i+1}/{total_tasks} 最終収集で失敗として処理")
                else:
                    logger.warning(f"タスク {i+1}/{total_tasks} 最終状態: {final_state} - 未完了")
            except Exception as e:
                logger.warning(f"タスク {i+1}/{total_tasks} 最終収集エラー: {str(e)[:100]}")

    # 結果サマリー
    elapsed_total = time.time() - start_time
    logger.info(f"""
    =====================================
    結果収集完了:
    - 成功: {len(completed_tasks)}/{total_tasks}タスク
    - 失敗: {len(failed_tasks)}タスク
    - 未完了: {total_tasks - len(completed_tasks) - len(failed_tasks)}タスク
    - 生成Q/Aペア: {len(all_qa_pairs)}個
    - 所要時間: {elapsed_total:.1f}秒
    =====================================
    """)

    if failed_chunks:
        logger.warning(f"失敗したチャンク（最初の5個）: {failed_chunks[:5]}")

    return all_qa_pairs


if __name__ == "__main__":
    # Celeryワーカーを起動する場合
    # celery -A celery_tasks worker --loglevel=info --concurrency=4
    pass
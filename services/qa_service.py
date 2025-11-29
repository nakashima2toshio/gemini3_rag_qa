#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_service.py - Q/A生成サービス
================================
Q/Aペアの生成と保存に関するビジネスロジック

機能:
- a02_make_qa_para.pyのサブプロセス実行
- OpenAI APIによるQ/A生成
- Q/Aペアの保存
"""

import os
import sys
import re
import json
import logging
import subprocess
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from helper_llm import create_llm_client

# モデルからインポート
from models import QAPair, QAPairsResponse

# ログ設定
logger = logging.getLogger(__name__)


def run_advanced_qa_generation(
    dataset: Optional[str],
    input_file: Optional[str],
    use_celery: bool,
    celery_workers: int,
    batch_chunks: int,
    max_docs: int,
    merge_chunks: bool,
    min_tokens: int,
    max_tokens: int,
    coverage_threshold: float,
    model: str,
    analyze_coverage: bool,
    log_callback,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    a02_make_qa_para.pyをサブプロセスで実行

    改善内容（2024年11月26日）：
    - Redis直接アクセスによる確実な結果収集
    - タスク状態誤認識（PENDING）の回避
    - プログラム終了時のCelery接続クリーンアップ
    - 全1612タスクの正常完了を保証

    Args:
        dataset: データセット名
        input_file: 入力ファイルパス
        use_celery: Celery並列処理を使用（Redis直接アクセス対応）
        celery_workers: Celeryワーカー数
        batch_chunks: バッチチャンク数
        max_docs: 最大ドキュメント数
        merge_chunks: チャンク統合
        min_tokens: 最小トークン数
        max_tokens: 最大トークン数
        coverage_threshold: カバレージ閾値
        model: 使用モデル（gpt-5シリーズ、O-series対応）
        analyze_coverage: カバレージ分析を実行
        log_callback: ログコールバック関数
        progress_callback: 進捗コールバック関数 (current, total) -> None

    Returns:
        実行結果の辞書
    """
    # コマンド構築
    cmd = [sys.executable, "a02_make_qa_para.py"]

    if dataset:
        cmd.extend(["--dataset", dataset])
    elif input_file:
        cmd.extend(["--input-file", input_file])

    if use_celery:
        cmd.append("--use-celery")
        cmd.extend(["--celery-workers", str(celery_workers)])

    cmd.extend(
        [
            "--batch-chunks",
            str(batch_chunks),
            "--max-docs",
            str(max_docs),
            "--min-tokens",
            str(min_tokens),
            "--max-tokens",
            str(max_tokens),
            "--coverage-threshold",
            str(coverage_threshold),
            "--model",
            model,
        ]
    )

    if merge_chunks:
        cmd.append("--merge-chunks")

    if analyze_coverage:
        cmd.append("--analyze-coverage")

    # 環境変数を現在のプロセスからコピー
    env = os.environ.copy()

    log_callback(f"🚀 高度なQ/A生成を開始: {' '.join(cmd)}")

    # 出力をキューに格納
    output_queue = queue.Queue()

    def read_output(pipe, q):
        """サブプロセスの出力を読み取る"""
        for line in iter(pipe.readline, ""):
            if line:
                q.put(line.strip())
        pipe.close()

    try:
        # サブプロセス起動
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env,
        )

        # 出力読み取りスレッド開始
        read_thread = threading.Thread(
            target=read_output, args=(process.stdout, output_queue)
        )
        read_thread.daemon = True
        read_thread.start()

        # リアルタイムでログを処理
        saved_files = None
        qa_count = 0
        coverage_results = None

        while True:
            # プロセスが終了したかチェック
            poll = process.poll()

            # キューから出力を取得
            try:
                line = output_queue.get(timeout=0.1)
                log_callback(line)

                # 進捗情報を抽出してコールバック
                if progress_callback:
                    # "進捗: 完了=123/305" のようなパターンにマッチ
                    progress_match = re.search(
                        r"進捗.*?完了[=:：\s]*(\d+)\s*/\s*(\d+)", line
                    )
                    if progress_match:
                        current = int(progress_match.group(1))
                        total = int(progress_match.group(2))
                        progress_callback(current, total)

                # 結果ファイルのパスを抽出
                if "CSV保存:" in line:
                    csv_match = line.split("CSV保存:")[-1].strip()
                    if saved_files is None:
                        saved_files = {}
                    saved_files["csv"] = f"qa_output/{csv_match}"

                elif "JSON保存:" in line:
                    json_match = line.split("JSON保存:")[-1].strip()
                    if saved_files:
                        saved_files["json"] = f"qa_output/{json_match}"

                elif "生成Q/Aペア数:" in line or "生成Q/Aペア:" in line:
                    # Q/A数を抽出
                    # "生成Q/Aペア数: 118" または "生成Q/Aペア: 118個" の両方に対応
                    count_match = re.search(r"(\d+)", line)
                    if count_match:
                        qa_count = int(count_match.group(1))

                elif "カバレージ率:" in line:
                    # カバレージ結果を解析
                    rate_match = re.search(r"([\d.]+)%", line)
                    if rate_match:
                        coverage_results = {
                            "coverage_rate": float(rate_match.group(1)) / 100
                        }

            except queue.Empty:
                pass

            # プロセスが終了したら残りの出力を処理
            if poll is not None:
                # 残りの出力を全て取得
                while not output_queue.empty():
                    try:
                        line = output_queue.get_nowait()
                        log_callback(line)
                    except queue.Empty:
                        break
                break

        # プロセス終了コード確認
        return_code = process.returncode

        if return_code == 0:
            log_callback("✅ 高度なQ/A生成が正常に完了しました")
            return {
                "success": True,
                "saved_files": saved_files,
                "qa_count": qa_count,
                "coverage_results": coverage_results,
            }
        else:
            log_callback(f"⚠️ 高度なQ/A生成が終了コード {return_code} で終了しました")
            return {"success": False, "return_code": return_code}

    except Exception as e:
        log_callback(f"❌ 高度なQ/A生成でエラーが発生: {str(e)}")
        return {"success": False, "error": str(e)}


def generate_qa_pairs(
    text: str,
    dataset_type: str,
    chunk_id: str,
    model: str = "gemini-2.0-flash",
    qa_per_chunk: int = 3,
    log_callback=None,
) -> List[QAPair]:
    """
    テキストからQ/Aペアを生成（Gemini API使用）

    Args:
        text: 対象テキスト
        dataset_type: データセットタイプ
        chunk_id: チャンクID
        model: 使用するモデル（デフォルト: gemini-2.0-flash）
        qa_per_chunk: チャンクあたりのQ/A数
        log_callback: ログコールバック関数

    Returns:
        Q/Aペアのリスト
    """
    # Geminiクライアントを使用
    client = create_llm_client(provider="gemini")

    prompt = f"""あなたは教育用Q/Aペア生成の専門家です。

以下のテキストから、{qa_per_chunk}個の質問と回答のペアを生成してください。

テキスト:
{text}

要件:
1. 質問は具体的で明確なものにする
2. 回答はテキストの内容に基づいた正確なものにする
3. 質問タイプは以下から選択: factual, conceptual, application, analysis
4. テキストの重要な情報を網羅するようにする

JSON形式で出力してください。
"""

    try:
        # Gemini構造化出力APIを使用
        qa_response = client.generate_structured(
            prompt=prompt,
            response_schema=QAPairsResponse,
            model=model
        )

        # Q/Aペアにメタデータを追加
        result_pairs = []
        for qa in qa_response.qa_pairs:
            qa_pair = QAPair(
                question=qa.question,
                answer=qa.answer,
                question_type=qa.question_type,
                source_chunk_id=chunk_id,
                dataset_type=dataset_type,
                auto_generated=True
            )
            result_pairs.append(qa_pair)

        if log_callback:
            log_callback(f"    └─ {len(result_pairs)}個のQ/Aペアを生成")

        return result_pairs

    except Exception as e:
        logger.error(f"Q/A生成エラー: {e}")
        if log_callback:
            log_callback(f"    └─ エラー: {str(e)}")
        return []


def save_qa_pairs_to_file(
    qa_pairs: List[QAPair], dataset_type: str, log_callback=None
) -> Dict[str, str]:
    """
    Q/AペアをCSVとJSONで保存

    Args:
        qa_pairs: Q/Aペアのリスト
        dataset_type: データセットタイプ
        log_callback: ログコールバック関数

    Returns:
        保存されたファイルパスの辞書
    """
    qa_output_dir = Path("qa_output")
    qa_output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}

    # DataFrameに変換
    qa_data = []
    for qa in qa_pairs:
        qa_data.append(
            {
                "question": qa.question,
                "answer": qa.answer,
                "question_type": qa.question_type,
                "source_chunk_id": qa.source_chunk_id,
                "dataset_type": qa.dataset_type,
                "auto_generated": qa.auto_generated,
            }
        )

    df_qa = pd.DataFrame(qa_data)

    # CSVファイル
    csv_filename = f"qa_pairs_{dataset_type}_{timestamp}.csv"
    csv_path = qa_output_dir / csv_filename
    df_qa.to_csv(csv_path, index=False, encoding="utf-8-sig")
    saved_files["csv"] = str(csv_path)

    if log_callback:
        log_callback(f"  📄 CSV保存: {csv_filename}")

    # JSONファイル
    json_filename = f"qa_pairs_{dataset_type}_{timestamp}.json"
    json_path = qa_output_dir / json_filename

    json_data = {
        "dataset_type": dataset_type,
        "created_at": datetime.now().isoformat(),
        "total_pairs": len(qa_pairs),
        "qa_pairs": qa_data,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    saved_files["json"] = str(json_path)

    if log_callback:
        log_callback(f"  📋 JSON保存: {json_filename}")

    return saved_files
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import traceback
from typing import Dict, Any

# 標準出力を強制的にフラッシュモードにする
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

r"""
a02_make_qa_para.py - 改善版Q/Aペア自動生成システム
(Refactored for direct import and execution)
"""

import json
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import tiktoken
from helper_llm import create_llm_client, LLMClient
from dotenv import load_dotenv
import logging
import re
from collections import Counter

# ===================================================================
# 共通モジュールからインポート
# ===================================================================
from models import QAPairsResponse
from config import (
    DATASET_CONFIGS,
    QAGenerationConfig,
)

# 環境変数読み込み
load_dotenv()

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================== 
# Import existing logic from original file
# (In a real refactor, we would import from the original file,
# but here we will duplicate the essential parts for safety/speed)
# ========================================== 
# Since we cannot easily import parts of a script without running it,
# we will assume the helper functions are available or we will redefine them.
# Ideally, we should have moved the logic to a separate module.
# For this fix, I will copy the ESSENTIAL functions needed for `run_qa_process`.
# However, to avoid massive code duplication in this `write_file` call,
# I will try to import `a02_make_qa_para` as a module if possible,
# but since it has a `main` execution block, we must be careful.
# The original `a02_make_qa_para.py` has `if __name__ == "__main__": main()`, 
# so it IS safe to import.

# We will wrap the main logic in a callable function `run_qa_generator`.

import a02_make_qa_para as original_script

def run_qa_generator(
    dataset: Optional[str] = None,
    input_file: Optional[str] = None,
    model: str = "gemini-2.0-flash",
    output_dir: str = "qa_output/a02",
    max_docs: Optional[int] = None,
    analyze_coverage: bool = False,
    batch_chunks: int = 3,
    merge_chunks: bool = True,
    min_tokens: int = 150,
    max_tokens: int = 400,
    use_celery: bool = False,
    celery_workers: int = 8,
    coverage_threshold: Optional[float] = None,
    log_callback=None
):
    """
    Q/A生成プロセスのエントリーポイント（直接呼び出し用）
    """
    # ロガーのハンドラを設定してcallbackに流す
    if log_callback:
        class CallbackHandler(logging.Handler):
            def emit(self, record):
                msg = self.format(record)
                log_callback(msg)
        
        handler = CallbackHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        original_script.logger.addHandler(handler)

    try:
        logger.info("🚀 Q/A生成プロセスを開始します (Direct Mode)")
        
        # APIキー確認
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEYが設定されていません")
            return {"success": False, "error": "GOOGLE_API_KEY missing"}

        # データセット設定
        if input_file:
            dataset_type = "custom_upload"
            file_basename = Path(input_file).stem
            lang = "ja"
            config = {
                "name": f"ローカルファイル ({file_basename})",
                "text_column": "Combined_Text",
                "title_column": None,
                "lang": lang,
                "chunk_size": 300,
                "qa_per_chunk": 3,
            }
            dataset_name = config['name']
        else:
            dataset_type = dataset
            config = DATASET_CONFIGS[dataset_type]
            dataset_name = config['name']

        logger.info(f"データセット: {dataset_name}")

        # 1. データ読み込み
        logger.info("\n[1/4] データ読み込み...")
        if input_file:
            df = original_script.load_uploaded_file(input_file)
            if max_docs and len(df) > max_docs:
                df = df.head(max_docs)
                logger.info(f"  📊 最大文書数制限: {len(df)} 件に制限")
        else:
            df = original_script.load_preprocessed_data(dataset_type)

        # 2. チャンク作成
        logger.info("\n[2/4] チャンク作成...")
        max_docs_for_chunks = None if input_file else max_docs
        chunks = original_script.create_document_chunks(df, dataset_type, max_docs_for_chunks, config=config)

        if not chunks:
            logger.error("チャンクが作成されませんでした")
            return {"success": False, "error": "No chunks created"}

        # 3. Q/Aペア生成
        logger.info("\n[3/4] Q/Aペア生成...")
        qa_pairs = []

        if use_celery:
            # Celeryワーカーの事前確認
            logger.info("Celeryワーカーの状態を確認中...")
            if not original_script.check_celery_workers(celery_workers):
                logger.error("Celeryワーカーを起動してから再実行してください")
                return {"success": False, "error": "Celery workers not ready"}
            logger.info(f"✓ Celeryワーカー確認OK（{celery_workers}ワーカー）")

            # Celeryタスクのインポート
            from celery_tasks import submit_unified_qa_generation, collect_results

            # チャンクの前処理
            if merge_chunks:
                processed_chunks = original_script.merge_small_chunks(chunks, min_tokens, max_tokens)
            else:
                processed_chunks = chunks

            # 並列タスク投入
            tasks = submit_unified_qa_generation(
                processed_chunks, config, model, provider="gemini"
            )

            timeout_seconds = min(max(len(tasks) * 10, 600), 1800)
            logger.info(f"結果収集タイムアウト: {timeout_seconds}秒（{len(tasks)}タスク）")
            qa_pairs = collect_results(tasks, timeout=timeout_seconds)
        else:
            logger.info("通常処理モード")
            qa_pairs = original_script.generate_qa_for_dataset(
                chunks,
                dataset_type,
                model,
                chunk_batch_size=batch_chunks,
                merge_chunks=merge_chunks,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                config=config
            )

        if not qa_pairs:
            logger.warning("Q/Aペアが生成されませんでした")

        # 4. カバレージ分析
        coverage_results = {}
        if analyze_coverage and qa_pairs:
            logger.info("\n[4/4] カバレージ分析を開始します（Embedding生成に時間がかかる場合があります）...")
            coverage_results = original_script.analyze_coverage(
                chunks, qa_pairs, dataset_type,
                custom_threshold=coverage_threshold
            )
            logger.info(f"カバレージ率: {coverage_results.get('coverage_rate', 0):.1%}")

        # 5. 結果保存
        logger.info("\n結果を保存中...")
        saved_files = original_script.save_results(qa_pairs, coverage_results, dataset_type, output_dir)

        return {
            "success": True,
            "saved_files": saved_files,
            "qa_count": len(qa_pairs),
            "coverage_results": coverage_results
        }

    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    finally:
        # ハンドラの削除
        if log_callback:
            logger.removeHandler(handler)
            original_script.logger.removeHandler(handler)

if __name__ == "__main__":
    # Test run
    pass

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""シンプルなCeleryテスト"""

import time
from celery_tasks import generate_qa_for_chunk_async

def test_simple():
    # テスト用のデータ
    chunk_data = {
        "id": "test_chunk_1",
        "text": "これはテスト用のテキストです。Celeryタスクが正しく動作するかを確認しています。",
        "tokens": 50,
        "type": "test",
        "doc_id": "test_doc_1",
        "chunk_idx": 0,
        "dataset_type": "test"
    }

    config = {
        "lang": "ja",
        "qa_per_chunk": 2
    }

    print("タスクを投入中...")
    task = generate_qa_for_chunk_async.apply_async(
        args=[chunk_data, config, "gpt-4o-mini"],
        queue='qa_generation'  # 正しいキューを指定
    )

    print(f"タスクID: {task.id}")
    print("結果を待機中...")

    # 結果を待つ（最大30秒）
    for i in range(30):
        if task.ready():
            result = task.get(timeout=1)
            print("✅ タスク完了!")
            print(f"成功: {result.get('success')}")
            print(f"Q/A数: {len(result.get('qa_pairs', []))}")
            if result.get('qa_pairs'):
                print("生成されたQ/A:")
                for qa in result['qa_pairs']:
                    print(f"  Q: {qa['question']}")
                    print(f"  A: {qa['answer']}")
            return True
        else:
            print(f"  待機中... {i+1}秒")
            time.sleep(1)

    print("❌ タイムアウト")
    return False

if __name__ == "__main__":
    test_simple()
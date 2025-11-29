#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_celery.py - Celeryワーカーの動作確認
"""

import time
import sys
from celery_tasks import app, generate_qa_for_chunk_async
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_celery_connection():
    """Celeryワーカーへの接続テスト"""

    # 1. ワーカーの状態確認
    print("\n=== Celeryワーカー状態確認 ===")
    try:
        # インスペクト
        inspect = app.control.inspect()

        # アクティブなワーカー
        active_workers = inspect.active()
        if active_workers:
            print(f"✅ アクティブワーカー: {len(active_workers)}個")
            for worker, tasks in active_workers.items():
                print(f"  - {worker}: {len(tasks)}個のタスク実行中")
        else:
            print("⚠️  アクティブなワーカーが見つかりません")
            print("   ./start_celery.sh start -w 4 を実行してください")
            return False

        # 登録済みタスク
        registered = inspect.registered()
        if registered:
            print("\n✅ 登録済みタスク:")
            for worker, tasks in registered.items():
                for task in tasks[:3]:  # 最初の3つだけ表示
                    print(f"  - {task}")

        # 統計情報
        stats = inspect.stats()
        if stats:
            print("\n✅ ワーカー統計:")
            for worker, stat in stats.items():
                print(f"  - {worker}:")
                print(f"    プール: {stat.get('pool', {}).get('max-concurrency', 'N/A')}")

    except Exception as e:
        print(f"❌ Celeryワーカーに接続できません: {e}")
        print("   Redisとワーカーが起動しているか確認してください")
        return False

    return True


def test_simple_task():
    """シンプルなテストタスクの実行"""

    print("\n=== テストタスク実行 ===")

    # テスト用のチャンクデータ
    test_chunk = {
        'id': 'test_chunk_001',
        'text': 'これはテスト用のテキストです。Celeryワーカーが正常に動作しているか確認します。',
        'doc_id': 'test_doc',
        'dataset_type': 'test'
    }

    test_config = {
        'lang': 'ja',
        'qa_per_chunk': 2
    }

    try:
        # タスクを投入
        print("テストタスクを投入中...")
        task = generate_qa_for_chunk_async.apply_async(
            args=[test_chunk, test_config, 'gpt-5-mini'],
            queue='qa_generation'
        )

        print(f"タスクID: {task.id}")
        print("結果を待機中...")

        # タイムアウト付きで結果を待つ
        start_time = time.time()
        timeout = 30

        while not task.ready():
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"❌ タイムアウト: {timeout}秒経過")
                return False

            print(f"  状態: {task.state}, 経過: {elapsed:.1f}秒")
            time.sleep(2)

        # 結果を取得
        result = task.get(timeout=1)

        if result.get('success'):
            print("✅ タスク成功!")
            print(f"  生成Q/A数: {len(result.get('qa_pairs', []))}")
            if result.get('qa_pairs'):
                print(f"  サンプル質問: {result['qa_pairs'][0].get('question', '')[:50]}...")
        else:
            print(f"❌ タスク失敗: {result.get('error')}")
            return False

    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

    return True


def main():
    """メイン処理"""

    print("=" * 50)
    print("Celery動作確認ツール")
    print("=" * 50)

    # 接続テスト
    if not test_celery_connection():
        print("\n❌ Celeryワーカーが起動していません")
        print("\n以下の手順で起動してください:")
        print("1. redis-cli ping でRedisを確認")
        print("2. ./start_celery.sh start -w 4 でワーカー起動")
        sys.exit(1)

    # 実際のタスクテスト（オプション）
    print("\n実際のOpenAI APIを使用したテストを実行しますか？")
    print("（APIコストが発生します）")

    response = input("実行する場合は 'y' を入力: ")
    if response.lower() == 'y':
        if test_simple_task():
            print("\n✅ すべてのテストが成功しました!")
        else:
            print("\n❌ テストが失敗しました")
            sys.exit(1)
    else:
        print("\nワーカー接続テストのみ完了しました")

    print("\n✅ Celeryは正常に動作しています")


if __name__ == "__main__":
    main()
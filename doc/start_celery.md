# start_celery.sh 詳細設計書

## 1. 概要
`start_celery.sh` は、RAG Q&A生成システムにおける非同期タスク処理基盤（Celeryワーカー）のライフサイクル管理を行うBashスクリプトです。
RedisをメッセージブローカーとするCeleryワーカープロセス群の「起動」「停止」「再起動」「状態監視」を安全かつ確実に行うための制御ロジックを提供します。

本スクリプトは、`a02_make_qa_para.py` などのPythonスクリプトから投入される `qa_generation` キューのタスクを処理するワーカープロセスを管理します。

## 2. 技術仕様

### 2.1 実行環境
*   **OS**: Linux / macOS (Bashシェル環境)
*   **依存コマンド**:
    *   `celery`: Python Celeryパッケージ
    *   `redis-cli`: Redisコマンドラインツール（サーバー確認・キュー監視用）
    *   `pgrep`, `pkill`, `kill`: プロセス制御
    *   `awk`, `grep`: テキスト処理
*   **ミドルウェア**: Redis Server (デフォルトポート 6379)

### 2.2 デフォルト設定値
スクリプト内で定義されている固定パラメータおよびデフォルト値です。

| パラメータ | 値 | 説明 |
| :--- | :--- | :--- |
| `WORKERS` | 8 | デフォルトの同時実行ワーカープロセス数（Gemini APIレート制限対策） |
| `LOG_LEVEL` | info | Celeryのログレベル (debug, info, warning, error) |
| `QUEUE_NAME` | qa_generation | 監視対象のCeleryキュー名 |
| `PID_FILE` | /tmp/celery_qa.pid | マスタープロセスのPIDファイルパス |
| `LOG_FILE_PATTERN` | logs/celery_qa_%n.log | ログ出力パス（`%n`はノード名に置換） |

## 3. 処理方式詳細

### 3.1 起動処理 (`start`)
Celeryワーカーをバックグラウンドプロセス（デーモン）として起動します。

1.  **Redis接続確認**: `check_redis()` 関数により `redis-cli ping` を実行。応答がない場合は起動を中止。
2.  **多重起動防止**: `pgrep` を使用して既に同名のワーカープロセス（`celery.*worker.*qa_generation`）が存在するか確認。存在する場合はクリーンアップ後に再起動フローへ移行。
3.  **Celery起動コマンド実行**:
    以下のオプションでCeleryを実行します。
    ```bash
    celery -A celery_tasks worker \
        --loglevel=${LOG_LEVEL} \
        --concurrency=${WORKERS} \  # 並列プロセス数
        --pool=prefork \            # 実行プール方式（CPUバウンド処理向け）
        --queues=${QUEUE_NAME} \    # 監視キュー指定
        --hostname=qa_worker@%h \   # ノード名（%hはホスト名）
        --pidfile=/tmp/celery_qa.pid \
        --logfile=logs/celery_qa_%n.log \
        --detach                    # デタッチモード（バックグラウンド化）
    ```
4.  **ログディレクトリ作成**: `logs/` ディレクトリが存在しない場合は `mkdir -p` で作成。

### 3.2 停止処理 (`stop`)
稼働中のワーカープロセスを安全に停止（Graceful Shutdown）します。

1.  **PIDファイルによる停止**: `/tmp/celery_qa.pid` が存在する場合、記載されたPIDに対して `SIGTERM` を送信。
    *   **Wait & Check**: 2秒間待機し、プロセスが終了したか確認。
    *   **強制終了**: 終了しない場合は `SIGKILL` (-9) を送信して強制終了し、PIDファイルを削除。
2.  **プロセス名による停止（フォールバック）**: PIDファイルがない、または機能しない場合、`pkill -f "celery.*worker.*qa_generation"` を実行して関連プロセスを一括停止。

### 3.3 再起動処理 (`restart`)
設定変更の適用やメモリリーク解消のために使用します。

1.  `stop_workers` 関数を実行。
2.  2秒間の待機（ポート解放やプロセス終了待ち）。
3.  `check_redis` によるRedis確認後、`start_workers` を実行。

### 3.4 ステータス確認 (`status`)
システムの健全性と現在の負荷状況を表示します。

1.  **Redis状態**: `redis-cli ping` による接続確認。
2.  **キュー滞留数**: `redis-cli llen celery` コマンドで、処理待ちのタスク数を取得・表示。
3.  **プロセス生存確認**: `pgrep` によるCeleryプロセスの有無。
4.  **内部状態検査**: `celery -A celery_tasks inspect active` および `stats` コマンドを実行し、各ワーカーが現在処理中のタスクやメモリ使用状況などを取得（タイムアウト2秒）。
5.  **最新ログ表示**: `logs/celery_qa_*.log` の末尾5行を表示。

### 3.5 クリーンアップ処理 (`cleanup_workers`)
異常終了などでPIDファイルが残留した場合や、ゾンビプロセスが発生した場合の強制回復処理です。

1.  PIDファイルの削除。
2.  `pkill -9` による全関連プロセスの即時強制終了。

## 4. コマンドライン引数仕様

```bash
./start_celery.sh [COMMAND] [OPTIONS]
```

### コマンド
*   `start`: ワーカーを起動。
*   `stop`: ワーカーを停止。
*   `restart`: ワーカーを再起動。
*   `status`: 現在の状態を表示。

### オプション
*   `-w, --workers NUM`: 並列実行するワーカープロセス数を指定（デフォルト: 8）。
    *   *使用例*: `./start_celery.sh start -w 16`
*   `-l, --loglevel LEVEL`: ログの詳細度を指定（debug, info, warning, error）。
    *   *使用例*: `./start_celery.sh start -l debug`

## 5. 関連ファイル構成

| ファイルパス | 説明 |
| :--- | :--- |
| `celery_tasks.py` | Celeryアプリケーション本体。`generate_qa_unified_async` などのタスク定義が含まれる。 |
| `celery_config.py` | Python側の設定ファイル。Broker URL, レート制限 (`rate_limit`), タイムアウト設定などを定義。 |
| `logs/celery_qa_*.log` | 実行ログ。`%n` 部分がノード名（例: `qa_worker@hostname-1`）に展開され、ワーカーごとのログが保存される。 |
| `/tmp/celery_qa.pid` | 実行中のメインプロセスのPIDを保持する一時ファイル。 |

## 6. エラーハンドリング・トラブルシューティング

*   **Redis未接続**: `start` 実行時にRedisへのPingが通らない場合、エラーメッセージを表示して即時終了（Exit Code 1）。
*   **起動失敗**: `celery ... --detach` コマンドの戻り値が非0の場合、起動失敗とみなしエラーを表示。
*   **停止不全**: `SIGTERM` 送信後もプロセスが残留する場合、`SIGKILL` で強制排除を行うロジックを実装済み。

## 7. 運用上の注意点

1.  **並列数の調整**: Gemini APIのレート制限に抵触する場合、`-w` オプションでワーカー数を減らす（例: `-w 4`）か、`celery_config.py` の `rate_limit` 設定を調整する。
2.  **ログ管理**: ログファイルは自動ローテーションされないため、長時間稼働させる場合はディスク容量に注意するか、外部のログローテーション設定（logrotate等）を追加する。
3.  **Redisフラッシュ**: 開発中にタスク定義を変更した場合や、不要なタスクがキューに溜まった場合は、`redis-cli FLUSHDB` を実行してから `restart` することを推奨。

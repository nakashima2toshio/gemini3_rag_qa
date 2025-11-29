# helper_api.py 仕様書

作成日: 2024-10-29
更新日: 2025-11-27

## 概要

OpenAI API連携を支援するヘルパーモジュール。設定管理、メッセージ処理、トークン管理、レスポンス処理、キャッシュなどOpenAI API利用のための包括的な機能を提供。

## ファイル情報

- **ファイル名**: helper_api.py
- **行数**: 840行
- **主な機能**: OpenAI API統合とキャッシュ
- **主要API**: Responses API、Chat Completions API

## OpenAI API 利用状況一覧

### プロジェクト内で使用中のAPI

| API名 | メソッド | 用途 | 使用箇所 | 説明 |
|------|---------|-----|---------|-----|
| **Responses API** | `client.responses.create()` | テキスト生成（新形式） | helper_api.py:735 | 新しいメッセージ形式（developer role含む）をサポート |
| **Chat Completions API** | `client.chat.completions.create()` | チャット形式の対話生成 | helper_api.py:750 | 従来のチャット形式API、JSON出力対応 |
| **Structured Outputs API** | `client.responses.parse()` | 構造化出力（Pydantic連携） | celery_tasks.py:299-304,532-537 | Pydanticモデルで型安全な出力を保証 |

### プロジェクトで利用可能な追加API

| API名 | メソッド | 用途 | 対応状況 | 説明 |
|------|---------|-----|---------|-----|
| **Embeddings API** | `client.embeddings.create()` | テキストのベクトル化 | 未実装 | RAGシステムでの類似検索用 |
| **Moderation API** | `client.moderations.create()` | コンテンツの安全性チェック | 未実装 | 不適切なコンテンツの検出 |
| **Images API** | `client.images.generate()` | 画像生成 | 未実装 | DALL-E 3による画像生成 |
| **Audio API** | `client.audio.transcriptions.create()` | 音声認識 | 未実装 | Whisperによる文字起こし |
| **Files API** | `client.files.create()` | ファイル管理 | 未実装 | Fine-tuning用データのアップロード |
| **Fine-tuning API** | `client.fine_tuning.jobs.create()` | モデルのファインチューニング | 未実装 | カスタムモデルの作成 |

### API使用時の主要パラメータ

| パラメータ | 対応API | 型 | 説明 | デフォルト値 |
|-----------|--------|---|------|------------|
| `model` | 全API共通 | str | 使用するモデル名 | "gpt-5-mini" |
| `messages`/`input` | Responses, Chat | List | 入力メッセージ | 必須 |
| `temperature` | Chat Completions | float | 生成の多様性（0-2） | 0.7 |
| `max_tokens` | Chat Completions | int | 最大生成トークン数 | モデル依存 |
| `max_completion_tokens` | Responses (新モデル) | int | 最大出力トークン数 | 1000 |
| `max_output_tokens` | Structured Outputs | int | 構造化出力の最大トークン | 1000-2000 |
| `response_format` | Chat/Structured | dict/Model | 出力形式の指定 | なし |
| `text_format` | responses.parse | Pydantic Model | 構造化出力の型定義 | 必須 |

### サポートモデル一覧

| モデルシリーズ | モデル名 | 最大入力トークン | 最大出力トークン | 用途 | 備考 |
|--------------|---------|----------------|----------------|------|------|
| **GPT-4o** | gpt-4o, gpt-4o-mini | 128,000 | 4,096 | 汎用 | 現行主力モデル |
| **GPT-4.1** | gpt-4.1, gpt-4.1-mini | 128,000 | 4,096 | 汎用 | 改良版 |
| **GPT-5** | gpt-5, gpt-5-mini, gpt-5-nano | 未定 | 未定 | 次世代 | 最新モデル |
| **O-Series** | o1, o1-mini | 128,000 | 32,768-65,536 | 推論特化 | temperatureパラメータ非対応 |
| **O-Series (新)** | o3, o3-mini, o4, o4-mini | 200,000-256,000 | 100,000-128,000 | 大規模推論 | 大容量入出力対応 |

---

## アーキテクチャ

### モジュール構造

```
helper_api.py
├── 型定義・インポート (L1-34)
│   └── RoleType = Literal["user", "assistant", "system", "developer"]
│
├── ConfigManager (L39-212)
│   ├── シングルトンパターン
│   ├── YAML設定ファイル読み込み
│   ├── 環境変数オーバーライド
│   └── ロガー設定
│
├── MemoryCache (L222-264)
│   ├── TTL付きキャッシュ
│   ├── サイズ制限管理
│   └── 自動削除機能
│
├── JSON処理関数 (L274-323)
│   ├── safe_json_serializer()
│   └── safe_json_dumps()
│
├── デコレータ (L329-383)
│   ├── error_handler
│   ├── timer
│   └── cache_result
│
├── デフォルトプロンプト (L388-438)
│   ├── developer_text, user_text, assistant_text
│   ├── get_default_messages()
│   ├── append_user_message()
│   ├── append_developer_message()
│   └── append_assistant_message()
│
├── MessageManager (L443-508)
│   ├── メッセージリスト管理
│   ├── ロール検証
│   └── エクスポート/インポート
│
├── TokenManager (L514-598)
│   ├── トークン数計算
│   ├── テキスト切り詰め
│   ├── コスト推定
│   └── モデル制限取得
│
├── ResponseProcessor (L604-687)
│   ├── レスポンス解析
│   ├── テキスト抽出
│   └── ファイル保存
│
├── OpenAIClient (L693-750)
│   ├── create_response() - Responses API
│   └── create_chat_completion() - Chat Completions API
│
└── ユーティリティ関数 (L756-799)
    ├── sanitize_key()
    ├── load_json_file()
    ├── save_json_file()
    ├── format_timestamp()
    └── create_session_id()
```

---

## 主要クラス

### 1. ConfigManager (L39-212)

設定管理のシングルトンクラス。

#### コンストラクタ

```python
def __new__(cls, config_path: str = "config.yml"):
    """シングルトンインスタンスの生成"""
    if cls._instance is None:
        cls._instance = super().__new__(cls)
    return cls._instance

def __init__(self, config_path: str = "config.yml"):
    if hasattr(self, '_initialized'):
        return
    self._initialized = True
    self.config_path = Path(config_path)
    self._config = self._load_config()
    self._cache = {}
    self.logger = self._setup_logger()
```

#### 主要メソッド

| メソッド | 行番号 | 説明 |
|---------|--------|------|
| `_setup_logger()` | L59-92 | ロガー設定（コンソール＋ファイル） |
| `_load_config()` | L95-109 | YAML設定ファイル読み込み |
| `_apply_env_overrides()` | L112-124 | 環境変数オーバーライド |
| `_get_default_config()` | L127-165 | デフォルト設定取得 |
| `get(key, default)` | L167-183 | 設定値取得（ドット区切り対応） |
| `set(key, value)` | L185-194 | 設定値更新 |
| `reload()` | L196-199 | 設定再読み込み |
| `save(filepath)` | L201-211 | 設定保存 |

#### 環境変数オーバーライド

| 環境変数 | 対象設定 |
|---------|---------|
| `OPENAI_API_KEY` | api.openai_api_key |
| `LOG_LEVEL` | logging.level |
| `DEBUG_MODE` | experimental.debug_mode |

#### デフォルト設定

```python
{
    "models": {
        "default": "gpt-5-mini",
        "available": ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"]
    },
    "api": {
        "timeout": 30,
        "max_retries": 3,
        "openai_api_key": None
    },
    "cache": {
        "enabled": True,
        "ttl": 3600,
        "max_size": 100
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": None,
        "max_bytes": 10485760,
        "backup_count": 5
    }
}
```

---

### 2. MemoryCache (L222-264)

TTL付きインメモリキャッシュ。

#### コンストラクタ

```python
def __init__(self):
    self._storage = {}
    self._enabled = config.get("cache.enabled", True)
    self._ttl = config.get("cache.ttl", 3600)
    self._max_size = config.get("cache.max_size", 100)
```

#### 主要メソッド

| メソッド | 行番号 | 説明 |
|---------|--------|------|
| `get(key)` | L231-241 | キャッシュ取得（TTLチェック付き） |
| `set(key, value)` | L243-256 | キャッシュ設定（サイズ制限付き） |
| `clear()` | L258-260 | キャッシュクリア |
| `size()` | L262-264 | キャッシュサイズ取得 |

#### 特徴

- **TTL管理**: 期限切れデータは自動削除
- **サイズ制限**: 最大サイズ超過時は最古のアイテムを削除
- **有効/無効切り替え**: config.cache.enabledで制御

---

### 3. MessageManager (L443-508)

メッセージ履歴の管理クラス。

#### コンストラクタ

```python
def __init__(self, messages: List[EasyInputMessageParam] = None):
    self._messages = messages or self.get_default_messages()
```

#### 主要メソッド

| メソッド | 行番号 | 説明 |
|---------|--------|------|
| `get_default_messages()` | L449-471 | デフォルトメッセージ取得 |
| `add_message(role, content)` | L473-488 | メッセージ追加（ロール検証付き） |
| `get_messages()` | L490-492 | メッセージリスト取得 |
| `clear_messages()` | L494-496 | メッセージクリア |
| `export_messages()` | L498-503 | メッセージエクスポート |
| `import_messages(data)` | L505-508 | メッセージインポート |

#### ロール種別

```python
RoleType = Literal["user", "assistant", "system", "developer"]
```

#### メッセージ数制限

- デフォルト: 50メッセージ
- `developer`メッセージは常に保持
- 古いメッセージから自動削除

---

### 4. TokenManager (L514-598)

トークン数の計算と管理。

#### モデル別エンコーディング (L518-531)

```python
MODEL_ENCODINGS = {
    "gpt-4o": "cl100k_base",
    "gpt-4o-mini": "cl100k_base",
    "gpt-4o-audio-preview": "cl100k_base",
    "gpt-4o-mini-audio-preview": "cl100k_base",
    "gpt-4.1": "cl100k_base",
    "gpt-4.1-mini": "cl100k_base",
    "o1": "cl100k_base",
    "o1-mini": "cl100k_base",
    "o3": "cl100k_base",
    "o3-mini": "cl100k_base",
    "o4": "cl100k_base",
    "o4-mini": "cl100k_base",
}
```

#### 主要メソッド

| メソッド | 行番号 | 説明 |
|---------|--------|------|
| `count_tokens(text, model)` | L533-546 | トークン数計算 |
| `truncate_text(text, max_tokens, model)` | L548-564 | テキスト切り詰め |
| `estimate_cost(input_tokens, output_tokens, model)` | L566-581 | コスト推定 |
| `get_model_limits(model)` | L583-598 | モデル制限取得 |

#### モデル制限一覧 (L586-597)

| モデル | max_tokens | max_output |
|--------|-----------|------------|
| gpt-4o | 128,000 | 4,096 |
| gpt-4o-mini | 128,000 | 4,096 |
| gpt-4.1 | 128,000 | 4,096 |
| gpt-4.1-mini | 128,000 | 4,096 |
| o1 | 128,000 | 32,768 |
| o1-mini | 128,000 | 65,536 |
| o3 | 200,000 | 100,000 |
| o3-mini | 200,000 | 100,000 |
| o4 | 256,000 | 128,000 |
| o4-mini | 256,000 | 128,000 |

---

### 5. ResponseProcessor (L604-687)

APIレスポンスの処理クラス。

#### 主要メソッド

| メソッド | 行番号 | 説明 |
|---------|--------|------|
| `extract_text(response)` | L607-625 | レスポンスからテキスト抽出 |
| `_serialize_usage(usage_obj)` | L627-653 | usageオブジェクトのシリアライズ |
| `format_response(response)` | L655-668 | レスポンスを辞書形式に変換 |
| `save_response(response, filename)` | L670-687 | レスポンスをファイル保存 |

#### extract_text() の処理フロー

1. `response.output` 配列を走査
2. `type == "message"` のアイテムを探索
3. `content` 内の `type == "output_text"` から `text` を抽出
4. フォールバック: `response.output_text` を使用

---

### 6. OpenAIClient (L693-750)

OpenAI API統一クライアント。

#### コンストラクタ

```python
def __init__(self, api_key: str = None):
    if api_key is None:
        api_key = config.get("api.openai_api_key") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("APIキーが設定されていません")

    self.client = OpenAI(api_key=api_key)
```

**APIキー取得優先順位**:
1. コンストラクタ引数
2. config.yml (`api.openai_api_key`)
3. 環境変数 `OPENAI_API_KEY`

#### create_response() (L707-735)

```python
@error_handler
@timer
def create_response(
    self,
    messages: List[EasyInputMessageParam] = None,
    *,
    input: List[EasyInputMessageParam] = None,
    model: str = None,
    **kwargs,
) -> Response:
    """Responses API呼び出し

    `messages` と `input` の両方に対応（互換性維持）
    """
```

**特徴**:
- `messages`（旧仕様）と`input`（新仕様）の両方をサポート
- `@error_handler`: エラーログ記録と例外再送出
- `@timer`: 実行時間計測

#### create_chat_completion() (L737-750)

```python
@error_handler
@timer
def create_chat_completion(
    self,
    messages: List[ChatCompletionMessageParam],
    model: str = None,
    **kwargs
):
    """Chat Completions API呼び出し"""
```

---

## デコレータ

### error_handler (L329-341)

```python
def error_handler(func):
    """エラーハンドリングデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise  # API用なので再送出
    return wrapper
```

### timer (L344-356)

```python
def timer(func):
    """実行時間計測デコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"{func.__name__} took {execution_time:.2f} seconds")
        return result
    return wrapper
```

### cache_result (L359-383)

```python
def cache_result(ttl: int = None):
    """結果キャッシュデコレータ"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not config.get("cache.enabled", True):
                return func(*args, **kwargs)

            cache_key = f"{func.__name__}_{hashlib.md5(...).hexdigest()}"

            cached = cache.get(cache_key)
            if cached is not None:
                return cached

            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        return wrapper
    return decorator
```

---

## ユーティリティ関数

### JSON処理

| 関数 | 行番号 | 説明 |
|-----|--------|------|
| `safe_json_serializer(obj)` | L274-306 | カスタムJSONシリアライザー |
| `safe_json_dumps(data, **kwargs)` | L309-323 | 安全なJSON文字列化 |

#### safe_json_serializer() の処理

1. Pydanticモデル: `model_dump()` を呼び出し
2. dict()メソッド: `dict()` を呼び出し
3. datetime: `isoformat()` で文字列化
4. ResponseUsage: 属性を手動抽出
5. その他: `str()` で文字列化

### ファイル操作

| 関数 | 行番号 | 説明 |
|-----|--------|------|
| `load_json_file(filepath)` | L761-768 | JSONファイル読み込み |
| `save_json_file(data, filepath)` | L771-783 | JSONファイル保存 |

### その他

| 関数 | 行番号 | 説明 |
|-----|--------|------|
| `sanitize_key(name)` | L756-758 | キー名の安全化（英数字とアンダースコアのみ） |
| `format_timestamp(timestamp)` | L786-794 | タイムスタンプフォーマット |
| `create_session_id()` | L797-799 | セッションID生成（8文字のMD5ハッシュ） |

---

## デフォルトプロンプト (L388-438)

### 定数

```python
developer_text = "You are a strong developer..."  # ソフトウェア開発用
user_text = "Organize and identify the problem..."  # 問題解決用
assistant_text = "OpenAIのAPIを使用するには..."  # 日本語回答
```

### ヘルパー関数

| 関数 | 行番号 | 説明 |
|-----|--------|------|
| `get_default_messages()` | L408-414 | デフォルトメッセージリスト取得 |
| `append_user_message(text)` | L416-422 | ユーザーメッセージ追加 |
| `append_developer_message(text)` | L424-430 | 開発者メッセージ追加 |
| `append_assistant_message(text)` | L432-438 | アシスタントメッセージ追加 |

---

## 使用例

### 設定管理

```python
from helper_api import config

# 設定値の取得
default_model = config.get("models.default")  # "gpt-5-mini"
api_timeout = config.get("api.timeout", 30)   # 30

# 設定値の更新
config.set("models.default", "gpt-4o")

# 設定の保存
config.save()
```

### メッセージ管理

```python
from helper_api import MessageManager

msg_manager = MessageManager()

# メッセージの追加
msg_manager.add_message("user", "こんにちは")
msg_manager.add_message("assistant", "こんにちは！何かお手伝いできることはありますか？")

# メッセージリストの取得
messages = msg_manager.get_messages()

# エクスポート
exported = msg_manager.export_messages()
```

### トークン管理

```python
from helper_api import TokenManager

# トークン数計算
text = "これはテストテキストです"
token_count = TokenManager.count_tokens(text, model="gpt-4o-mini")

# テキスト切り詰め
truncated = TokenManager.truncate_text(text, max_tokens=100)

# コスト推定
cost = TokenManager.estimate_cost(1000, 500, model="gpt-4o-mini")
print(f"推定コスト: ${cost:.4f}")

# モデル制限取得
limits = TokenManager.get_model_limits("gpt-4o")
print(f"最大入力: {limits['max_tokens']:,} トークン")
```

### API クライアント

```python
from helper_api import OpenAIClient, MessageManager, ResponseProcessor

# クライアント作成
client = OpenAIClient()

# メッセージ準備
msg_manager = MessageManager()
msg_manager.add_message("user", "OpenAI APIについて教えてください")
messages = msg_manager.get_messages()

# Responses API呼び出し
response = client.create_response(input=messages, model="gpt-4o-mini")

# レスポンス処理
texts = ResponseProcessor.extract_text(response)
print(texts[0])

# レスポンス保存
filepath = ResponseProcessor.save_response(response)
print(f"保存先: {filepath}")
```

### キャッシュ

```python
from helper_api import cache, cache_result

# 直接キャッシュ使用
cache.set("my_key", "my_value")
value = cache.get("my_key")

# デコレーターでキャッシュ
@cache_result(ttl=3600)
def expensive_operation(param):
    # 重い処理
    return result

result = expensive_operation("test")  # 初回実行
result = expensive_operation("test")  # 2回目はキャッシュから取得
```

---

## エクスポート定義

```python
__all__ = [
    # 型定義
    'RoleType',

    # クラス
    'ConfigManager',
    'MessageManager',
    'TokenManager',
    'ResponseProcessor',
    'OpenAIClient',
    'MemoryCache',

    # デコレータ
    'error_handler',
    'timer',
    'cache_result',

    # ユーティリティ
    'sanitize_key',
    'load_json_file',
    'save_json_file',
    'format_timestamp',
    'create_session_id',
    'safe_json_serializer',
    'safe_json_dumps',

    # 定数
    'developer_text',
    'user_text',
    'assistant_text',

    # グローバルインスタンス
    'config',
    'logger',
    'cache',
]
```

---

## 設定ファイル (config.yml)

```yaml
models:
  default: gpt-5-mini
  available:
    - gpt-4o-mini
    - gpt-4o
    - gpt-4.1
    - gpt-4.1-mini

api:
  timeout: 30
  max_retries: 3
  message_limit: 50
  openai_api_key: null  # 環境変数から取得

cache:
  enabled: true
  ttl: 3600
  max_size: 100

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/app.log
  max_bytes: 10485760
  backup_count: 5

experimental:
  debug_mode: false
  performance_monitoring: true
```

---

## 制限事項

1. **APIキー**: 環境変数またはconfig.ymlで設定必須
2. **シングルトン**: ConfigManagerは1インスタンスのみ
3. **キャッシュTTL**: デフォルト1時間で期限切れ
4. **ログファイル**: ローテーション設定（10MB、5ファイル）
5. **メッセージ制限**: デフォルト50メッセージ

---

## トラブルシューティング

### 問題1: APIキーエラー

**症状**: "APIキーが設定されていません"

**解決策**:
```bash
export OPENAI_API_KEY='your-api-key'
# または config.yml に設定
```

### 問題2: 設定ファイルが見つからない

**症状**: "設定ファイルが見つかりません"

**解決策**: config.yml作成またはデフォルト設定使用

### 問題3: キャッシュが効かない

**症状**: キャッシュから値が取得できない

**解決策**: config.ymlでcache.enabledを確認

---

## 注意事項（CLAUDE.mdより）

1. **モデル名**: config.pyで定義されたモデル名をそのまま使用すること。マッピングを作成しないこと。

2. **API選択**:
   - 構造化出力が必要: `client.responses.parse()`
   - 通常のテキスト生成: `client.responses.create()`

3. **temperatureパラメータ**: GPT-5シリーズ、O-Seriesはtemperature=1のみサポート。

---

作成日: 2024-10-29
更新日: 2025-11-27
作成者: OpenAI RAG Q/A JP Development Team
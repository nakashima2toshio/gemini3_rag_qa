# Q/Aペア生成処理ドキュメント

本ドキュメントでは、RAG Q/A生成システムにおけるQ/Aペア生成の実行・処理フローについて解説する。
Geminiモデルと `UnifiedLLMClient` への移行を反映した最新版である。

## 目次

- [1. 概要](#1-概要)
  - [1.1 本ドキュメントの位置づけ](#11-本ドキュメントの位置づけ)
  - [1.2 関連ファイル一覧](#12-関連ファイル一覧)
  - [1.3 Q/A生成処理の全体フロー図](#13-qa生成処理の全体フロー図)
- [2. a02_make_qa_para.py の実行フロー](#2-a02_make_qa_parapy-の実行フロー)
  - [2.1 コマンドライン引数一覧](#21-コマンドライン引数一覧)
  - [2.2 処理ステップ詳細](#22-処理ステップ詳細)
  - [2.3 同期処理 vs 非同期処理の選択](#23-同期処理-vs-非同期処理の選択)
- [3. 非同期並列処理の必要性（Gemini時代）](#3-非同期並列処理の必要性gemini時代)
  - [3.1 API特性の変化（OpenAI vs Gemini）](#31-api特性の変化openai-vs-gemini)
  - [3.2 並列化によるスループット向上](#32-並列化によるスループット向上)
- [4. Celery並列処理アーキテクチャ](#4-celery並列処理アーキテクチャ)
  - [4.1 システム構成図](#41-システム構成図)
  - [4.2 タスク実装詳細（UnifiedLLMClient利用）](#42-タスク実装詳細unifiedllmclient利用)
  - [4.3 ワーカー設定と推奨値](#43-ワーカー設定と推奨値)
  - [4.4 結果収集メカニズム](#44-結果収集メカニズム)
- [5. 出力とファイル保存](#5-出力とファイル保存)
  - [5.1 出力形式](#51-出力形式)
  - [5.2 保存ディレクトリ構造](#52-保存ディレクトリ構造)
- [6. カバレージ分析](#6-カバレージ分析)
- [7. UIからの実行（Streamlit）](#7-uiからの実行streamlit)
- [8. 実行例とトラブルシューティング](#8-実行例とトラブルシューティング)

---

## 1. 概要

### 1.1 本ドキュメントの位置づけ

本ドキュメントは「Q/Aペア生成の実行・処理フロー」に焦点を当てる。

| ドキュメント | 焦点 | 内容 |
|-------------|------|------|
| `doc/01_chunk.md` | チャンク分割技術 | SemanticCoverage、文分割 |
| `doc/04_prompt.md` | プロンプト設計 | Gemini向けプロンプト、UnifiedLLMClient |
| `doc/05_qa_pair.md`（本書） | 実行・処理フロー | 並列処理、Celery、出力、カバレージ |

### 1.2 関連ファイル一覧

| ファイル | 役割 |
|---------|------|
| `a02_make_qa_para.py` | Q/A生成のメインスクリプト |
| `helper_llm.py` | LLM抽象化レイヤー（Gemini/OpenAI共通） |
| `celery_tasks.py` | Celery非同期タスク定義 |
| `config.py` | 設定管理 |
| `models.py` | Pydanticモデル |

### 1.3 Q/A生成処理の全体フロー図

```
[入力データ]
    │
    ▼
[1. データ読み込み]
    │
    ▼
[2. チャンク作成] (create_document_chunks)
    │
    ▼
[3. チャンク前処理] (merge_small_chunks)
    │
    ▼
[4. Q/A生成]
    │
    ├── 【同期処理】 (UnifiedLLMClient直接呼び出し)
    │       └── generate_qa_pairs_for_batch()
    │
    └── 【非同期処理】 (Celery経由)
            ├── generate_qa_for_batch_async()
            │       └── UnifiedLLMClient.generate_structured()
            └── collect_results() (Redis)
    │
    ▼
[5. カバレージ分析]
    │
    ▼
[6. 結果保存]
```

---

## 2. a02_make_qa_para.py の実行フロー

### 2.1 コマンドライン引数一覧

Geminiモデルへの移行に伴い、デフォルトモデルが変更されています。

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--dataset` | なし | データセット名 |
| `--model` | **gemini-2.0-flash** | 使用するモデル |
| `--batch-chunks` | 3 | 1回のAPIで処理するチャンク数 |
| `--use-celery` | 無効 | Celery並列処理を使用 |
| `--celery-workers` | 8 | Celeryワーカー数（Geminiのレート制限考慮） |

### 2.2 処理ステップ詳細

基本的なフローは変わりませんが、内部で `UnifiedLLMClient` が使用される点が異なります。

1.  **データ読み込み**: `preprocessed_{dataset}.csv` またはローカルファイルを読み込み。
2.  **チャンク化**: セマンティック分割を実行。
3.  **Q/A生成**: 指定されたモード（同期/非同期）で生成を実行。
4.  **保存**: JSON/CSV形式で保存。

---

## 3. 非同期並列処理の必要性（Gemini時代）

### 3.1 API特性の変化（OpenAI vs Gemini）

Gemini 2.0 Flash は OpenAI GPT-4o 等と比較して**圧倒的に高速かつ安価**です。
しかし、大量のデータを処理する場合や、レート制限（RPM: Requests Per Minute）を効率的に使い切るためには、依然として並列処理が有効です。

*   **Gemini 2.0 Flash**: 高速。バッチサイズを大きく（3〜5）してもレスポンスが速い。
*   **OpenAI GPT-4o**: 比較的高価。バッチ処理によるコスト削減効果が高い。

### 3.2 並列化によるスループット向上

Gemini API のレート制限内で最大限のスループットを出すために、Celeryによる並列制御が役立ちます。

| ワーカー数 | 処理速度目安 | 備考 |
|-----------|-------------|------|
| 1 (同期) | 基準 | |
| 8 | ~7倍 | 推奨設定（安定） |
| 24 | ~20倍 | 大量データ向け（レート制限注意） |

---

## 4. Celery並列処理アーキテクチャ

### 4.1 システム構成図

構成自体はOpenAI時代と同様ですが、ワーカー内部で実行されるコードが `UnifiedLLMClient` に置き換わっています。

```
[Main Process] -> [Redis Queue] -> [Celery Workers] -> [Redis Result] -> [Main Process]
```

### 4.2 タスク実装詳細（UnifiedLLMClient利用）

`celery_tasks.py` 内のタスクは `helper_llm.py` を利用して抽象化されています。

```python
@app.task(bind=True, max_retries=3)
def generate_qa_for_batch_async(self, chunks: List[Dict], config: Dict, model: str) -> Dict:
    """
    複数チャンクからQ/Aペアを非同期生成
    """
    try:
        # UnifiedLLMClient (Providerは自動または指定)
        client = create_llm_client(provider="gemini")
        
        # プロンプト構築...
        
        # 構造化出力生成
        response = client.generate_structured(
            prompt=combined_input,
            response_schema=QAPairsResponse,
            model=model
        )
        
        return {"success": True, "qa_pairs": [qa.dict() for qa in response.qa_pairs]}
        
    except Exception as e:
        raise self.retry(exc=e)
```

### 4.3 ワーカー設定と推奨値

Gemini API は高速なため、少なめのワーカー数でも十分なスループットが出ます。

*   **推奨ワーカー数**: 8 - 16
*   **起動コマンド**: `./start_celery.sh start -w 8`

### 4.4 結果収集メカニズム

Redisへの直接アクセスを行い、タスクの完了状態を監視します。これにより、Celeryの `AsyncResult.get()` で発生しがちなタイムアウトや状態不整合を防ぎます。

---

## 5. 出力とファイル保存

### 5.1 出力形式

`models.py` の `QAPair` 定義に基づき、以下のメタデータを含むJSON/CSVが出力されます。

*   `question`
*   `answer`
*   `question_type` (fact, reason, comparison, application)
*   `source_chunk_id`
*   `doc_id`

### 5.2 保存ディレクトリ構造

```
qa_output/
├── a02/
│   ├── qa_pairs_{dataset}_{timestamp}.json
│   ├── coverage_{dataset}_{timestamp}.json
│   └── summary_{dataset}_{timestamp}.json
```

---

## 6. カバレージ分析

生成されたQ/Aペアが元のテキストをどの程度カバーしているかを、埋め込みベクトル（`text-embedding-3-small` 推奨）のコサイン類似度で判定します。
Geminiへの移行後も、埋め込みモデルにはOpenAI等の専用モデルを使用するのが一般的です（`SemanticCoverage` クラスが担当）。

---

## 7. UIからの実行（Streamlit）

`ui/pages/qa_generation_page.py` から以下の操作が可能です。

1.  入力ソース選択（データセット/ローカルファイル）
2.  Celeryワーカー数の設定
3.  モデル選択（`gemini-2.0-flash` 等）
4.  進捗のリアルタイム表示

---

## 8. 実行例とトラブルシューティング

### 典型的な実行コマンド

```bash
# Gemini 2.0 Flash を使用した並列生成
python a02_make_qa_para.py \
  --dataset cc_news \
  --use-celery \
  --celery-workers 8 \
  --batch-chunks 3 \
  --model gemini-2.0-flash \
  --analyze-coverage
```

### トラブルシューティング

*   **レート制限エラー**: ワーカー数を減らすか、`batch-chunks` を増やしてAPI呼び出し頻度を下げる。
*   **JSONパースエラー**: `UnifiedLLMClient` 内部で吸収されるが、頻発する場合はプロンプトの調整が必要。
*   **Redis接続エラー**: `start_celery.sh` でRedisが起動しているか確認。

```
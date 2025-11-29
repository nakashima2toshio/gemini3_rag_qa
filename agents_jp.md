# リポジトリガイドライン

本リポジトリは、OpenAI 埋め込みと Qdrant を用いた日本語向け RAG（Retrieval‑Augmented Generation）Q&A システムです。変更は最小・明確・再現可能に保ってください。

## プロジェクト構成・モジュール配置
- コア: `rag_qa.py`（SemanticCoverage）、`helper_api.py`、`helper_rag.py`、`helper_st.py`
- スクリプト: `make_qa.py`（分析/デモ）、`server.py`（起動/検証）
- ドキュメント/データ: `doc/`（設計/説明）、`datasets/`（入力）、生成物は `OUTPUT/`
- インフラ: `docker-compose/`（Qdrant）、`config.yml`（設定）、`.env`（秘匿設定）
- レガシー: `old_code/`（`aXX_*.py` ユーティリティ、旧 Streamlit UI）

## ビルド・テスト・開発コマンド
- 依存関係: `pip install -r requirements.txt` または `python setup.py --quick`
- Qdrant 起動: `docker-compose -f docker-compose/docker-compose.yml up -d`
- 実行/検証: `python server.py`
- 旧 UI（任意）: `streamlit run old_code/a50_rag_search_local_qdrant.py`
- 静的解析/整形: `ruff check .`、`ruff format`

例:
```bash
pip install -r requirements.txt
docker-compose -f docker-compose/docker-compose.yml up -d
python server.py
ruff check . && ruff format
```

## コーディングスタイル・命名規則
- Python 3.8+、PEP 8、インデント 4 スペース、型ヒント推奨
- 命名: 関数/変数 `snake_case`、クラス `CamelCase`、定数 `UPPER_SNAKE_CASE`、ファイル `snake_case.py`
- 単一責務・短関数を心掛け、要点の docstring（三重引用符）を付与
- 秘密情報/パスの直書き禁止。`.env` と `config.yml` から読み込み

## テスト方針
- フレームワーク: `pytest`（必要に応じて導入）。配置は `tests/`、命名は `test_*.py`
- 導入: `pip install -U pytest pytest-cov`（または `uv add -d pytest pytest-cov`）
- 実行: `pytest -q`
- 重要ロジック（例: `rag_qa.SemanticCoverage`、`helper_*`）を優先してユニットテスト
- ネットワーク依存はモック化/スキップ。小さなローカル fixture を使用
- カバレッジ: `pytest -q --cov=rag_qa,helper_api,helper_rag --cov-report=term-missing --cov-fail-under=80`（目標: 80%以上）

## コミット・PR ガイドライン
- Conventional Commits: `feat:` `fix:` `docs:` `refactor:` `test:` `chore:`
- PR には説明、関連 Issue、検証手順（実行コマンド）、UI 変更はスクリーンショット/ログ
- `ruff check .` 合格必須。挙動/コマンド変更時は `README.md` や `doc/` を更新

## セキュリティ・設定
- `.env` に `OPENAI_API_KEY`、`QDRANT_URL`（コミット禁止）
- サンプルデータは匿名化・最小限。設定はパラメータ化を優先

## CI 推奨
- GitHub Actions で PR/Push 時に `ruff check .` と `pytest` を実行
- 依存導入: `pip install -r requirements.txt && pip install -U pytest pytest-cov ruff`
- カバレッジ準拠: `pytest --cov=rag_qa,helper_api,helper_rag --cov-report=term-missing --cov-fail-under=80`
- pip キャッシュを有効化し、Python 3.10/3.11/3.12 でのマトリクステストを推奨
- 生成された `coverage.xml` をアーティファクトとしてアップロード
- 統合テストがある場合は Qdrant サービスコンテナを起動し、`/readyz` で待機

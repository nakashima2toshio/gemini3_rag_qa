# a31_make_gemini_corpus.py - Gemini Semantic Retriever (Corpus) 管理・検索ツール

作成日: 2025-11-28 (Gemini移行対応)

## 目次

1. [概要](#1-概要)
2. [アーキテクチャ](#2-アーキテクチャ)
3. [主な機能](#3-主な機能)
4. [UI操作ガイド](#4-ui操作ガイド)
   - [4.1 作成・管理モード](#41-作成管理モード)
   - [4.2 検索・Q&Aモード](#42-検索qaモード)
5. [設定と準備](#5-設定と準備)
6. [実行方法](#6-実行方法)
7. [トラブルシューティング](#7-トラブルシューティング)

---

## 1. 概要

`a31_make_gemini_corpus.py` は、Google Gemini API の **Semantic Retriever** 機能（Corpus）を利用して、ドキュメントの管理と検索を行うための統合 Streamlit アプリケーションです。
CSVファイルからテキストデータを読み込み、Gemini Corpus として登録することで、**AQA (Attributed Question Answering)** モデルを使用した「根拠付き回答生成」が可能になります。

以前の OpenAI Vector Store 管理ツール (`a31_make_cloud_vector_store_vsid.py`) の機能を継承しつつ、Gemini エコシステムに完全移行しました。

## 2. アーキテクチャ

本ツールは `google-generativeai` Python SDK を使用して、以下のコンポーネントを操作します。

```mermaid
flowchart TB
    subgraph Streamlit_App["Streamlit App (a31)"]
        UI["UI (作成/検索モード)"]
        Manager["GeminiCorpusManager"]
        Processor["CorpusProcessor"]
    end

    subgraph Gemini_API["Google Gemini API"]
        Corpus["Corpus (Semantic Retriever)"]
        Doc["Document"]
        AQA["AQA Model (models/aqa)"]
    end

    subgraph Local_Data["ローカルデータ"]
        CSV["qa_output/*.csv"]
    end

    UI --> Manager
    Manager --> Processor
    Processor --> CSV : 読み込み & 変換
    Manager --> Corpus : 作成 / 削除 / 一覧取得
    Manager --> Doc : アップロード (create_document)
    Manager --> AQA : 検索クエリ (generate_answer)
    AQA --> Corpus : 参照
    AQA -- 回答 + 根拠 --> UI
```

## 3. 主な機能

1.  **Corpus 作成**:
    *   ローカルのCSVファイル（`qa_output/` 内）を読み込み、Gemini Corpus の `Document` として登録します。
    *   単一データセットの処理および複数データセットの統合（Unified Corpus）に対応しています。
2.  **Corpus 管理**:
    *   作成済みの Corpus 一覧を表示します。
    *   不要になった Corpus を削除できます。
3.  **Semantic Search & QA (AQA)**:
    *   作成した Corpus を選択して自然言語で質問できます。
    *   Gemini の AQA モデル (`models/aqa`) が、Corpus 内の情報を根拠として回答を生成します。
    *   回答の根拠となるドキュメント（Passage）を提示します。

## 4. UI操作ガイド

サイドバーの「機能選択」でモードを切り替えます。

### 4.1 作成・管理モード

**タブ1: 🔗 Corpus作成**
1.  「データセット選択」で登録したいデータセット（例: `CC News Q&A`）にチェックを入れます。
2.  「🚀 Corpus作成開始」ボタンをクリックします。
3.  処理が完了すると、作成された Corpus ID が表示されます。

**タブ2: 📚 既存Corpus管理**
*   現在のアカウントに関連付けられている Corpus の一覧が表示されます。
*   各行の「🗑️ 削除」ボタンで Corpus を削除できます。

### 4.2 検索・Q&Aモード

**🔎 Semantic Search & QA (AQA)**
1.  **検索対象Corpus**: プルダウンから検索したい Corpus を選択します。
2.  **質問**: テキストエリアに質問を入力します（例: "AI技術の最新動向は？"）。
3.  **「🔍 検索・回答生成」**: ボタンをクリックすると AQA モデルが実行されます。
4.  **結果表示**:
    *   **🤖 AI回答**: 質問に対する回答。
    *   **📚 参照ドキュメント (根拠)**: 回答の根拠となったテキスト部分と Source ID。

## 5. 設定と準備

**必須環境変数 (.env):**

```env
GOOGLE_API_KEY=your_gemini_api_key
```

**依存ライブラリ:**

```bash
pip install google-generativeai streamlit pandas python-dotenv
```

**データセット設定:**
`CorpusConfig` クラスで定義されています。新しいデータセットを追加する場合は、コード内の `get_all_configs` メソッドを更新してください。

## 6. 実行方法

```bash
# 仮想環境を有効化 (推奨)
source .venv/bin/activate

# アプリケーション起動
streamlit run a31_make_gemini_corpus.py --server.port=8502
```

ブラウザで `http://localhost:8502` にアクセスしてください。

## 7. トラブルシューティング

| エラー | 原因と対策 |
|---|---|
| `Google Generative AI SDK が見つかりません` | `pip install google-generativeai` を実行してください。 |
| `GOOGLE_API_KEYが設定されていません` | `.env` ファイルに正しい API キーが設定されているか確認してください。 |
| `404 Not Found` (Corpus作成時) | プロジェクト設定やAPIの権限を確認してください。Semantic Retriever は一部のリージョンやプランでのみ利用可能な場合があります。 |
| `AQA実行エラー` | 選択した Corpus に関連する情報が含まれていないか、質問が具体的でない可能性があります。 |

---
**注意**: Gemini Semantic Retriever (Corpus) は、Google AI Studio / Vertex AI の機能であり、利用には適切な権限と課金設定が必要です。

# a31_make_gemini_file_manager.py - Gemini File API Manager (ドキュメント管理・検索ツール)

作成日: 2025-11-28 (Gemini File API移行対応)

## 目次

1. [概要](#1-概要)
2. [アーキテクチャ](#2-アーキテクチャ)
3. [主な機能](#3-主な機能)
4. [UI操作ガイド](#4-ui操作ガイド)
   - [4.1 ファイル管理モード](#41-ファイル管理モード)
   - [4.2 検索・Q&Aモード](#42-検索qaモード)
5. [設定と準備](#5-設定と準備)
6. [実行方法](#6-実行方法)
7. [トラブルシューティング](#7-トラブルシューティング)

---

## 1. 概要

`a31_make_gemini_file_manager.py` は、Google Gemini API の **File API** を利用して、ドキュメントのアップロード、管理、そして検索・Q&Aを行うための統合 Streamlit アプリケーションです。
CSVファイルからテキストデータを読み込み、Gemini File としてアップロードすることで、`gemini-2.0-flash` モデルがそのファイルをコンテキストとして参照し、質問に対する回答を生成できます。

以前の OpenAI Vector Store や Gemini Semantic Retriever (Corpus) 管理ツールから、より直接的で柔軟な Gemini File API ベースの機能に再構築されました。

## 2. アーキテクチャ

本ツールは `google-generativeai` Python SDK を使用して、以下のコンポーネントを操作します。

```mermaid
flowchart TB
    subgraph Streamlit_App["Streamlit App (a31)"]
        UI["UI (ファイル管理/検索モード)"]
        Manager["GeminiFileManager"]
        Processor["FileProcessor"]
    end

    subgraph Gemini_API["Google Gemini API"]
        FileAPI["File API (genai.upload_file/list_files)"]
        GeminiModel["GenerativeModel (gemini-2.0-flash)"]
    end

    subgraph Local_Data["ローカルデータ"]
        CSV["qa_output/*.csv"]
    end

    UI --> Manager
    Manager --> Processor
    Processor --> CSV : 読み込み & 変換
    Manager --> FileAPI : アップロード / 削除 / 一覧取得
    Manager --> GeminiModel : 検索クエリ + Fileコンテキスト
    GeminiModel -- 回答 --> UI
```

## 3. 主な機能

1.  **ファイルアップロード**:
    *   ローカルのCSVファイル（`qa_output/` 内）を読み込み、テキスト形式に変換後、Gemini File としてアップロードします。
    *   Q&A形式のCSVを読みやすいMarkdown風テキストに変換してアップロードします。
2.  **ファイル管理**:
    *   アップロード済みの Gemini File の一覧を表示します。
    *   不要になったファイルを削除できます。
3.  **Semantic Search & QA**:
    *   アップロード済みのファイルを選択して自然言語で質問できます。
    *   `gemini-2.0-flash` モデルが、選択したファイルをコンテキストとして参照し、質問に対する回答を生成します。

## 4. UI操作ガイド

サイドバーの「機能選択」でモードを切り替えます。

### 4.1 ファイル管理モード

**タブ1: 🔗 アップロード**
1.  「データセット選択 (アップロード)」でアップロードしたいデータセット（例: `Wikipedia JA Q&A (a02)`）にチェックを入れます。
2.  「🚀 アップロード開始」ボタンをクリックします。
3.  処理が完了すると、アップロードされたファイル情報が表示されます。

**タブ2: 📚 ファイル一覧**
*   現在のアカウントに関連付けられている Gemini File の一覧が表示されます。
*   各行の「🗑️ 削除」ボタンでファイルを削除できます。
*   「🔄 一覧更新」ボタンで最新の状態に更新できます。

### 4.2 検索・Q&Aモード

**🔎 Semantic Search & QA**
1.  **検索対象ファイル**: プルダウンから検索したいアップロード済みファイルを選択します。
2.  **質問**: テキストエリアに質問を入力します（例: "日本の人口は？"）。
3.  **「🔍 検索・回答生成」**: ボタンをクリックすると `gemini-2.0-flash` モデルが実行され、回答が生成されます。
4.  **結果表示**:
    *   **🤖 AI回答**: 質問に対する回答。

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
`FileConfig` クラスで定義されています。`qa_output/` ディレクトリ内のCSVファイルとマッピングされます。

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
| `Google Generative AI SDK が見つかりません` | `pip install google-generativeai` を実行してください。Pythonの仮想環境が正しく有効化されているか確認してください。 |
| `GOOGLE_API_KEYが設定されていません` | `.env` ファイルに正しい API キーが設定されているか確認してください。 |
| `ファイルが見つかりません` | `qa_output/` ディレクトリに、選択したデータセットに対応するCSVファイルが存在するか確認してください。ファイル名のパターンマッチングも考慮されます。 |
| `404 models/gemini-2.0-flash is not found` | `gemini-2.0-flash` モデルが利用できないリージョンであるか、API バージョンでサポートされていない可能性があります。`genai.list_models()` を実行して利用可能なモデルを確認してください。また、`helper_llm.py` で定義されているモデルリストも参照してください。 |
| `Q&A実行エラー` | 質問内容がドキュメントから回答できない場合や、API呼び出しに問題が発生した場合に起こります。ドキュメントの内容と質問の関連性を確認し、より具体的な質問を試してください。 |

---
**注意**: Gemini File API およびモデルの利用には、適切な権限と課金設定が必要です。

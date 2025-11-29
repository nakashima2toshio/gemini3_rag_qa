# a34_rag_search_cloud_vs.py - 技術仕様書

## 目次

1. [概要](#1-概要)
2. [アーキテクチャ](#2-アーキテクチャ)
3. [VectorStoreManager](#3-vectorstoremanager)
4. [ModernRAGManager](#4-modernragmanager)
5. [日本語回答生成](#5-日本語回答生成)
6. [UI構成](#6-ui構成)
7. [テスト質問](#7-テスト質問)
8. [使用方法](#8-使用方法)
9. [設定ファイル](#9-設定ファイル)
10. [トラブルシューティング](#10-トラブルシューティング)

---

## 1. 概要

### 1.1 目的

`a34_rag_search_cloud_vs.py`は、OpenAI Vector StoreとResponses APIを使用した最新RAG検索Streamlitアプリケーションです。動的なVector Store管理と重複ID問題の解決機能を実装しています。

### 1.2 起動コマンド

```bash
streamlit run a34_rag_search_cloud_vs.py --server.port=8503
```

### 1.3 主要機能

- **OpenAI Responses API + file_searchツール**: 最新APIでVector Store検索
- **動的Vector Store ID管理**: `vector_stores.json`による設定管理
- **重複Vector Store対応**: 同名Store時は最新作成日時を優先
- **ファイル引用表示**: 検索結果の出典を表示
- **日本語回答生成**: 英語検索結果から自然な日本語回答を生成
- **型安全実装**: 型エラー完全修正
- **環境変数APIキー管理**: セキュアな設定方法
- **Agent SDK連携**: セッション管理（オプション）

### 1.4 対応Vector Store

| カテゴリ | データセット名 | 説明 |
|---------|--------------|------|
| CC News | CC News Q&A (a02_llm) | LLM生成方式 |
| CC News | CC News Q&A (a03_rule) | ルールベース生成方式 |
| CC News | CC News Q&A (a10_hybrid) | ハイブリッド生成方式 |
| Livedoor | Livedoor Q&A (a02_llm) | LLM生成方式 |
| Livedoor | Livedoor Q&A (a03_rule) | ルールベース生成方式 |
| Livedoor | Livedoor Q&A (a10_hybrid) | ハイブリッド生成方式 |
| その他 | Customer Support FAQ | カスタマーサポートFAQ |
| その他 | Science & Technology Q&A | サイエンス・テクノロジーQ&A |
| その他 | Medical Q&A | 医療Q&A |
| その他 | Legal Q&A | 法務Q&A |
| その他 | Trivia Q&A | トリビアQ&A |
| その他 | Unified Knowledge Base | 統合ナレッジベース |

---

## 2. アーキテクチャ

### 2.1 システム構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                    a34_rag_search_cloud_vs.py                   │
├─────────────────────────────────────────────────────────────────┤
│  VectorStoreManager (74-368)                                    │
│  ├── DEFAULT_VECTOR_STORES - デフォルト設定（12種類）             │
│  ├── STORE_NAME_MAPPING - Store名マッピング                      │
│  ├── DISPLAY_NAME_MAPPING - 表示名マッピング                     │
│  ├── load_vector_stores() - 設定ファイル読み込み                  │
│  ├── save_vector_stores() - 設定ファイル保存                     │
│  ├── fetch_latest_vector_stores() - OpenAI API取得（重複解決）   │
│  ├── get_vector_stores() - キャッシュ付き取得                    │
│  ├── refresh_and_save() - 強制更新・保存                        │
│  └── debug_vector_stores() - デバッグ情報取得                    │
├─────────────────────────────────────────────────────────────────┤
│  ModernRAGManager (466-691)                                     │
│  ├── search_with_responses_api() - Responses API検索             │
│  ├── search_with_agent_sdk() - Agent SDK検索（オプション）        │
│  ├── search() - 統合検索メソッド                                 │
│  ├── _extract_response_text() - レスポンステキスト抽出            │
│  ├── _extract_citations() - ファイル引用抽出                     │
│  └── _extract_tool_calls() - ツール呼び出し情報抽出               │
├─────────────────────────────────────────────────────────────────┤
│  UI関数群 (700-1106)                                            │
│  ├── initialize_session_state() - セッション状態初期化            │
│  ├── display_search_history() - 検索履歴表示                     │
│  ├── display_test_questions() - テスト質問表示                   │
│  ├── display_vector_store_management() - Vector Store管理UI      │
│  ├── display_search_options() - 検索オプション設定               │
│  ├── generate_enhanced_response() - 日本語回答生成               │
│  └── display_search_results() - 検索結果表示                     │
├─────────────────────────────────────────────────────────────────┤
│  main() (1108-1475)                                             │
│  ├── ページ設定・セッション初期化                                 │
│  ├── サイドバー（Vector Store選択、モデル選択、検索オプション）     │
│  ├── 質問入力フォーム                                            │
│  ├── 検索実行・結果表示                                          │
│  └── 検索履歴表示                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 依存モジュール

```python
from openai import OpenAI
from helper_rag import AppConfig, select_model, show_model_info
import streamlit as st
import logging
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

# オプション
from agents import Agent, Runner, SQLiteSession  # Agent SDK
```

### 2.3 データフロー（検索）

```
ユーザー質問入力
       │
       ▼
Vector Store選択
       │
       ▼
ModernRAGManager.search()
       │
       ├──→ search_with_responses_api()
       │         │
       │         ▼
       │    file_searchツール設定
       │         │
       │         ▼
       │    Responses API呼び出し
       │         │
       │         ▼
       │    レスポンステキスト抽出
       │         │
       │         ▼
       │    ファイル引用抽出
       │
       └──→ search_with_agent_sdk()（オプション）
               │
               ▼
          Agent SDKセッション実行
               │
               ▼
          フォールバック: Responses API
       │
       ▼
メタデータ構築
       │
       ▼
日本語回答生成（generate_enhanced_response）
       │
       ▼
検索結果表示
       │
       ▼
検索履歴追加（最新50件）
```

---

## 3. VectorStoreManager

### 3.1 概要

動的Vector Store管理と重複ID問題の解決を担当するクラス。

### 3.2 DEFAULT_VECTOR_STORES (81-97)

```python
DEFAULT_VECTOR_STORES = {
    # CC News データセット（3種類の生成方式）
    "CC News Q&A (a02_llm)": "vs_cc_news_a02_llm_placeholder",
    "CC News Q&A (a03_rule)": "vs_cc_news_a03_rule_placeholder",
    "CC News Q&A (a10_hybrid)": "vs_cc_news_a10_hybrid_placeholder",
    # Livedoor データセット（3種類の生成方式）
    "Livedoor Q&A (a02_llm)": "vs_livedoor_a02_llm_placeholder",
    "Livedoor Q&A (a03_rule)": "vs_livedoor_a03_rule_placeholder",
    "Livedoor Q&A (a10_hybrid)": "vs_livedoor_a10_hybrid_placeholder",
    # その他のデータセット
    "Customer Support FAQ": "vs_68c94da49c80819189dd42d6e941c4b5",
    "Science & Technology Q&A": "vs_68c94db932fc8191b6e17f86e6601bc1",
    "Medical Q&A": "vs_68c94daffc708191b3c561f4dd6b2af8",
    "Legal Q&A": "vs_68c94dc1cc008191a197bdbc3947a67b",
    "Trivia Q&A": "vs_68c94dc9e6b08191946d7cafcd9880a3",
    "Unified Knowledge Base": "vs_unified_placeholder",
}
```

### 3.3 主要メソッド

#### load_vector_stores (128-150)

```python
def load_vector_stores(self) -> Dict[str, str]:
    """Vector Store設定を読み込み

    - vector_stores.jsonから読み込み
    - ファイルがない場合はDEFAULT_VECTOR_STORESを使用
    """
```

#### fetch_latest_vector_stores (173-278) - 重複問題解決版

```python
def fetch_latest_vector_stores(self) -> Dict[str, str]:
    """OpenAI APIから最新のVector Store一覧を取得（重複解決）

    処理フロー:
    1. OpenAI APIからVector Store一覧を取得
    2. 作成日時（created_at）で降順ソート
    3. 同名Storeの場合は最新を優先
    4. DEFAULT_VECTOR_STORESの順序を維持
    """
```

**重複解決ロジック**:
```python
# 作成日時でソート（新しい順）
sorted_stores = sorted(
    stores_response.data,
    key=lambda x: x.created_at if hasattr(x, 'created_at') else 0,
    reverse=True
)

# 同名Storeの重複チェック
if matched_display_name not in store_candidates:
    # 新規候補として登録
    store_candidates[matched_display_name] = {...}
else:
    # 既存候補と比較
    if created_at > existing['created_at']:
        # 新しい方を優先
        store_candidates[matched_display_name] = {...}
```

#### get_vector_stores (279-309)

```python
def get_vector_stores(self, force_refresh: bool = False) -> Dict[str, str]:
    """Vector Store一覧を取得（キャッシュ機能付き）

    - 5分間のキャッシュ有効期限
    - force_refreshで強制更新
    """
```

---

## 4. ModernRAGManager

### 4.1 概要

Responses APIとfile_searchツールを使用したRAG検索を担当するクラス。

### 4.2 search_with_responses_api (472-561)

```python
def search_with_responses_api(
    self, query: str, store_name: str, store_id: str, **kwargs
) -> Tuple[str, Dict[str, Any]]:
    """最新Responses API + file_searchツールを使用した検索"""
```

**file_searchツール設定**:
```python
file_search_tool_dict: Dict[str, Any] = {
    "type": "file_search",
    "vector_store_ids": [store_id]
}

# オプション設定
if max_results and isinstance(max_results, int):
    file_search_tool_dict["max_num_results"] = max_results
if filters is not None:
    file_search_tool_dict["filters"] = filters
```

**Responses API呼び出し**:
```python
response = openai_client.responses.create(
    model=selected_model,
    input=query,
    tools=[file_search_tool_dict],
    include=include_params if include_params else None
)
```

### 4.3 search_with_agent_sdk (563-620)

```python
def search_with_agent_sdk(
    self, query: str, store_name: str, store_id: str
) -> Tuple[str, Dict[str, Any]]:
    """Agent SDKを使用した検索（簡易版）

    - セッション管理機能のみ
    - file_searchはResponses APIに委譲
    - エラー時はResponses APIにフォールバック
    """
```

### 4.4 ヘルパーメソッド

| メソッド | 行 | 説明 |
|---------|-----|------|
| `_extract_response_text()` | 630-650 | レスポンスからテキストを抽出 |
| `_extract_citations()` | 652-672 | ファイル引用情報を抽出 |
| `_extract_tool_calls()` | 674-690 | ツール呼び出し情報を抽出 |

---

## 5. 日本語回答生成

### 5.1 generate_enhanced_response (954-1034)

```python
def generate_enhanced_response(
    query: str, search_result: str, has_result: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """検索結果を基に日本語回答を生成

    - ChatCompletion APIを使用
    - 検索結果がある場合/ない場合で異なるプロンプト
    - モデル別パラメータ調整
    """
```

### 5.2 モデル別パラメータ調整

```python
# temperature非対応モデル
no_temperature_models = any(
    prefix in model_lower
    for prefix in ['o1-', 'o3-', 'o4-', 'gpt-5-mini']
)

# max_completion_tokens対応モデル
if any(prefix in model_lower
       for prefix in ['gpt-5-mini', 'gpt-4.1', 'gpt-5', 'o1-', 'o3-', 'o4-']):
    completion_params["max_completion_tokens"] = 2000
else:
    completion_params["max_tokens"] = 2000
```

---

## 6. UI構成

### 6.1 サイドバー

| セクション | 内容 |
|-----------|------|
| Vector Store選択 | 12種類のVector Storeから選択 |
| モデル選択 | AppConfig.AVAILABLE_MODELSから選択 |
| 検索オプション | 最大結果数、引用表示、Agent SDK使用 |
| Vector Store管理 | 最新情報更新、デバッグ情報表示 |
| システム情報 | 利用可能な機能一覧 |
| テスト用質問 | 選択されたVector Storeに対応した質問 |

### 6.2 メインエリア

| セクション | 内容 |
|-----------|------|
| 使い方 | Expanderで表示 |
| 質問入力 | テキストエリア + 検索実行ボタン |
| 質問例 | クリックで質問入力欄に設定 |
| 検索結果 | 英語回答 + 引用 + メタデータ + 日本語回答 |
| 検索履歴 | 最新10件をExpander形式で表示 |

### 6.3 セッション状態

```python
# initialize_session_state() (700-721)
session_state = {
    'search_history': [],           # 検索履歴（最新50件）
    'current_query': "",            # 現在の質問
    'selected_store': "...",        # 選択Vector Store
    'selected_model': "gpt-4o-mini", # 選択モデル
    'use_agent_sdk': False,         # Agent SDK使用フラグ
    'search_options': {
        'max_results': 20,
        'include_results': True,
        'show_citations': True
    },
    'auto_refresh_stores': True     # 自動更新フラグ
}
```

---

## 7. テスト質問

### 7.1 質問マッピング (766-791)

```python
store_question_mapping = {
    "Customer Support FAQ": test_questions_en,
    "Science & Technology Q&A": test_questions_2_en,
    "Medical Q&A": test_questions_3_en,
    "Legal Q&A": test_questions_4_en,
    "Unified Knowledge Base": test_questions_unified_en,
    "CC News Q&A (LLM)": test_questions_cc_news_en,
    "CC News Q&A (Coverage)": test_questions_cc_news_en,
    "CC News Q&A (Hybrid)": test_questions_cc_news_en,
}
```

### 7.2 質問カテゴリ

| カテゴリ | 質問例 |
|---------|--------|
| Customer Support | "How do I create a new account?" |
| Science & Technology | "What are the latest trends in artificial intelligence?" |
| Medical | "How to prevent high blood pressure?" |
| Legal | "What are the important clauses in contracts?" |
| CC News | "What are the latest developments in artificial intelligence?" |
| Unified | 全ドメインから混合 |

---

## 8. 使用方法

### 8.1 環境設定

```bash
# OpenAI APIキー設定（必須）
export OPENAI_API_KEY='your-api-key-here'

# 永続化
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc

# 必要なライブラリ
pip install streamlit openai
pip install openai-agents  # オプション
```

### 8.2 アプリケーション起動

```bash
streamlit run a34_rag_search_cloud_vs.py --server.port=8503
```

### 8.3 基本操作

1. **Vector Storeの選択**: 左サイドバーでVector Storeを選択
2. **質問の入力**: テスト質問から選択、または直接入力
3. **検索実行**: 「🔍 検索実行」ボタンをクリック
4. **結果確認**: 英語検索結果 + 日本語回答（自動生成）

### 8.4 Vector Store管理

```
サイドバー → Vector Store管理
├── 🔄 最新情報に更新: OpenAI APIから最新一覧を取得
├── 📊 デバッグ情報表示: キャッシュ・API情報を確認
└── 📁 設定ファイル表示: vector_stores.jsonの内容を表示
```

---

## 9. 設定ファイル

### 9.1 vector_stores.json

```json
{
  "vector_stores": {
    "CC News Q&A (a02_llm)": "vs_xxx...",
    "CC News Q&A (a03_rule)": "vs_yyy...",
    "CC News Q&A (a10_hybrid)": "vs_zzz...",
    "Livedoor Q&A (a02_llm)": "vs_aaa...",
    "Livedoor Q&A (a03_rule)": "vs_bbb...",
    "Livedoor Q&A (a10_hybrid)": "vs_ccc...",
    "Customer Support FAQ": "vs_ddd...",
    "Science & Technology Q&A": "vs_eee...",
    "Medical Q&A": "vs_fff...",
    "Legal Q&A": "vs_ggg...",
    "Trivia Q&A": "vs_hhh...",
    "Unified Knowledge Base": "vs_iii..."
  },
  "last_updated": "2025-11-27T14:30:00",
  "source": "a34_rag_search_cloud_vs.py",
  "version": "1.1"
}
```

### 9.2 設定ファイル更新タイミング

- アプリ起動時（自動更新が有効な場合）
- 「最新情報に更新」ボタンクリック時
- キャッシュ有効期限（5分）切れ時

---

## 10. トラブルシューティング

### 10.1 重複ID選択エラー

**症状**: 古いVector Store IDが選択される

**対処**:
1. サイドバー「Vector Store管理」→「最新情報に更新」
2. デバッグ情報で選択されたIDを確認
3. ログで最新作成日時のIDが選択されているか確認

### 10.2 APIキーエラー

**症状**: "OpenAI API キーの設定に問題があります"

**対処**:
```bash
# 環境変数確認
echo $OPENAI_API_KEY

# 設定
export OPENAI_API_KEY='your-api-key-here'

# 永続化
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 10.3 Vector Store IDエラー

**症状**: "Vector Store ID が見つかりません"

**対処**:
1. 「最新情報に更新」ボタンをクリック
2. a31_make_cloud_vector_store_vsid.py で新規作成後は更新が必要
3. 設定ファイル（vector_stores.json）を確認

### 10.4 検索結果なし

**症状**: "レスポンステキストの抽出に失敗"

**対処**:
1. Vector Store IDが正しいか確認
2. 選択したVector Storeにデータが存在するか確認
3. 質問内容がVector Storeのドメインに適合しているか確認

### 10.5 日本語回答生成エラー

**症状**: "回答生成エラー"

**対処**:
1. 選択モデルがAPI制限に達していないか確認
2. モデルパラメータ（temperature, max_tokens）を確認
3. ネットワーク接続を確認

---

## 付録: メタデータ

| 項目 | 値 |
|------|-----|
| ファイル行数 | 1475行 |
| ポート | 8503（または8501） |
| キャッシュ有効期限 | 5分 |
| 検索履歴保持数 | 最新50件 |
| 対応Vector Store数 | 12種類 |

## 関連ファイル

| ファイル | 説明 |
|---------|------|
| a31_make_cloud_vector_store_vsid.py | Vector Store作成 |
| vector_stores.json | Vector Store設定ファイル |
| helper_rag.py | モデル設定管理（AppConfig） |
| agents SDK | セッション管理（オプション） |
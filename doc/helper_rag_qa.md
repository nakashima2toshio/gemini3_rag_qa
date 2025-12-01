# helper_rag_qa.py 技術仕様書

作成日: 2025-11-28 (最終更新: Gemini移行対応)

## 概要

RAG Q&A生成のための包括的ユーティリティモジュール。キーワード抽出、Q/A生成、セマンティックカバレッジ分析など、多数の専門クラスを提供します。
Google Gemini API (`gemini-2.0-flash`, `gemini-embedding-001`) に完全対応し、`UnifiedLLMClient` を介して OpenAI API との互換性も維持しています。

## ファイル情報

- **ファイル名**: helper_rag_qa.py
- **主要機能**: キーワード抽出、Q/A生成、カバレッジ分析、多言語対応
- **対応モデル**: Gemini 3 (2.0 Flash/Pro), Gemini 1.5, OpenAI GPT-4o

---

## アーキテクチャ

### モジュール構造

```
helper_rag_qa.py
├── インポート・設定 (L1-27)
│   ├── helper_llm (UnifiedLLMClient)
│   └── helper_embedding (EmbeddingClient)
│
├── キーワード抽出関連
│   ├── BestKeywordSelector: 3手法比較選択
│   ├── SmartKeywordSelector: 自動最適化
│   ├── QACountOptimizer: 最適Q/A数決定 (UnifiedLLMClient利用)
│   └── QAOptimizedExtractor: Q/A特化抽出
│
├── セマンティックカバレッジ関連
│   ├── SemanticCoverage: 網羅性測定 (gemini-embedding-001利用)
│   └── QAGenerationConsiderations: 生成前チェック
│
├── データモデル
│   ├── QAPair (Pydantic)
│   └── QAPairsList (Pydantic)
│
├── Q/A生成クラス
│   ├── LLMBasedQAGenerator (UnifiedLLMClient利用)
│   ├── ChainOfThoughtQAGenerator
│   ├── RuleBasedQAGenerator
│   ├── TemplateBasedQAGenerator
│   ├── HybridQAGenerator
│   ├── AdvancedQAGenerationTechniques
│   ├── QAGenerationOptimizer
│   └── OptimizedHybridQAGenerator
│
└── バッチ処理クラス
    └── BatchHybridQAGenerator
```

---

## 主要クラス詳細

## 1. キーワード抽出関連クラス

### 1.3 QACountOptimizer

最適なQ/A生成数を動的に決定します。
`UnifiedLLMClient` を使用して、Gemini モデルのトークナイザーに基づいた正確なトークンカウントを行います。

```python
def __init__(self, llm_model="gemini-2.0-flash"):
    self.unified_client = create_llm_client(provider="gemini", default_model=llm_model)
    # ...
```

## 2. セマンティックカバレッジ関連クラス

### 2.1 SemanticCoverage

意味的な網羅性を測定します。
Embedding モデルとして **`gemini-embedding-001` (3072次元)** を標準使用し、より高精度な類似度計算を実現しています。

#### 特徴

*   **Gemini Embedding**: `create_embedding_client(provider="gemini")` を使用して初期化されます。
*   **トークンカウント**: `UnifiedLLMClient.count_tokens` を使用して Gemini モデルに準拠したカウントを行います。
*   **強制分割**: テキストの物理的な分割処理 (`_force_split_sentence`) には、デコード機能が必要なため `tiktoken` (cl100k_base) を併用しています。

```python
class SemanticCoverage:
    def __init__(self, embedding_model="gemini-embedding-001"):
        self.embedding_client = create_embedding_client(provider="gemini")
        self.unified_client = create_llm_client(provider="gemini")
        self.tokenizer = tiktoken.get_encoding("cl100k_base") # 強制分割用
        # ...
```

## 4. Q/A生成クラス

### 4.1 LLMBasedQAGenerator

LLMを使用したQ/A生成の基底クラスです。`UnifiedLLMClient` を使用することで、Gemini と OpenAI の両方のモデルをサポートします。構造化出力（Pydantic）にも対応しています。

### 4.7 BatchHybridQAGenerator

バッチ処理に最適化されたクラスです。Gemini API の特性（レート制限、コンテキストウィンドウ）を考慮したバッチサイズで処理を行います。

#### 品質重視モード（quality_mode=True）

*   `gemini-embedding-001` を用いた高精度なカバレッジ分析に基づき、目標カバレッジ (95%) を達成するまで生成を最適化します。

---

## パフォーマンス最適化

### 1. Gemini API の活用

*   **高速な推論**: Gemini 2.0 Flash を使用することで、大量のQ/A生成を高速かつ低コストに実行可能。
*   **高次元Embedding**: 3072次元のベクトルにより、意味的なマッチング精度が向上。

### 2. バッチ処理による効率化

*   Embedding生成: Gemini API の制限に合わせてバッチサイズ（通常100）を制御。
*   LLM生成: 複数のチャンクをまとめて処理し、API呼び出し回数を削減。

---

## 依存ライブラリ

*   `google-generativeai`: Gemini API 利用
*   `openai`: OpenAI API 利用（互換性のため）
*   `tiktoken`: トークン操作（補助的利用）
*   `spacy`, `mecab-python3`: 自然言語処理
*   `pydantic`: データモデル定義

---

## 更新履歴

**2025-11-28 (Gemini移行対応)**:
- `UnifiedLLMClient`, `EmbeddingClient` (helper_llm, helper_embedding) への依存に変更。
- デフォルトモデルを `gemini-2.0-flash`, `gemini-embedding-001` に更新。
- `SemanticCoverage` のトークンカウントロジックを Gemini 対応に修正。

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
a02_make_qa_para.py - preprocessedファイルからQ/Aペア生成
=====================================================
OUTPUTフォルダ内のpreprocessedファイルから自動的にQ/Aペアを生成

対応ファイル:
- OUTPUT/preprocessed_cc_news.csv (CC-News英語ニュース、7,376件)
- OUTPUT/preprocessed_japanese_text.csv (日本語Webテキスト)
- OUTPUT/preprocessed_wikipedia_ja.csv (Wikipedia日本語版)
- OUTPUT/preprocessed_livedoor.csv (Livedoorニュース日本語、7,376件)

使用方法:
    python a02_make_qa_para.py [--dataset DATASET_TYPE] [--model MODEL_NAME] [--output OUTPUT_DIR]

例:
本番運用向け：
実行コマンド：
python a02_make_qa_para.py \
    --dataset cc_news \
    --batch-chunks 3 \
    --merge-chunks \
    --min-tokens 100 \
    --max-tokens 300 \
    --model gpt-4o-mini \
    --analyze-coverage

# 全7,352文書の処理は約212時間（8.8日）かかると予測されます。
# 実用上は--max-docsで制限するか、データを分割して実行することを強く推奨します。
# 以下で、3.3分：
redis-cli FLUSHDB && ./start_celery.sh restart -w 24

python a02_make_qa_single.py \
  --dataset livedoor \
  --batch-chunks 3 \
  --merge-chunks \
  --min-tokens 100 \
  --max-tokens 300 \
  --model gpt-4o-mini \
  --max-docs 2 \
  --analyze-coverage


[OLD_Command]
python a02_make_qa_para.py \
      --dataset cc_news \
      --batch-chunks 5 \
      --merge-chunks \
      --min-tokens 150 \
      --max-tokens 400 \
      --model gpt-4o-mini \
      --analyze-coverage


実行時間の見積もり

| 項目      | 値                     |
|---------|-----------------------|
| 処理文書数   | 497件（全件）              |
| チャンク数   | ~1,825個 → 統合後 ~1,820個 |
| API呼び出し | 約365回（バッチサイズ5）        |
| 推定実行時間  | 60-75分                |
| カバレージ分析 | +3-5分                 |
| 合計      | 約65-80分               |


  生成Q/Aペア数: 525
  保存ファイル:
  - Q/A (JSON): qa_output/qa_pairs_cc_news_20251020_143052.json
  - Q/A (CSV): qa_output/qa_pairs_cc_news_20251020_143052.csv
  - カバレージ: qa_output/coverage_cc_news_20251020_143052.json
  - サマリー: qa_output/summary_cc_news_20251020_143052.json
    • 文書後半部分のカバレージがやや低い（85.0%）
    • Shortチャンクで追加Q/A生成の余地あり
 ----------------------------------------------------------
    python a02_make_qa_para.py --dataset cc_news --batch-chunks 5 --merge-chunks --analyze-coverage

    # データセット別テスト実行
    python a02_make_qa_para.py --dataset cc_news --model gpt-4o-mini --batch-chunks 5  --analyze-coverage --max-docs 10
    python a02_make_qa_para.py --dataset wikipedia_ja --model gpt-4o-mini  --analyze-coverage --max-docs 10
    python a02_make_qa_para.py --dataset japanese_text --model gpt-4o-mini  --analyze-coverage --max-docs 10
    python a02_make_qa_para.py --dataset livedoor --model gpt-4o-mini --analyze-coverage --max-docs 100
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import tiktoken
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import re
from collections import Counter

# ローカルモジュール
from a03_rag_qa_coverage_improved import SemanticCoverage

# 環境変数読み込み
load_dotenv()

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================
# Pydantic モデル定義
# ==========================================

class QAPair(BaseModel):
    """Q/Aペアのデータモデル"""
    question: str
    answer: str
    question_type: str
    source_chunk_id: Optional[str] = None
    dataset_type: Optional[str] = None
    auto_generated: bool = False


class QAPairsResponse(BaseModel):
    """Q/Aペア生成レスポンス"""
    qa_pairs: List[QAPair]


# ==========================================
# データセット設定
# ==========================================

DATASET_CONFIGS = {
    "cc_news": {
        "name": "CC-News英語ニュース",
        "file": "OUTPUT/preprocessed_cc_news.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "en",
        "chunk_size": 300,  # トークン数
        "qa_per_chunk": 5,  # チャンクあたりのQ/A数（改善: 3→5）
    },
    "japanese_text": {
        "name": "日本語Webテキスト",
        "file": "OUTPUT/preprocessed_japanese_text.csv",
        "text_column": "Combined_Text",
        "title_column": None,
        "lang": "ja",
        "chunk_size": 200,
        "qa_per_chunk": 2,
    },
    "wikipedia_ja": {
        "name": "Wikipedia日本語版",
        "file": "OUTPUT/preprocessed_wikipedia_ja.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja",
        "chunk_size": 250,
        "qa_per_chunk": 3,
    },
    "livedoor": {
        "name": "Livedoorニュースコーパス",
        "file": "OUTPUT/preprocessed_livedoor.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja",
        "chunk_size": 200,  # 日本語ニュース記事に適したサイズ
        "qa_per_chunk": 4,  # ニュース記事の情報密度を考慮
    }
}


# ==========================================
# MeCab キーワード抽出クラス (sample_regex_mecab.py から移植)
# ==========================================

class KeywordExtractor:
    """
    MeCabと正規表現を統合したキーワード抽出クラス
    MeCabが利用可能な場合は複合名詞抽出を優先し、
    利用不可の場合は正規表現版に自動フォールバック
    """

    def __init__(self, prefer_mecab: bool = True):
        """
        Args:
            prefer_mecab: MeCabを優先的に使用するか（デフォルト: True）
        """
        self.prefer_mecab = prefer_mecab
        self.mecab_available = self._check_mecab_availability()

        # ストップワード定義
        self.stopwords = {
            'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん',
            'ます', 'です', 'ある', 'いる', 'する', 'なる', 'できる',
            'いう', '的', 'な', 'に', 'を', 'は', 'が', 'で', 'と',
            'の', 'から', 'まで', '等', 'など', 'よる', 'おく', 'くる'
        }

        if self.mecab_available:
            logger.info("✅ MeCabが利用可能です（複合名詞抽出モード）")
        else:
            logger.info("⚠️ MeCabが利用できません（正規表現モード）")

    def _check_mecab_availability(self) -> bool:
        """MeCabの利用可能性をチェック"""
        try:
            import MeCab
            # 実際にインスタンス化して動作確認
            tagger = MeCab.Tagger()
            tagger.parse("テスト")
            return True
        except (ImportError, RuntimeError):
            return False

    def extract(self, text: str, top_n: int = 5) -> List[str]:
        """
        テキストからキーワードを抽出（自動フォールバック対応）

        Args:
            text: 分析対象テキスト
            top_n: 抽出するキーワード数

        Returns:
            キーワードリスト
        """
        if self.mecab_available and self.prefer_mecab:
            try:
                keywords = self._extract_with_mecab(text, top_n)
                if keywords:  # 空でなければ成功
                    return keywords
            except Exception as e:
                logger.warning(f"⚠️ MeCab抽出エラー: {e}")

        # フォールバック: 正規表現版
        return self._extract_with_regex(text, top_n)

    def _extract_with_mecab(self, text: str, top_n: int) -> List[str]:
        """MeCabを使用した複合名詞抽出"""
        import MeCab

        tagger = MeCab.Tagger()
        node = tagger.parseToNode(text)

        # 複合名詞の抽出
        compound_buffer = []
        compound_nouns = []

        while node:
            features = node.feature.split(',')
            pos = features[0]  # 品詞

            if pos == '名詞':
                compound_buffer.append(node.surface)
            else:
                # 名詞以外が来たらバッファをフラッシュ
                if compound_buffer:
                    compound_noun = ''.join(compound_buffer)
                    if len(compound_noun) > 0:
                        compound_nouns.append(compound_noun)
                    compound_buffer = []

            node = node.next

        # 最後のバッファをフラッシュ
        if compound_buffer:
            compound_noun = ''.join(compound_buffer)
            if len(compound_noun) > 0:
                compound_nouns.append(compound_noun)

        # フィルタリングと頻度カウント
        return self._filter_and_count(compound_nouns, top_n)

    def _extract_with_regex(self, text: str, top_n: int) -> List[str]:
        """正規表現を使用したキーワード抽出"""
        # カタカナ語、漢字複合語、英数字を抽出
        pattern = r'[ァ-ヴー]{2,}|[一-龥]{2,}|[A-Za-z]{2,}[A-Za-z0-9]*'
        words = re.findall(pattern, text)

        # フィルタリングと頻度カウント
        return self._filter_and_count(words, top_n)

    def _filter_and_count(self, words: List[str], top_n: int) -> List[str]:
        """頻度ベースのフィルタリング"""
        # ストップワード除外
        filtered = [w for w in words if w not in self.stopwords and len(w) > 1]

        # 頻度カウント
        word_freq = Counter(filtered)

        # 上位N件を返す
        return [word for word, freq in word_freq.most_common(top_n)]


# グローバルなKeywordExtractorインスタンス（一度だけ初期化）
_keyword_extractor = None

def get_keyword_extractor() -> KeywordExtractor:
    """KeywordExtractorのシングルトンインスタンスを取得"""
    global _keyword_extractor
    if _keyword_extractor is None:
        _keyword_extractor = KeywordExtractor()
    return _keyword_extractor


# ==========================================
# MeCab ベースのチャンク作成関数
# ==========================================

def create_mecab_chunks(text: str, lang: str = "ja", max_tokens: int = 200, chunk_id_prefix: str = "chunk") -> List[Dict]:
    """
    セマンティック分割によるチャンク作成（段落優先・MeCab対応）

    helper_rag_qa.pyのSemanticCoverage.create_semantic_chunks()を使用し、
    段落境界を最優先したセマンティック分割を実行。
    日本語の場合はMeCabによる高精度な文境界検出を自動適用。

    Args:
        text: 分割対象テキスト
        lang: 言語（"ja" or "en"）※現在は自動判定
        max_tokens: チャンクの最大トークン数
        chunk_id_prefix: チャンクIDのプレフィックス

    Returns:
        チャンクのリスト
    """
    # SemanticCoverageを使用してセマンティック分割を実行
    from helper_rag_qa import SemanticCoverage

    semantic_analyzer = SemanticCoverage(embedding_model="text-embedding-3-small")

    # 段落優先のセマンティック分割を実行
    # prefer_paragraphs=True: 段落境界を最優先
    # max_tokens: チャンクの最大トークン数
    # min_tokens: 最小トークン数（小さすぎるチャンクは自動マージ）
    # verbose=False: 詳細ログを抑制
    semantic_chunks = semantic_analyzer.create_semantic_chunks(
        document=text,
        max_tokens=max_tokens,
        min_tokens=50,  # 最小トークン数
        prefer_paragraphs=True,  # 段落優先モード
        verbose=False
    )

    # SemanticCoverageの出力形式をa02の形式に変換
    chunks = []
    tokenizer = tiktoken.get_encoding("cl100k_base")

    for i, semantic_chunk in enumerate(semantic_chunks):
        chunk_text = semantic_chunk['text']
        chunk_tokens = len(tokenizer.encode(chunk_text))

        chunks.append({
            'id': f"{chunk_id_prefix}_{i}",
            'text': chunk_text,
            'tokens': chunk_tokens,
            'type': semantic_chunk.get('type', 'unknown'),  # paragraph/sentence_group/forced_split
            'sentences': semantic_chunk.get('sentences', [])
        })

    return chunks


# ==========================================
# データ読み込み・前処理
# ==========================================

def load_preprocessed_data(dataset_type: str) -> pd.DataFrame:
    """preprocessedデータを読み込み
    Args:
        dataset_type: データセットタイプ
    Returns:
        読み込んだDataFrame
    """
    config = DATASET_CONFIGS.get(dataset_type)
    if not config:
        raise ValueError(f"未対応のデータセット: {dataset_type}")

    file_path = config["file"]
    if not Path(file_path).exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    logger.info(f"データ読み込み中: {file_path}")
    df = pd.read_csv(file_path)

    # 必要なカラムの確認
    text_col = config["text_column"]
    if text_col not in df.columns:
        raise ValueError(f"テキストカラム '{text_col}' が見つかりません")

    # 空のテキストを除外
    df = df[df[text_col].notna() & (df[text_col].str.strip() != '')]

    logger.info(f"読み込み完了: {len(df)}件のデータ")
    return df


def create_document_chunks(df: pd.DataFrame, dataset_type: str, max_docs: Optional[int] = None) -> List[Dict]:
    """DataFrameから文書チャンクを作成（MeCabベース）
    Args:
        df: データフレーム
        dataset_type: データセットタイプ
        max_docs: 処理する最大文書数
    Returns:
        チャンクのリスト
    """
    config = DATASET_CONFIGS[dataset_type]
    text_col = config["text_column"]
    title_col = config.get("title_column")
    chunk_size = config["chunk_size"]
    lang = config["lang"]

    all_chunks = []

    # 処理する文書数を制限
    docs_to_process = df.head(max_docs) if max_docs else df

    logger.info(f"チャンク作成開始: {len(docs_to_process)}件の文書（MeCabベース）")

    for idx, row in docs_to_process.iterrows():
        # row[text_col]はSeriesやオブジェクトの可能性があるため、明示的にstrに変換
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""

        # タイトルがある場合は含める
        if title_col and title_col in row and pd.notna(row[title_col]):
            doc_id = f"{dataset_type}_{idx}_{str(row[title_col])[:30]}"
        else:
            doc_id = f"{dataset_type}_{idx}"

        # MeCabベースのチャンク作成を使用
        try:
            chunk_id_prefix = f"{doc_id}_chunk"
            chunks = create_mecab_chunks(
                text=text,
                lang=lang,
                max_tokens=chunk_size,
                chunk_id_prefix=chunk_id_prefix
            )

            # 各チャンクにメタデータを追加
            for i, chunk in enumerate(chunks):
                chunk['doc_id'] = doc_id
                chunk['doc_idx'] = idx
                chunk['chunk_idx'] = i
                chunk['dataset_type'] = dataset_type
                all_chunks.append(chunk)

        except Exception as e:
            logger.warning(f"チャンク作成エラー (doc {idx}): {e}")
            continue

    logger.info(f"チャンク作成完了: {len(all_chunks)}個のチャンク（MeCabベース）")
    return all_chunks


def merge_small_chunks(chunks: List[Dict], min_tokens: int = 150, max_tokens: int = 400) -> List[Dict]:
    """小さいチャンクを統合して適切なサイズにする
    Args:
        chunks: チャンクのリスト
        min_tokens: このトークン数未満のチャンクは統合対象
        max_tokens: 統合後の最大トークン数
    Returns:
        統合されたチャンクのリスト
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    merged_chunks = []
    current_merge = None

    for chunk in chunks:
        chunk_tokens = len(tokenizer.encode(chunk['text']))

        # 大きいチャンクはそのまま追加
        if chunk_tokens >= min_tokens:
            if current_merge:
                merged_chunks.append(current_merge)
                current_merge = None
            merged_chunks.append(chunk)
        else:
            # 小さいチャンクは統合候補
            if current_merge is None:
                current_merge = chunk.copy()
                current_merge['merged'] = True
                current_merge['original_chunks'] = [chunk['id']]
            else:
                # 統合可能かチェック
                merge_tokens = len(tokenizer.encode(current_merge['text']))
                if merge_tokens + chunk_tokens <= max_tokens:
                    # 同じ文書からのチャンクのみ統合
                    if current_merge.get('doc_id') == chunk.get('doc_id'):
                        current_merge['text'] += "\n\n" + chunk['text']
                        current_merge['original_chunks'].append(chunk['id'])
                        if 'chunk_idx' in current_merge:
                            current_merge['chunk_idx'] = f"{current_merge['chunk_idx']}-{chunk['chunk_idx']}"
                    else:
                        # 異なる文書の場合は別々に
                        merged_chunks.append(current_merge)
                        current_merge = chunk.copy()
                        current_merge['merged'] = True
                        current_merge['original_chunks'] = [chunk['id']]
                else:
                    # サイズオーバーの場合は現在の統合を追加して新規開始
                    merged_chunks.append(current_merge)
                    current_merge = chunk.copy()
                    current_merge['merged'] = True
                    current_merge['original_chunks'] = [chunk['id']]

    # 最後の統合チャンクを追加
    if current_merge:
        merged_chunks.append(current_merge)

    logger.info(f"チャンク統合: {len(chunks)}個 → {len(merged_chunks)}個 ({100*(1-len(merged_chunks)/len(chunks)):.1f}%削減)")
    return merged_chunks


# ==========================================
# Q/Aペア生成
# ==========================================

def determine_qa_count(chunk: Dict, config: Dict) -> int:
    """チャンクに最適なQ/A数を決定（改善版：動的調整）
    Args:
        chunk: チャンクデータ
        config: データセット設定
    Returns:
        Q/Aペア数
    """
    base_count = config["qa_per_chunk"]
    tokenizer = tiktoken.get_encoding("cl100k_base")
    token_count = len(tokenizer.encode(chunk['text']))

    # チャンク位置を考慮（文書後半の補正）
    chunk_position = chunk.get('chunk_idx', 0)

    # トークン数に基づく基本Q&A数決定（改善版）
    if token_count < 50:
        qa_count = 2  # 旧: 1 → 新: 2（短いチャンクでも最低2個）
    elif token_count < 100:
        qa_count = 3  # 旧: 2 → 新: 3（Shortチャンク強化）
    elif token_count < 200:
        qa_count = base_count + 1  # Mediumは+1
    elif token_count < 300:
        qa_count = base_count + 2  # Longチャンクは+2
    else:
        qa_count = base_count + 3  # 超長文は+3

    # 文書後半の位置バイアス補正（6番目以降のチャンクは+1）
    if isinstance(chunk_position, int) and chunk_position >= 5:
        qa_count += 1

    return min(qa_count, 8)  # 上限を8に設定


def generate_qa_pairs_for_batch(
    chunks: List[Dict],
    config: Dict,
    model: str = "gpt-4o-mini",
    client: Optional[OpenAI] = None
) -> List[Dict]:
    """複数チャンクから一度にQ/Aペアを生成（3チャンクバッチ処理対応）
    Args:
        chunks: チャンクデータのリスト（最大3個）
        config: データセット設定
        model: 使用するモデル
        client: OpenAIクライアント
    Returns:
        生成されたQ/Aペアのリスト
    """
    if client is None:
        client = OpenAI()

    if len(chunks) == 0:
        return []

    # 単一チャンクの場合は従来の処理
    if len(chunks) == 1:
        return generate_qa_pairs_for_chunk(chunks[0], config, model, client)

    lang = config["lang"]
    all_qa_pairs = []

    # 言語別のプロンプト設定
    if lang == "ja":
        system_prompt = """あなたは教育コンテンツ作成の専門家です。
複数の日本語テキストから、学習効果の高いQ&Aペアを生成してください。

生成ルール:
1. 質問は明確で具体的に
2. 回答は簡潔で正確に（1-2文程度）
3. テキストの内容に忠実に
4. 多様な観点から質問を作成"""

        # 複数チャンクを結合してプロンプト構築
        combined_text = ""
        chunks_data = {}
        total_pairs = 0

        for i, chunk in enumerate(chunks, 1):
            num_pairs = determine_qa_count(chunk, config)
            total_pairs += num_pairs
            chunk_text = chunk['text']

            # 長すぎる場合は短縮
            if len(chunk_text) > 1000:
                chunk_text = chunk_text[:1000] + "..."

            combined_text += f"\n\n【テキスト{i}】\n{chunk_text}"
            chunks_data[f"chunk_{i}"] = {"num_pairs": num_pairs, "chunk": chunk}

        user_prompt = f"""以下の{len(chunks)}個のテキストから、合計{total_pairs}個のQ&Aペアを生成してください。
{combined_text}

質問タイプ:
- fact: 事実確認型（〜は何ですか？）
- reason: 理由説明型（なぜ〜ですか？）
- comparison: 比較型（〜と〜の違いは？）
- application: 応用型（〜はどのように活用されますか？）

JSON形式で出力:
{{
  "qa_pairs": [
    {{
      "question": "質問文",
      "answer": "回答文",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""

    else:
        system_prompt = """You are an expert in educational content creation.
Generate high-quality Q&A pairs from multiple English texts.

Generation rules:
1. Questions should be clear and specific
2. Answers should be concise and accurate (1-2 sentences)
3. Stay faithful to the text content
4. Create questions from diverse perspectives"""

        # 複数チャンクを結合してプロンプト構築
        combined_text = ""
        chunks_data = {}
        total_pairs = 0

        for i, chunk in enumerate(chunks, 1):
            num_pairs = determine_qa_count(chunk, config)
            total_pairs += num_pairs
            chunk_text = chunk['text']

            # 長すぎる場合は短縮
            if len(chunk_text) > 1000:
                chunk_text = chunk_text[:1000] + "..."

            combined_text += f"\n\n【Text {i}】\n{chunk_text}"
            chunks_data[f"chunk_{i}"] = {"num_pairs": num_pairs, "chunk": chunk}

        user_prompt = f"""Generate {total_pairs} Q&A pairs from the following {len(chunks)} texts.
{combined_text}

Question types:
- fact: Factual questions (What is...?)
- reason: Explanatory questions (Why...?)
- comparison: Comparative questions (What's the difference...?)
- application: Application questions (How is... used?)

Output in JSON format:
{{
  "qa_pairs": [
    {{
      "question": "question text",
      "answer": "answer text",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""

    try:
        # 最新のOpenAI Responses API (client.responses.parse) を使用
        # システムプロンプトとユーザープロンプトを統合
        combined_input = f"{system_prompt}\n\n{user_prompt}"

        response = client.responses.parse(
            input=combined_input,
            model=model,
            text_format=QAPairsResponse,  # Pydanticモデルを直接指定
            max_output_tokens=4000  # バッチ処理のため増加（3チャンク対応）
        )

        # レスポンスの解析
        for output in response.output:
            if output.type == "message":
                for item in output.content:
                    if item.type == "output_text" and item.parsed:
                        # パース済みデータを取得
                        parsed_data = item.parsed

                        # 生成されたQ/Aペアを各チャンクに分配
                        # 各チャンクに期待される数だけQ/Aを割り当て
                        qa_index = 0
                        for i, chunk in enumerate(chunks, 1):
                            chunk_key = f"chunk_{i}"
                            expected_pairs = chunks_data[chunk_key]["num_pairs"]

                            # このチャンクに割り当てるQ/Aペアを取得
                            for _ in range(expected_pairs):
                                if qa_index < len(parsed_data.qa_pairs):
                                    qa_data = parsed_data.qa_pairs[qa_index]
                                    qa = {
                                        "question": qa_data.question,
                                        "answer": qa_data.answer,
                                        "question_type": qa_data.question_type,
                                        "source_chunk_id": chunk.get('id', ''),
                                        "doc_id": chunk.get('doc_id', ''),
                                        "dataset_type": chunk.get('dataset_type', ''),
                                        "chunk_idx": chunk.get('chunk_idx', 0)
                                    }
                                    all_qa_pairs.append(qa)
                                    qa_index += 1

        return all_qa_pairs

    except Exception as e:
        logger.error(f"バッチQ/A生成エラー: {e}")
        # フォールバック: 個別処理
        logger.info("フォールバック: チャンクを個別処理します")
        for chunk in chunks:
            try:
                qa_pairs = generate_qa_pairs_for_chunk(chunk, config, model, client)
                all_qa_pairs.extend(qa_pairs)
            except Exception as chunk_error:
                logger.error(f"チャンク個別処理エラー: {chunk_error}")
        return all_qa_pairs


def generate_qa_pairs_for_chunk(
    chunk: Dict,
    config: Dict,
    model: str = "gpt-4o-mini",
    client: Optional[OpenAI] = None
) -> List[Dict]:
    """単一チャンクからQ/Aペアを生成（後方互換性のため維持）
    Args:
        chunk: チャンクデータ
        config: データセット設定
        model: 使用するモデル
        client: OpenAIクライアント
    Returns:
        生成されたQ/Aペアのリスト
    """
    if client is None:
        client = OpenAI()

    num_pairs = determine_qa_count(chunk, config)
    lang = config["lang"]

    # 言語別のプロンプト設定
    if lang == "ja":
        system_prompt = """あなたは教育コンテンツ作成の専門家です。
与えられた日本語テキストから、学習効果の高いQ&Aペアを生成してください。

生成ルール:
1. 質問は明確で具体的に
2. 回答は簡潔で正確に（1-2文程度）
3. テキストの内容に忠実に
4. 多様な観点から質問を作成"""

        question_types_desc = """
- fact: 事実確認型（〜は何ですか？）
- reason: 理由説明型（なぜ〜ですか？）
- comparison: 比較型（〜と〜の違いは？）
- application: 応用型（〜はどのように活用されますか？）"""
    else:
        system_prompt = """You are an expert in educational content creation.
Generate high-quality Q&A pairs from the given English text.

Generation rules:
1. Questions should be clear and specific
2. Answers should be concise and accurate (1-2 sentences)
3. Stay faithful to the text content
4. Create questions from diverse perspectives"""

        question_types_desc = """
- fact: Factual questions (What is...?)
- reason: Explanatory questions (Why...?)
- comparison: Comparative questions (What's the difference...?)
- application: Application questions (How is... used?)"""

    # 言語に応じたユーザープロンプト
    if lang == "ja":
        user_prompt = f"""以下のテキストから{num_pairs}個のQ&Aペアを生成してください。

質問タイプ:
{question_types_desc}

テキスト:
{chunk['text']}

JSON形式で出力:
{{
  "qa_pairs": [
    {{
      "question": "質問文",
      "answer": "回答文",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""
    else:
        user_prompt = f"""Generate {num_pairs} Q&A pairs from the following text.

Question types:
{question_types_desc}

Text:
{chunk['text']}

Output in JSON format:
{{
  "qa_pairs": [
    {{
      "question": "question text",
      "answer": "answer text",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""

    try:
        # チャンクが長すぎる場合は短縮（日本語テキストは長い傾向があるため）
        max_chunk_length = 2000  # 文字数制限
        chunk_text = chunk['text']
        if len(chunk_text) > max_chunk_length:
            chunk_text = chunk_text[:max_chunk_length] + "..."
            logger.debug(f"チャンクを{max_chunk_length}文字に短縮")

        # プロンプトの再構築（短縮されたテキストを使用）
        if lang == "ja":
            user_prompt = f"""以下のテキストから{num_pairs}個のQ&Aペアを生成してください。

質問タイプ:
{question_types_desc}

テキスト:
{chunk_text}

JSON形式で出力:
{{
  "qa_pairs": [
    {{
      "question": "質問文",
      "answer": "回答文",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""
        else:
            user_prompt = f"""Generate {num_pairs} Q&A pairs from the following text.

Question types:
{question_types_desc}

Text:
{chunk_text}

Output in JSON format:
{{
  "qa_pairs": [
    {{
      "question": "question text",
      "answer": "answer text",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""

        # 最新のOpenAI Responses API (client.responses.parse) を使用
        combined_input = f"{system_prompt}\n\n{user_prompt}"

        response = client.responses.parse(
            input=combined_input,
            model=model,
            text_format=QAPairsResponse,  # Pydanticモデルを直接指定
            max_output_tokens=1000  # 出力トークン数を制限
        )

        # レスポンスの解析
        qa_pairs = []
        for output in response.output:
            if output.type == "message":
                for item in output.content:
                    if item.type == "output_text" and item.parsed:
                        # パース済みデータを取得
                        parsed_data = item.parsed

                        for qa_data in parsed_data.qa_pairs:
                            qa = {
                                "question": qa_data.question,
                                "answer": qa_data.answer,
                                "question_type": qa_data.question_type,
                                "source_chunk_id": chunk.get('id', ''),
                                "doc_id": chunk.get('doc_id', ''),
                                "dataset_type": chunk.get('dataset_type', ''),
                                "chunk_idx": chunk.get('chunk_idx', 0)
                            }
                            qa_pairs.append(qa)
        return qa_pairs

    except Exception as e:
        logger.error(f"Q/A生成エラー (chunk {chunk.get('id', 'unknown')}): {e}")
        return []


def generate_qa_for_dataset(
    chunks: List[Dict],
    dataset_type: str,
    model: str = "gpt-4o-mini",
    chunk_batch_size: int = 3,
    merge_chunks: bool = True,
    min_tokens: int = 150,
    max_tokens: int = 400
) -> List[Dict]:
    """データセット全体のQ/Aペア生成（改善版）
    Args:
        chunks: チャンクリスト
        dataset_type: データセットタイプ
        model: 使用するモデル
        chunk_batch_size: 1回のAPIで処理するチャンク数（1-5）
        merge_chunks: 小さいチャンクを統合するか
        min_tokens: 統合対象の最小トークン数
        max_tokens: 統合後の最大トークン数
    Returns:
        生成されたQ/Aペアのリスト
    """
    config = DATASET_CONFIGS[dataset_type]
    client = OpenAI()
    all_qa_pairs = []

    # チャンクの前処理（小さいチャンクの統合）
    if merge_chunks:
        processed_chunks = merge_small_chunks(chunks, min_tokens, max_tokens)
    else:
        processed_chunks = chunks

    total_chunks = len(processed_chunks)
    api_calls = (total_chunks + chunk_batch_size - 1) // chunk_batch_size

    logger.info(f"""
    Q/Aペア生成開始:
    - 元チャンク数: {len(chunks)}
    - 処理チャンク数: {total_chunks}
    - バッチサイズ: {chunk_batch_size}
    - API呼び出し予定: {api_calls}回
    - モデル: {model}
    """)

    # バッチ処理
    for i in range(0, total_chunks, chunk_batch_size):
        batch = processed_chunks[i:i+chunk_batch_size]
        batch_num = i // chunk_batch_size + 1
        total_batches = api_calls

        logger.info(f"バッチ {batch_num}/{total_batches} 処理中 ({len(batch)}チャンク)...")

        # リトライ機能付きQ/A生成
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if chunk_batch_size == 1:
                    # 単一チャンク処理
                    qa_pairs = generate_qa_pairs_for_chunk(batch[0], config, model, client)
                else:
                    # バッチ処理
                    qa_pairs = generate_qa_pairs_for_batch(batch, config, model, client)

                if qa_pairs:
                    all_qa_pairs.extend(qa_pairs)
                    logger.debug(f"バッチ {batch_num}: {len(qa_pairs)}個のQ/Aペア生成")
                break

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"バッチ {batch_num} 生成失敗: {e}")
                    # 最終試行失敗時は個別処理にフォールバック
                    logger.info("個別処理にフォールバック...")
                    for chunk in batch:
                        try:
                            qa_pairs = generate_qa_pairs_for_chunk(chunk, config, model, client)
                            if qa_pairs:
                                all_qa_pairs.extend(qa_pairs)
                        except Exception as chunk_error:
                            logger.error(f"チャンク処理エラー: {chunk_error}")
                else:
                    wait_time = 2 ** attempt
                    logger.warning(f"リトライ {attempt + 1}/{max_retries} (待機: {wait_time}秒)")
                    time.sleep(wait_time)

        # API制限対策（最後のバッチ以外で待機）
        if i + chunk_batch_size < total_chunks:
            time.sleep(0.2)  # 短縮（バッチ処理により呼び出し数が減っているため）

    logger.info(f"""
    Q/Aペア生成完了:
    - 生成されたQ/Aペア: {len(all_qa_pairs)}個
    - 実行されたAPI呼び出し: 約{api_calls}回
    """)

    return all_qa_pairs


# ==========================================
# カバレージ分析
# ==========================================

# データセット別最適閾値設定
OPTIMAL_THRESHOLDS = {
    "cc_news": {
        "strict": 0.80,
        "standard": 0.70,
        "lenient": 0.60
    },
    "japanese_text": {
        "strict": 0.75,
        "standard": 0.65,
        "lenient": 0.55
    },
    "wikipedia_ja": {
        "strict": 0.85,   # 専門的な内容 → 高い類似度要求
        "standard": 0.75,
        "lenient": 0.65
    },
    "livedoor": {
        "strict": 0.78,   # ニュース記事は具体的で厳密性が中程度
        "standard": 0.68, # cc_newsとwikipedia_jaの中間
        "lenient": 0.58
    }
}


def get_optimal_thresholds(dataset_type: str) -> Dict[str, float]:
    """データセット別の最適閾値を取得
    Args:
        dataset_type: データセットタイプ
    Returns:
        閾値辞書 {strict, standard, lenient}
    """
    return OPTIMAL_THRESHOLDS.get(dataset_type, {
        "strict": 0.8,
        "standard": 0.7,
        "lenient": 0.6
    })


def multi_threshold_coverage(coverage_matrix: np.ndarray, chunks: List[Dict],
                             qa_pairs: List[Dict], thresholds: Dict[str, float]) -> Dict:
    """複数閾値でカバレージを評価
    Args:
        coverage_matrix: カバレージ行列
        chunks: チャンクリスト
        qa_pairs: Q/Aペアリスト
        thresholds: 閾値辞書
    Returns:
        多段階カバレージ結果
    """
    results = {}
    max_similarities = coverage_matrix.max(axis=1)

    for level, threshold in thresholds.items():
        covered = sum(1 for s in max_similarities if s >= threshold)
        uncovered_chunks = [
            {
                "chunk_id": chunks[i].get("id", f"chunk_{i}"),
                "similarity": float(max_similarities[i]),
                "gap": float(threshold - max_similarities[i])
            }
            for i, sim in enumerate(max_similarities)
            if sim < threshold
        ]

        results[level] = {
            "threshold": threshold,
            "covered_chunks": covered,
            "coverage_rate": covered / len(chunks) if chunks else 0,
            "uncovered_count": len(uncovered_chunks),
            "uncovered_chunks": uncovered_chunks
        }

    return results


def analyze_chunk_characteristics_coverage(chunks: List[Dict], coverage_matrix: np.ndarray,
                                          qa_pairs: List[Dict], threshold: float = 0.7) -> Dict:
    """チャンク特性別のカバレージ分析
    Args:
        chunks: チャンクリスト
        coverage_matrix: カバレージ行列
        qa_pairs: Q/Aペアリスト
        threshold: 判定閾値
    Returns:
        チャンク特性別カバレージ結果
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    results = {
        "by_length": {},      # 長さ別
        "by_position": {},    # 位置別
        "summary": {}
    }

    # 1. 長さ別分析
    for i, chunk in enumerate(chunks):
        token_count = len(tokenizer.encode(chunk['text']))
        length_category = (
            "short" if token_count < 100 else
            "medium" if token_count < 200 else
            "long"
        )

        if length_category not in results["by_length"]:
            results["by_length"][length_category] = {
                "count": 0,
                "covered": 0,
                "avg_similarity": 0.0,
                "similarities": []
            }

        max_sim = coverage_matrix[i].max()
        results["by_length"][length_category]["count"] += 1
        results["by_length"][length_category]["similarities"].append(float(max_sim))

        if max_sim >= threshold:
            results["by_length"][length_category]["covered"] += 1

    # 平均類似度とカバレージ率を計算
    for length_cat in results["by_length"]:
        data = results["by_length"][length_cat]
        data["avg_similarity"] = float(np.mean(data["similarities"])) if data["similarities"] else 0.0
        data["coverage_rate"] = data["covered"] / data["count"] if data["count"] > 0 else 0.0
        # similaritiesは大きいので削除（メモリ節約）
        del data["similarities"]

    # 2. 位置別分析（文書の前半/中盤/後半）
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        position = (
            "beginning" if i < total_chunks * 0.33 else
            "middle" if i < total_chunks * 0.67 else
            "end"
        )

        if position not in results["by_position"]:
            results["by_position"][position] = {
                "count": 0,
                "covered": 0,
                "avg_similarity": 0.0,
                "similarities": []
            }

        max_sim = coverage_matrix[i].max()
        results["by_position"][position]["count"] += 1
        results["by_position"][position]["similarities"].append(float(max_sim))

        if max_sim >= threshold:
            results["by_position"][position]["covered"] += 1

    # 平均類似度とカバレージ率を計算
    for position in results["by_position"]:
        data = results["by_position"][position]
        data["avg_similarity"] = float(np.mean(data["similarities"])) if data["similarities"] else 0.0
        data["coverage_rate"] = data["covered"] / data["count"] if data["count"] > 0 else 0.0
        del data["similarities"]

    # 3. サマリー情報
    results["summary"] = {
        "total_chunks": len(chunks),
        "total_qa_pairs": len(qa_pairs),
        "threshold_used": threshold,
        "insights": []
    }

    # インサイト生成
    for length_cat, data in results["by_length"].items():
        if data["coverage_rate"] < 0.7:
            results["summary"]["insights"].append(
                f"{length_cat}チャンクのカバレージが低い（{data['coverage_rate']:.1%}）"
            )

    for position, data in results["by_position"].items():
        if data["coverage_rate"] < 0.7:
            results["summary"]["insights"].append(
                f"文書{position}部分のカバレージが低い（{data['coverage_rate']:.1%}）"
            )

    return results


def analyze_coverage(chunks: List[Dict], qa_pairs: List[Dict], dataset_type: str = "wikipedia_ja") -> Dict:
    """生成されたQ/Aペアのカバレージを分析（多段階カバレージ分析対応）
    Args:
        chunks: チャンクリスト
        qa_pairs: Q/Aペアリスト
        dataset_type: データセットタイプ（閾値自動設定に使用）
    Returns:
        カバレージ分析結果（多段階評価、チャンク特性分析を含む）
    """
    analyzer = SemanticCoverage()

    # 埋め込み生成
    logger.info("埋め込みベクトル生成中...")
    doc_embeddings = analyzer.generate_embeddings(chunks)

    qa_embeddings = []
    for qa in qa_pairs:
        qa_text = f"{qa['question']} {qa['answer']}"
        embedding = analyzer.generate_embedding(qa_text)
        qa_embeddings.append(embedding)

    qa_embeddings = np.array(qa_embeddings) if qa_embeddings else np.array([])

    if len(qa_embeddings) == 0:
        return {
            "coverage_rate": 0.0,
            "covered_chunks": 0,
            "total_chunks": len(chunks),
            "uncovered_chunks": chunks,
            "multi_threshold": {},
            "chunk_analysis": {}
        }

    # カバレージ行列計算
    logger.info("カバレージ行列計算中...")
    coverage_matrix = np.zeros((len(chunks), len(qa_pairs)))
    for i in range(len(doc_embeddings)):
        for j in range(len(qa_embeddings)):
            similarity = analyzer.cosine_similarity(doc_embeddings[i], qa_embeddings[j])
            coverage_matrix[i, j] = similarity

    # データセット別最適閾値を取得
    thresholds = get_optimal_thresholds(dataset_type)
    standard_threshold = thresholds["standard"]

    # 基本カバレージ（標準閾値）
    max_similarities = coverage_matrix.max(axis=1)
    covered_count = sum(1 for s in max_similarities if s >= standard_threshold)
    coverage_rate = covered_count / len(chunks) if chunks else 0

    # 未カバーチャンクの特定
    uncovered_chunks = []
    for i, (chunk, sim) in enumerate(zip(chunks, max_similarities)):
        if sim < standard_threshold:
            uncovered_chunks.append({
                'chunk': chunk,
                'similarity': float(sim),
                'gap': float(standard_threshold - sim)
            })

    # 提案1の機能: 多段階カバレージ分析
    logger.info("多段階カバレージ分析実行中...")
    multi_threshold_results = multi_threshold_coverage(coverage_matrix, chunks, qa_pairs, thresholds)

    # 提案1の機能: チャンク特性別分析
    logger.info("チャンク特性別分析実行中...")
    chunk_characteristics = analyze_chunk_characteristics_coverage(
        chunks, coverage_matrix, qa_pairs, standard_threshold
    )

    # 結果を統合
    results = {
        # 基本メトリクス
        "coverage_rate": coverage_rate,
        "covered_chunks": covered_count,
        "total_chunks": len(chunks),
        "uncovered_chunks": uncovered_chunks,
        "max_similarities": max_similarities.tolist(),
        "threshold": standard_threshold,

        # 提案1: 多段階カバレージ
        "multi_threshold": multi_threshold_results,

        # 提案1: チャンク特性別分析
        "chunk_analysis": chunk_characteristics,

        # データセット情報
        "dataset_type": dataset_type,
        "optimal_thresholds": thresholds
    }

    # 分析結果のサマリーをログ出力
    logger.info(f"""
    多段階カバレージ分析結果:
    - Strict  (閾値{thresholds['strict']:.2f}): {multi_threshold_results['strict']['coverage_rate']:.1%}
    - Standard(閾値{thresholds['standard']:.2f}): {multi_threshold_results['standard']['coverage_rate']:.1%}
    - Lenient (閾値{thresholds['lenient']:.2f}): {multi_threshold_results['lenient']['coverage_rate']:.1%}

    チャンク特性別カバレージ:
    長さ別:
    - Short チャンク: {chunk_characteristics['by_length'].get('short', {}).get('coverage_rate', 0):.1%}
    - Medium チャンク: {chunk_characteristics['by_length'].get('medium', {}).get('coverage_rate', 0):.1%}
    - Long チャンク: {chunk_characteristics['by_length'].get('long', {}).get('coverage_rate', 0):.1%}

    位置別:
    - Beginning (前半): {chunk_characteristics['by_position'].get('beginning', {}).get('coverage_rate', 0):.1%}
    - Middle (中盤): {chunk_characteristics['by_position'].get('middle', {}).get('coverage_rate', 0):.1%}
    - End (後半): {chunk_characteristics['by_position'].get('end', {}).get('coverage_rate', 0):.1%}
    """)

    # インサイトがある場合は表示
    if chunk_characteristics['summary']['insights']:
        logger.info("\n📊 分析インサイト:")
        for insight in chunk_characteristics['summary']['insights']:
            logger.info(f"  • {insight}")

    return results


# ==========================================
# 結果保存
# ==========================================

def save_results(
    qa_pairs: List[Dict],
    coverage_results: Dict,
    dataset_type: str,
    output_dir: str = "qa_output/a02"
) -> Dict[str, str]:
    """結果をファイルに保存
    Args:
        qa_pairs: Q/Aペアリスト
        coverage_results: カバレージ分析結果
        dataset_type: データセットタイプ
        output_dir: 出力ディレクトリ
    Returns:
        保存したファイルパス
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Q/Aペアを保存（JSON）
    qa_file = output_path / f"qa_pairs_{dataset_type}_{timestamp}.json"
    with open(qa_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    # Q/Aペアを保存（CSV - 全カラム）
    qa_csv_file = output_path / f"qa_pairs_{dataset_type}_{timestamp}.csv"
    qa_df = pd.DataFrame(qa_pairs)
    qa_df.to_csv(qa_csv_file, index=False, encoding='utf-8')

    # Q/Aペアを保存（CSV - question/answerのみの統一フォーマット）
    qa_simple_file = Path("qa_output") / f"a02_qa_pairs_{dataset_type}.csv"
    qa_simple_file.parent.mkdir(parents=True, exist_ok=True)
    if 'question' in qa_df.columns and 'answer' in qa_df.columns:
        qa_simple_df = qa_df[['question', 'answer']]
        qa_simple_df.to_csv(qa_simple_file, index=False, encoding='utf-8')
        logger.info(f"統一フォーマットCSV保存: {qa_simple_file} ({len(qa_simple_df)}件)")

    # カバレージ分析結果を保存
    coverage_file = output_path / f"coverage_{dataset_type}_{timestamp}.json"
    # uncovered_chunksのシリアライズ対策
    coverage_save = coverage_results.copy()
    coverage_save['uncovered_chunks'] = [
        {
            'chunk_id': uc['chunk'].get('id', ''),
            'similarity': uc['similarity'],
            'gap': uc['gap'],
            'text_preview': uc['chunk']['text'][:200] + '...'
        }
        for uc in coverage_save.get('uncovered_chunks', [])
    ]

    with open(coverage_file, 'w', encoding='utf-8') as f:
        json.dump(coverage_save, f, ensure_ascii=False, indent=2)

    # サマリー情報を保存
    summary = {
        "dataset_type": dataset_type,
        "dataset_name": DATASET_CONFIGS[dataset_type]["name"],
        "generated_at": timestamp,
        "total_qa_pairs": len(qa_pairs),
        "coverage_rate": coverage_results['coverage_rate'],
        "covered_chunks": coverage_results['covered_chunks'],
        "total_chunks": coverage_results['total_chunks'],
        "files": {
            "qa_json": str(qa_file),
            "qa_csv": str(qa_csv_file),
            "coverage": str(coverage_file)
        }
    }

    summary_file = output_path / f"summary_{dataset_type}_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"結果を保存しました: {output_path}")

    return {
        "qa_json": str(qa_file),
        "qa_csv": str(qa_csv_file),
        "coverage": str(coverage_file),
        "summary": str(summary_file)
    }


# ==========================================
# メイン処理
# ==========================================

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="preprocessedファイルからQ/Aペアを生成"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        default="cc_news",
        help="処理するデータセット"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="使用するOpenAIモデル"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="qa_output/a02",
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="処理する最大文書数（テスト用）"
    )
    parser.add_argument(
        "--analyze-coverage",
        action="store_true",
        help="カバレージ分析を実行"
    )
    parser.add_argument(
        "--batch-chunks",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="1回のAPIで処理するチャンク数（デフォルト: 3）"
    )
    parser.add_argument(
        "--merge-chunks",
        action="store_true",
        default=True,
        help="小さいチャンクを統合する（デフォルト: 有効）"
    )
    parser.add_argument(
        "--no-merge-chunks",
        dest="merge_chunks",
        action="store_false",
        help="チャンク統合を無効化"
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=150,
        help="統合対象の最小トークン数（デフォルト: 150）"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=400,
        help="統合後の最大トークン数（デフォルト: 400）"
    )

    args = parser.parse_args()

    # APIキー確認
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key-here":
        logger.error("OPENAI_API_KEYが設定されていません")
        sys.exit(1)

    logger.info(f"""
    =====================================
    Q/Aペア生成開始
    =====================================
    データセット: {DATASET_CONFIGS[args.dataset]['name']}
    モデル: {args.model}
    出力先: {args.output}
    最大文書数: {args.max_docs if args.max_docs else '制限なし'}
    カバレージ分析: {'実行' if args.analyze_coverage else 'スキップ'}
    """)

    try:
        # 1. データ読み込み
        logger.info("\n[1/4] データ読み込み...")
        df = load_preprocessed_data(args.dataset)

        # 2. チャンク作成
        logger.info("\n[2/4] チャンク作成...")
        chunks = create_document_chunks(df, args.dataset, args.max_docs)

        if not chunks:
            logger.error("チャンクが作成されませんでした")
            sys.exit(1)

        # 3. Q/Aペア生成
        logger.info("\n[3/4] Q/Aペア生成...")
        logger.info(f"オプション: バッチサイズ={args.batch_chunks}, チャンク統合={'有効' if args.merge_chunks else '無効'}")
        qa_pairs = generate_qa_for_dataset(
            chunks,
            args.dataset,
            args.model,
            chunk_batch_size=args.batch_chunks,
            merge_chunks=args.merge_chunks,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens
        )

        if not qa_pairs:
            logger.warning("Q/Aペアが生成されませんでした")

        # 4. カバレージ分析（オプション）
        coverage_results = {}
        if args.analyze_coverage and qa_pairs:
            logger.info("\n[4/4] カバレージ分析...")
            coverage_results = analyze_coverage(chunks, qa_pairs, args.dataset)

            logger.info(f"""
            カバレージ分析結果:
            - カバレージ率: {coverage_results['coverage_rate']:.1%}
            - カバー済みチャンク: {coverage_results['covered_chunks']}/{coverage_results['total_chunks']}
            - 未カバーチャンク: {len(coverage_results['uncovered_chunks'])}
            """)
        else:
            logger.info("\n[4/4] カバレージ分析をスキップ")
            coverage_results = {
                "coverage_rate": 0,
                "covered_chunks": 0,
                "total_chunks": len(chunks),
                "uncovered_chunks": []
            }

        # 5. 結果保存
        logger.info("\n結果を保存中...")
        saved_files = save_results(qa_pairs, coverage_results, args.dataset, args.output)

        # 完了メッセージ
        logger.info(f"""
        =====================================
        処理完了
        =====================================
        生成Q/Aペア数: {len(qa_pairs)}
        保存ファイル:
        - Q/A (JSON): {saved_files['qa_json']}
        - Q/A (CSV): {saved_files['qa_csv']}
        - カバレージ: {saved_files['coverage']}
        - サマリー: {saved_files['summary']}
        """)

        # 統計情報表示
        if qa_pairs:
            question_types = {}
            for qa in qa_pairs:
                qt = qa.get('question_type', 'unknown')
                question_types[qt] = question_types.get(qt, 0) + 1

            print("\n質問タイプ別統計:")
            for qt, count in sorted(question_types.items()):
                print(f"  {qt}: {count}件")

    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
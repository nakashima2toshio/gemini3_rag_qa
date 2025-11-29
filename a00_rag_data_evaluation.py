#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
起動: streamlit run a00_rag_data_evaluation.py --server.port=8501

Q&A正解率検証ツール
QAペアとQdrantコレクションの回答を比較し、コサイン類似度による正解率を評価
"""

import streamlit as st
import pandas as pd
from qdrant_client import QdrantClient
from openai import OpenAI
import os
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import time

# Streamlitページ設定
st.set_page_config(
    page_title="Q&A正解率検証ツール",
    page_icon="✅",
    layout="wide"
)

# 定数
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Q/Aペアとコレクションの対応
QA_DATASETS = {
    "CC News (4,646件)": {
        "path": "qa_output/a02_qa_pairs_cc_news.csv",
        "collections": [
            ("raw_cc_news", "Raw データ (660件)"),
            ("qa_cc_news_a02_llm", "LLMベース (9,290件)"),
            ("qa_cc_news_a03_rule", "ルールベース (8,556件)"),
            ("qa_cc_news_a10_hybrid", "ハイブリッド (4,334件)")
        ]
    },
    "Livedoor (965件)": {
        "path": "qa_output/a02_qa_pairs_livedoor.csv",
        "collections": [
            ("raw_livedoor", "Raw データ (20,193件)"),
            ("qa_livedoor_a02_20_llm", "LLMベース (964件)"),
            ("qa_livedoor_a03_rule", "ルールベース (466件)"),
            ("qa_livedoor_a10_hybrid", "ハイブリッド (2,175件)")
        ]
    }
}

# OpenAI/Qdrantクライアント初期化
@st.cache_resource
def init_clients():
    """APIクライアントの初期化"""
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    qdrant_client = QdrantClient(url=QDRANT_URL)
    return openai_client, qdrant_client

@st.cache_data
def load_qa_data(file_path: str):
    """Q&Aペアデータの読み込み"""
    try:
        df = pd.read_csv(file_path)
        # データのバリデーション
        if 'question' not in df.columns or 'answer' not in df.columns:
            st.error("CSVファイルに'question'と'answer'カラムが必要です")
            return pd.DataFrame()
        return df
    except FileNotFoundError:
        st.error(f"ファイルが見つかりません: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return pd.DataFrame()

def get_embedding(text: str, client: OpenAI) -> List[float]:
    """テキストのembeddingベクトルを取得"""
    try:
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding取得エラー: {e}")
        return []

def search_qdrant(query: str, collection_name: str, openai_client: OpenAI, qdrant_client: QdrantClient, top_k: int = 1) -> Tuple[str, float]:
    """Qdrantコレクションから検索し、最高スコアの結果を返す"""
    try:
        # クエリのembedding取得
        query_vector = get_embedding(query, openai_client)
        if not query_vector:
            return "", 0.0

        # Qdrant検索
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k
        ).points

        if results:
            # 最高スコアの結果を取得
            top_result = results[0]
            answer = top_result.payload.get('answer', '') if top_result.payload else ''
            score = top_result.score
            return answer, score

        return "", 0.0
    except Exception as e:
        st.error(f"Qdrant検索エラー ({collection_name}): {e}")
        return "", 0.0

def calculate_cosine_similarity(text1: str, text2: str, openai_client: OpenAI) -> float:
    """2つのテキスト間のコサイン類似度を計算"""
    try:
        # 両方のテキストのembeddingを取得
        vec1 = get_embedding(text1, openai_client)
        vec2 = get_embedding(text2, openai_client)

        if not vec1 or not vec2:
            return 0.0

        # コサイン類似度計算
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        return float(similarity)
    except Exception as e:
        st.error(f"類似度計算エラー: {e}")
        return 0.0

def process_qa_pairs(df: pd.DataFrame, collection_name: str, openai_client: OpenAI, qdrant_client: QdrantClient, progress_bar=None) -> pd.DataFrame:
    """Q&Aペアを処理し、類似度を計算"""
    results = []
    total = len(df)

    for idx, row in df.iterrows():
        if progress_bar:
            progress_bar.progress((idx + 1) / total, text=f"処理中: {idx + 1}/{total}")

        qa_question = row['question']
        qa_answer = row['answer']

        # Qdrantから回答取得
        ans_output, search_score = search_qdrant(qa_question, collection_name, openai_client, qdrant_client)

        # コサイン類似度計算
        if ans_output:
            similarity = calculate_cosine_similarity(qa_answer, ans_output, openai_client)
        else:
            similarity = 0.0

        results.append({
            'question': qa_question,
            'correct_answer': qa_answer,  # CSVからの正解答
            'retrieved_answer': ans_output,  # Qdrantからの検索結果
            'search_score': search_score,
            'similarity': similarity * 100  # パーセント表示
        })

    return pd.DataFrame(results)

def create_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """統計サマリーテーブルの作成"""
    if df.empty or 'similarity' not in df:
        return pd.DataFrame()

    total = len(df)

    # スコアレンジごとの集計
    ranges = [
        ('80%以上', df[df['similarity'] >= 80].shape[0]),
        ('60%以上', df[(df['similarity'] >= 60) & (df['similarity'] < 80)].shape[0]),
        ('40%以上', df[(df['similarity'] >= 40) & (df['similarity'] < 60)].shape[0]),
        ('39%以下', df[df['similarity'] < 40].shape[0])
    ]

    summary_data = []
    for label, count in ranges:
        percentage = (count / total * 100) if total > 0 else 0
        summary_data.append({
            'スコア範囲': label,
            '件数': str(count),  # 文字列に統一
            '構成比(%)': f"{percentage:.1f}%"
        })

    # 平均値を追加
    avg_similarity = df['similarity'].mean()
    summary_data.append({
        'スコア範囲': '平均スコア',
        '件数': '-',
        '構成比(%)': f"{avg_similarity:.1f}%"
    })

    return pd.DataFrame(summary_data)

def main():
    """メインアプリケーション"""
    st.title("🎯 Q&A正解率検証ツール")
    st.markdown("QAペアとQdrantコレクションの回答を比較し、コサイン類似度による正解率を評価します")

    # クライアント初期化
    openai_client, qdrant_client = init_clients()

    # サイドバー：設定
    with st.sidebar:
        st.header("⚙️ 設定")

        # Q/Aデータセット選択
        st.subheader("Q/Aデータセット")
        selected_dataset = st.selectbox(
            "検証元データセット",
            options=list(QA_DATASETS.keys())
        )

        # 選択したデータセットに対応するコレクション表示
        dataset_config = QA_DATASETS[selected_dataset]
        st.subheader("検証対象コレクション")

        collection_options = {
            name: f"{display} - {name}"
            for name, display in dataset_config["collections"]
        }
        selected_collection = st.selectbox(
            "コレクション選択",
            options=list(collection_options.keys()),
            format_func=lambda x: collection_options[x]
        )

        # 処理件数制限
        st.subheader("処理オプション")
        limit_processing = st.checkbox("処理件数を制限", value=True)
        if limit_processing:
            max_rows = st.number_input("最大処理件数", min_value=1, max_value=5000, value=100, step=10)
        else:
            max_rows = None

        # 実行ボタン
        run_analysis = st.button("🚀 検証開始", type="primary", use_container_width=True)

    # メインコンテンツ
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 データ情報")
        # 選択したデータセットのパスからQ&Aデータ読み込み
        qa_file_path = dataset_config["path"]
        qa_df = load_qa_data(qa_file_path)
        if not qa_df.empty:
            st.metric("データセット", selected_dataset)
            st.metric("Q&Aペア総数", f"{len(qa_df):,}件")
            st.metric("選択コレクション", selected_collection)
            if limit_processing and max_rows:
                st.info(f"処理対象: 最初の{max_rows}件")

    with col2:
        st.subheader("📈 検証結果サマリー")
        result_container = st.container()

    # 検証実行
    if run_analysis and not qa_df.empty:
        with st.spinner(f"🔄 {selected_collection}の検証を実行中..."):
            # 処理対象データの準備
            process_df = qa_df.head(max_rows) if max_rows else qa_df

            # プログレスバー
            progress_bar = st.progress(0, text="初期化中...")

            # Q&Aペア処理
            results_df = process_qa_pairs(
                process_df,
                selected_collection,
                openai_client,
                qdrant_client,
                progress_bar
            )

            progress_bar.empty()

            if not results_df.empty:
                # サマリー統計
                with result_container:
                    summary_stats = create_summary_stats(results_df)
                    st.dataframe(summary_stats, use_container_width=True, hide_index=True)

                # 詳細結果
                st.subheader("🔍 詳細結果")

                # フィルタリング
                col1, col2, col3 = st.columns(3)
                with col1:
                    min_score = st.slider("最小類似度(%)", 0, 100, 0)
                with col2:
                    max_score = st.slider("最大類似度(%)", 0, 100, 100)
                with col3:
                    sort_order = st.selectbox("並び順", ["類似度降順", "類似度昇順"])

                # フィルタ適用
                filtered_df = results_df[
                    (results_df['similarity'] >= min_score) &
                    (results_df['similarity'] <= max_score)
                ]

                # ソート
                ascending = sort_order == "類似度昇順"
                filtered_df = filtered_df.sort_values('similarity', ascending=ascending)

                # 表示
                display_df = filtered_df[['question', 'correct_answer', 'retrieved_answer', 'similarity', 'search_score']].copy()
                display_df = display_df.rename(columns={
                    'question': '質問',
                    'correct_answer': '正解答',
                    'retrieved_answer': '検索回答',
                    'similarity': '類似度(%)',
                    'search_score': '検索スコア'
                })
                st.dataframe(
                    display_df.round(2),
                    use_container_width=True,
                    height=400
                )

                # CSV出力
                csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 結果をCSVダウンロード",
                    data=csv,
                    file_name=f"qa_verification_{selected_collection}_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

                # 統計情報
                with st.expander("📊 統計詳細"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("平均類似度", f"{results_df['similarity'].mean():.1f}%")
                    with col2:
                        st.metric("中央値", f"{results_df['similarity'].median():.1f}%")
                    with col3:
                        st.metric("標準偏差", f"{results_df['similarity'].std():.1f}%")
                    with col4:
                        st.metric("最高類似度", f"{results_df['similarity'].max():.1f}%")

                st.success("✅ 検証完了！")
            else:
                st.error("検証処理中にエラーが発生しました")

if __name__ == "__main__":
    main()
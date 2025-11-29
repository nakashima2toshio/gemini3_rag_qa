#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qdrant_show_page.py - Qdrantデータ表示ページ
============================================
Qdrantコレクションのデータ表示機能

機能:
- コレクション一覧表示
- データソース情報表示
- ポイントデータ取得・表示
- ヘルスチェック
"""

import time
from datetime import datetime

import pandas as pd
import streamlit as st
from qdrant_client import QdrantClient

# サービスモジュールからインポート
from services.qdrant_service import (
    QdrantHealthChecker,
    QdrantDataFetcher,
    QDRANT_CONFIG,
)


def display_source_info(source_info: dict) -> None:
    """データソース情報を表示"""
    if "error" in source_info:
        st.error(f"ソース情報取得エラー: {source_info['error']}")
        return

    total_points = source_info.get("total_points", 0)
    sources = source_info.get("sources", {})
    sample_size = source_info.get("sample_size", 0)

    if not sources:
        st.info("📂 データソース情報が見つかりません")
        return

    # メトリクス表示
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("総ポイント数", f"{total_points:,}")
    with col2:
        st.metric("ソース数", f"{len(sources)}")
    with col3:
        st.metric("サンプルサイズ", f"{sample_size}")

    # ソース情報テーブル
    source_data = []
    for source, stats in sorted(sources.items()):
        source_data.append({
            "ソース": source,
            "推定数": stats["estimated_total"],
            "割合": f"{stats['percentage']:.1f}%",
            "生成方法": stats.get("method", "unknown"),
            "ドメイン": stats.get("domain", "unknown"),
        })

    df_sources = pd.DataFrame(source_data)
    st.dataframe(df_sources, use_container_width=True, hide_index=True)


def show_qdrant_page():
    """画面4: Qdrant Show - コレクション表示"""
    st.title("🔍 Show-Qdrantコレクション")
    st.caption("Qdrant Vector Database の状態監視とデータ表示")

    # セッションステート初期化
    if "qdrant_debug_mode" not in st.session_state:
        st.session_state.qdrant_debug_mode = False
    if "qdrant_auto_refresh" not in st.session_state:
        st.session_state.qdrant_auto_refresh = False
    if "qdrant_refresh_interval" not in st.session_state:
        st.session_state.qdrant_refresh_interval = 30

    # サイドバー（左ペイン）
    with st.sidebar:
        st.header("⚙️ Qdrant接続状態")

        # デバッグモード切り替え
        debug_mode = st.checkbox(
            "🐛 デバッグモード", value=st.session_state.qdrant_debug_mode
        )
        st.session_state.qdrant_debug_mode = debug_mode

        # 自動リフレッシュ設定
        col1, col2 = st.columns(2)
        with col1:
            auto_refresh = st.checkbox(
                "🔄 自動更新", value=st.session_state.qdrant_auto_refresh
            )
            st.session_state.qdrant_auto_refresh = auto_refresh
        with col2:
            if auto_refresh:
                refresh_interval = st.number_input(
                    "間隔(秒)", min_value=5, max_value=300, value=30
                )
                st.session_state.qdrant_refresh_interval = refresh_interval

        # 接続チェック実行ボタン
        check_button = st.button(
            "🔍 接続チェック実行", type="primary", use_container_width=True
        )

        # HealthCheckerインスタンス
        checker = QdrantHealthChecker(debug_mode=debug_mode)

        # 接続状態表示エリア
        status_container = st.container()

        # 自動リフレッシュまたはボタン押下時に実行
        refresh_interval = st.session_state.qdrant_refresh_interval
        if check_button or (auto_refresh and time.time() % refresh_interval < 1):
            with status_container:
                with st.spinner("チェック中..."):
                    is_connected, message, metrics = checker.check_qdrant()

                # Qdrantの状態表示
                if is_connected:
                    st.success(f"{QDRANT_CONFIG['icon']} **{QDRANT_CONFIG['name']}**")
                    st.caption(f"✅ {message}")

                    # メトリクス表示
                    if metrics and debug_mode:
                        with st.expander("詳細情報", expanded=False):
                            for key, value in metrics.items():
                                st.text(f"{key}: {value}")
                else:
                    st.error(f"{QDRANT_CONFIG['icon']} **{QDRANT_CONFIG['name']}**")
                    st.caption(f"❌ {message}")

                    # エラー詳細（デバッグモード）
                    if debug_mode:
                        with st.expander("エラー詳細", expanded=False):
                            st.code(message)
                            st.caption(
                                f"Host: {QDRANT_CONFIG.get('host')}:{QDRANT_CONFIG.get('port')}"
                            )

                            # Docker起動コマンド表示
                            st.info("Docker起動コマンド:")
                            cmd = f"docker run -d -p {QDRANT_CONFIG['port']}:{QDRANT_CONFIG['port']} {QDRANT_CONFIG['docker_image']}"
                            st.code(cmd, language="bash")

    # メインエリア（右ペイン）
    st.header("📊 Qdrant データ表示")

    try:
        # Qdrantクライアントを作成
        client = QdrantClient(url=QDRANT_CONFIG["url"], timeout=5)
        data_fetcher = QdrantDataFetcher(client)

        # コレクション概要表示
        st.subheader("📚 コレクション一覧")

        # コレクション一覧を取得
        df_collections = data_fetcher.fetch_collections()

        if not df_collections.empty and "Collection" in df_collections.columns:
            st.dataframe(df_collections, use_container_width=True)

            # コレクション名のリストを作成
            collection_names = df_collections["Collection"].tolist()

            # ===== データソース情報の表示（メインエリア先頭） =====
            st.divider()
            st.subheader("📂 コレクションのデータソース情報")
            st.caption(
                "各コレクションがqa_output/ディレクトリーのどのファイルから構成されているかを表示します"
            )

            # 各コレクションのソース情報を表示
            for collection_name in collection_names:
                with st.expander(
                    f"📦 {collection_name}", expanded=(collection_name == "qa_corpus")
                ):
                    with st.spinner(f"{collection_name} のソース情報を取得中..."):
                        source_info = data_fetcher.fetch_collection_source_info(
                            collection_name
                        )
                        display_source_info(source_info)

            # エクスポート機能
            col1, col2 = st.columns(2)
            with col1:
                csv = df_collections.to_csv(index=False)
                st.download_button(
                    label="📥 CSVダウンロード",
                    data=csv,
                    file_name=f"qdrant_collections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            with col2:
                json_str = df_collections.to_json(orient="records", indent=2)
                st.download_button(
                    label="📥 JSONダウンロード",
                    data=json_str,
                    file_name=f"qdrant_collections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )

            # コレクション詳細表示
            st.divider()
            st.subheader("🔍 コレクション詳細データ")

            if collection_names:
                selected_collection = st.selectbox(
                    "詳細を表示するコレクションを選択",
                    options=collection_names,
                    key="selected_collection",
                )

                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    limit = st.number_input(
                        "表示件数",
                        min_value=1,
                        max_value=500,
                        value=50,
                        key="qdrant_limit",
                    )
                with col2:
                    show_details = st.button(
                        "📊 詳細情報を表示", key="show_collection_details"
                    )
                with col3:
                    fetch_points = st.button(
                        "🔍 ポイントデータを取得", key="fetch_collection_points"
                    )

                # コレクション詳細情報の表示
                if show_details:
                    with st.spinner(f"{selected_collection} の詳細情報を取得中..."):
                        info = data_fetcher.fetch_collection_info(selected_collection)

                        if "error" not in info:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("ベクトル数", info["vectors_count"])
                            with col2:
                                st.metric("ポイント数", info["points_count"])
                            with col3:
                                st.metric("インデックス済み", info["indexed_vectors"])
                            with col4:
                                st.metric("ステータス", info["status"])

                            # 設定情報
                            st.write("**ベクトル設定:**")
                            st.write(
                                f"  • ベクトル次元: {info['config']['vector_size']}"
                            )
                            st.write(f"  • 距離計算: {info['config']['distance']}")
                        else:
                            st.error(f"エラー: {info['error']}")

                # ポイントデータの表示
                if fetch_points:
                    with st.spinner(
                        f"{selected_collection} のポイントデータを取得中..."
                    ):
                        df_points = data_fetcher.fetch_collection_points(
                            selected_collection, limit
                        )

                        if not df_points.empty and "ID" in df_points.columns:
                            st.write(
                                f"**{selected_collection} のデータサンプル ({len(df_points)} 件):**"
                            )
                            st.dataframe(df_points, use_container_width=True)

                            # エクスポート機能
                            col1, col2 = st.columns(2)
                            with col1:
                                csv = df_points.to_csv(index=False)
                                st.download_button(
                                    label="📥 ポイントデータ CSVダウンロード",
                                    data=csv,
                                    file_name=f"{selected_collection}_points_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                )
                            with col2:
                                json_str = df_points.to_json(orient="records", indent=2)
                                st.download_button(
                                    label="📥 ポイントデータ JSONダウンロード",
                                    data=json_str,
                                    file_name=f"{selected_collection}_points_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                )
                        elif "Info" in df_points.columns:
                            st.info(df_points.iloc[0]["Info"])
                        elif "Error" in df_points.columns:
                            st.error(f"エラー: {df_points.iloc[0]['Error']}")
                        else:
                            st.info("ポイントデータが見つかりません")

        elif "Info" in df_collections.columns:
            st.info(df_collections.iloc[0]["Info"])
        elif "Error" in df_collections.columns:
            error_msg = df_collections.iloc[0]["Error"]
            _show_connection_error(error_msg, debug_mode)
        else:
            st.info("コレクションが見つかりません")

    except Exception as e:
        error_msg = str(e)
        _show_connection_error(error_msg, debug_mode)

    # フッター
    st.divider()
    st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # デバッグ情報表示
    if debug_mode:
        with st.expander("🐛 デバッグ情報", expanded=False):
            st.subheader("サーバー設定")
            st.json(QDRANT_CONFIG)


def _show_connection_error(error_msg: str, debug_mode: bool) -> None:
    """接続エラーを表示"""
    if "Connection refused" in error_msg or "[Errno 61]" in error_msg:
        st.error("❌ Qdrantサーバーに接続できません")
        st.warning("Qdrantサーバーが起動していることを確認してください")
        st.code("python server.py", language="bash")
        st.caption("または")
        st.code("docker run -p 6333:6333 qdrant/qdrant", language="bash")
        if debug_mode:
            with st.expander("🔍 詳細エラー情報", expanded=False):
                st.error(f"詳細エラー: {error_msg}")
    elif "timeout" in error_msg.lower():
        st.error("⏱️ Qdrantサーバーへの接続がタイムアウトしました")
        st.warning("サーバーが応答していないか、ネットワークの問題があります")
    else:
        st.error(f"Qdrant接続エラー: {error_msg}")
        st.info("Qdrantサーバーが正しく起動していることを確認してください")
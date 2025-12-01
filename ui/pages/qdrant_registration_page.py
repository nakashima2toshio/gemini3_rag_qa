#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qdrant_registration_page.py - Qdrant登録ページ
==============================================
Q/AデータのQdrantへの登録機能

機能:
- CSVファイルからQdrantへの登録
- コレクション管理（作成・削除）
- 埋め込みベクトル生成
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from qdrant_client import QdrantClient

# サービスモジュールからインポート
from services.qdrant_service import (
    get_collection_stats,
    get_all_collections,
    delete_all_collections,
    load_csv_for_qdrant,
    build_inputs_for_embedding,
    embed_texts_for_qdrant,
    create_or_recreate_collection_for_qdrant,
    build_points_for_qdrant,
    upsert_points_to_qdrant,
    merge_collections,
)

logger = logging.getLogger(__name__)


def show_qdrant_registration_page():
    """画面3: Q/AペアデータQdrant登録"""
    st.title("🗄️ Q/Aペアデータ・Qdrant登録")
    st.caption("qa_output/*.csvのデータをQdrantベクトルDBに登録")

    # サイドバー：設定
    with st.sidebar:
        st.header("⚙️ Qdrant設定")

        qdrant_url = st.text_input(
            "Qdrant URL", value="http://localhost:6333", help="QdrantサーバーのURL"
        )

        st.divider()
        st.header("📋 操作モード")

        operation_mode = st.radio(
            "操作モードを選択",
            options=["all_collections", "individual_csv", "collection_merge"],
            format_func=lambda x: {
                "all_collections": "📊 全コレクション操作",
                "individual_csv": "📄 個別CSV操作",
                "collection_merge": "🔗 コレクション統合",
            }[x],
            key="qdrant_operation_mode",
        )

        st.divider()

        # モード別設定
        if operation_mode == "individual_csv":
            st.subheader("📄 CSV設定")

            # qa_output/*.csvファイル一覧取得
            qa_output_dir = Path("qa_output")
            if qa_output_dir.exists():
                csv_files = sorted(qa_output_dir.glob("*.csv"))
                csv_options = [f.name for f in csv_files]
            else:
                csv_options = []

            if csv_options:
                selected_csv = st.selectbox(
                    "ファイル選択",
                    options=csv_options,
                    help="登録するCSVファイルを選択",
                )

                # コレクション名を自動生成（カスタマイズ可能）
                default_collection = f"qa_{Path(selected_csv).stem}"
                collection_name = st.text_input(
                    "コレクション名",
                    value=default_collection,
                    help="Qdrantコレクション名",
                )

                recreate_collection = st.checkbox(
                    "既存データ削除",
                    value=True,
                    help="既存コレクションを削除して再作成",
                )

                include_answer = st.checkbox(
                    "answerを含める", value=True, help="埋め込み生成時にanswerも含める"
                )

                data_limit = st.number_input(
                    "データ件数制限",
                    min_value=0,
                    max_value=100000,
                    value=0,
                    step=100,
                    help="0=無制限",
                )
            else:
                st.warning("qa_output/フォルダにCSVファイルが見つかりません")
                selected_csv = None
                collection_name = None
                recreate_collection = False
                include_answer = False
                data_limit = 0

    # Qdrant接続確認
    st.subheader("📡 Qdrant接続状態")

    try:
        client = QdrantClient(url=qdrant_url, timeout=30)
        client.get_collections()
        st.success(f"✅ Qdrant接続成功: {qdrant_url}")
        qdrant_connected = True
    except Exception as e:
        st.error(f"❌ Qdrant接続エラー: {e}")
        st.warning("Qdrantが起動していることを確認してください。")
        st.code("docker run -p 6333:6333 qdrant/qdrant", language="bash")
        qdrant_connected = False
        client = None

    st.divider()

    # モード別メインコンテンツ
    if operation_mode == "all_collections":
        # ===================================================================
        # 全コレクション操作モード
        # ===================================================================
        st.subheader("📊 全コレクション一覧")

        if qdrant_connected and client:
            try:
                collections = get_all_collections(client)

                if collections:
                    total_points = sum(c["points_count"] for c in collections)

                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.metric("総コレクション数", f"{len(collections)} 個")
                    with col_metric2:
                        st.metric("総ポイント数", f"{total_points:,} 件")

                    # コレクション一覧表
                    df_collections = pd.DataFrame(collections)
                    df_collections = df_collections.sort_values(
                        "points_count", ascending=False
                    )

                    st.dataframe(
                        df_collections,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "name": st.column_config.TextColumn(
                                "コレクション名", width="medium"
                            ),
                            "points_count": st.column_config.NumberColumn(
                                "ポイント数", format="%d"
                            ),
                            "status": st.column_config.TextColumn(
                                "ステータス", width="small"
                            ),
                        },
                    )

                    st.divider()

                    # 危険な操作セクション
                    st.subheader("⚠️ 危険な操作")

                    col_btn1, col_btn2 = st.columns(2)

                    with col_btn1:
                        if st.button(
                            "🗑️ 全コレクション削除",
                            type="secondary",
                            use_container_width=True,
                        ):
                            st.session_state["confirm_delete_all"] = True

                    with col_btn2:
                        if st.button(
                            "📊 詳細統計表示", type="primary", use_container_width=True
                        ):
                            st.session_state["show_detailed_stats"] = True

                    # 削除確認ダイアログ
                    if st.session_state.get("confirm_delete_all", False):
                        st.warning("⚠️ **警告：全コレクション削除**")
                        st.error(
                            f"**{len(collections)}個**のコレクション（合計**{total_points:,}ポイント**）が完全に削除されます。"
                        )
                        st.error("この操作は取り消せません！")

                        col_confirm1, col_confirm2 = st.columns(2)

                        with col_confirm1:
                            if st.button(
                                "✅ 削除を実行",
                                type="primary",
                                use_container_width=True,
                            ):
                                with st.spinner("削除中..."):
                                    deleted = delete_all_collections(client)
                                    st.success(
                                        f"✅ {deleted}個のコレクションを削除しました"
                                    )
                                    st.session_state["confirm_delete_all"] = False
                                    st.rerun()

                        with col_confirm2:
                            if st.button("❌ キャンセル", use_container_width=True):
                                st.session_state["confirm_delete_all"] = False
                                st.rerun()

                    # 詳細統計表示
                    if st.session_state.get("show_detailed_stats", False):
                        st.divider()
                        st.subheader("📊 詳細統計情報")

                        for col_info in collections:
                            with st.expander(
                                f"📦 {col_info['name']} ({col_info['points_count']:,} ポイント)"
                            ):
                                try:
                                    stats = get_collection_stats(
                                        client, col_info["name"]
                                    )
                                    if stats:
                                        st.json(stats)
                                    else:
                                        st.warning("統計情報を取得できませんでした")
                                except Exception as e:
                                    st.error(f"エラー: {e}")

                        if st.button("閉じる"):
                            st.session_state["show_detailed_stats"] = False
                            st.rerun()

                else:
                    st.info("コレクションが存在しません")

            except Exception as e:
                st.error(f"エラー: {e}")
                logger.error(f"コレクション一覧取得エラー: {e}")
        else:
            st.warning("Qdrantに接続できていません")

    elif operation_mode == "individual_csv":
        # ===================================================================
        # 個別CSV操作モード
        # ===================================================================
        st.subheader("📄 CSV登録設定")

        if not csv_options:
            st.warning("qa_output/フォルダにCSVファイルがありません")
            st.info("先に「Q/A生成」でデータを作成してください")
            return

        # ファイル情報表示
        csv_path = qa_output_dir / selected_csv
        file_size = csv_path.stat().st_size
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"""
**ファイル情報**
- ファイル名: {selected_csv}
- ファイルサイズ: {size_str}
- コレクション名: {collection_name}
            """)

        with col_info2:
            st.info(f"""
**登録設定**
- 既存データ削除: {"はい" if recreate_collection else "いいえ"}
- answerを含める: {"はい" if include_answer else "いいえ"}
- データ件数制限: {data_limit if data_limit > 0 else "無制限"}
            """)

        # データプレビュー
        with st.expander("📋 データプレビュー（最初の3件）"):
            try:
                df_preview = pd.read_csv(csv_path, nrows=3)
                st.dataframe(df_preview, use_container_width=True)
            except Exception as e:
                st.error(f"プレビュー読み込みエラー: {e}")

        st.divider()

        # 登録ボタン
        run_registration = st.button(
            "🚀 Qdrantに登録",
            type="primary",
            use_container_width=True,
            disabled=not qdrant_connected,
        )

        # ログ表示エリア
        st.subheader("📜 処理ログ")
        log_container = st.container()

        if "qdrant_logs" not in st.session_state:
            st.session_state["qdrant_logs"] = []

        def add_log(message: str):
            """ログを追加"""
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state["qdrant_logs"].append(f"[{timestamp}] {message}")

        # 登録処理実行
        if run_registration:
            st.session_state["qdrant_logs"] = []  # ログクリア
            add_log(f"🚀 登録処理開始: {selected_csv}")

            try:
                # ステップ1: CSVロード
                with st.spinner("📁 CSVファイル読み込み中..."):
                    add_log(f"📁 CSV読み込み: {csv_path}")
                    df = load_csv_for_qdrant(str(csv_path), limit=data_limit)
                    add_log(f"✅ {len(df)} 件のデータを読み込みました")

                # ステップ2: コレクション作成
                with st.spinner("🗄️ コレクション準備中..."):
                    add_log(f"🗄️ コレクション準備: {collection_name}")
                    create_or_recreate_collection_for_qdrant(
                        client, collection_name, recreate_collection
                    )
                    add_log("✅ コレクション準備完了")

                # ステップ3: 埋め込み生成
                with st.spinner("🔢 埋め込み生成中..."):
                    add_log("🔢 埋め込み生成開始")
                    texts = build_inputs_for_embedding(df, include_answer)
                    vectors = embed_texts_for_qdrant(
                        texts, model="gemini-embedding-001"
                    )
                    add_log(f"✅ {len(vectors)} 件の埋め込みを生成しました")

                # ステップ4: ポイント構築
                with st.spinner("📦 ポイント構築中..."):
                    add_log("📦 Qdrantポイント構築中")
                    # ドメイン名を推定
                    if "cc_news" in selected_csv.lower():
                        domain = "cc_news"
                    elif "livedoor" in selected_csv.lower():
                        domain = "livedoor"
                    else:
                        domain = "custom"

                    points = build_points_for_qdrant(df, vectors, domain, selected_csv)
                    add_log(f"✅ {len(points)} 個のポイントを構築しました")

                # ステップ5: Qdrantアップサート
                with st.spinner("⬆️ Qdrantアップサート中..."):
                    add_log("⬆️ Qdrantにアップサート中")
                    count = upsert_points_to_qdrant(client, collection_name, points)
                    add_log(f"✅ {count} 件をQdrantに登録しました")

                # 完了
                add_log("🎉 全処理完了！")
                st.success(f"✅ {count}件のデータをQdrantに登録しました")

                # 統計情報を表示
                try:
                    stats = get_collection_stats(client, collection_name)
                    if stats:
                        st.divider()
                        st.subheader("📊 登録結果")
                        st.json(stats)
                except Exception as e:
                    logger.warning(f"統計情報取得エラー: {e}")

            except Exception as e:
                add_log(f"❌ エラー発生: {str(e)}")
                st.error(f"エラーが発生しました: {str(e)}")

        # ログ表示
        with log_container:
            if st.session_state["qdrant_logs"]:
                log_text = "\n".join(st.session_state["qdrant_logs"])
                st.text_area("処理ログ", value=log_text, height=300, disabled=True)
            else:
                st.info("登録処理を開始するとここにログが表示されます")

    else:
        # ===================================================================
        # コレクション統合モード
        # ===================================================================
        st.subheader("🔗 コレクション統合")
        st.caption("複数のコレクションを選択して、1つの新しいコレクションに統合します")

        if not qdrant_connected or not client:
            st.warning("Qdrantに接続できていません")
            return

        # コレクション一覧を取得
        try:
            collections = get_all_collections(client)
        except Exception as e:
            st.error(f"コレクション一覧取得エラー: {e}")
            return

        if not collections:
            st.info("統合可能なコレクションがありません")
            return

        if len(collections) < 2:
            st.warning("統合には2つ以上のコレクションが必要です")
            return

        # コレクション選択UI
        st.markdown("### 統合対象コレクションを選択")
        st.caption("統合するコレクションを2つ以上選択してください")

        # コレクション情報を表示
        collection_names = [c["name"] for c in collections]
        collection_info_map = {c["name"]: c for c in collections}

        # チェックボックスで複数選択
        selected_collections = []
        selected_total_points = 0

        for col_info in collections:
            col_name = col_info["name"]
            points_count = col_info["points_count"]

            # チェックボックスを表示（コレクション名とポイント数を表示）
            is_selected = st.checkbox(
                f"{col_name} ({points_count:,} ポイント)",
                value=False,
                key=f"merge_checkbox_{col_name}",
            )

            if is_selected:
                selected_collections.append(col_name)
                selected_total_points += points_count

        # 選択されたコレクションのサマリー
        if selected_collections:
            st.divider()
            st.markdown("#### 選択されたコレクション")
            st.success(f"**{len(selected_collections)}** 件選択中 / 統合後の予想ポイント数: **{selected_total_points:,}** 件")

        st.divider()

        # 統合先コレクション名の設定
        st.markdown("### 統合先コレクション名")

        # デフォルト名を生成
        if selected_collections:
            default_merge_name = f"integration_{selected_collections[0]}"
        else:
            default_merge_name = "integration_"

        merge_collection_name = st.text_input(
            "新しいコレクション名",
            value=default_merge_name,
            help="統合先のコレクション名を入力してください",
            key="merge_collection_name_input",
        )

        # 名前の重複チェック
        name_exists = merge_collection_name in collection_names
        if name_exists:
            st.warning(f"コレクション '{merge_collection_name}' は既に存在します。上書きされます。")

        recreate_merge = st.checkbox(
            "既存コレクションを削除して再作成",
            value=True,
            help="同名のコレクションが存在する場合、削除して新規作成します",
            key="merge_recreate_checkbox",
        )

        st.divider()

        # 統合実行ボタン
        can_merge = len(selected_collections) >= 2 and merge_collection_name.strip()

        run_merge = st.button(
            "🔗 コレクションを統合",
            type="primary",
            use_container_width=True,
            disabled=not can_merge,
        )

        if not can_merge:
            if len(selected_collections) < 2:
                st.info("統合には2つ以上のコレクションを選択してください")
            elif not merge_collection_name.strip():
                st.info("統合先のコレクション名を入力してください")

        # ログ表示エリア
        st.subheader("📜 処理ログ")
        merge_log_container = st.container()

        if "merge_logs" not in st.session_state:
            st.session_state["merge_logs"] = []

        def add_merge_log(message: str):
            """統合ログを追加"""
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state["merge_logs"].append(f"[{timestamp}] {message}")

        # プログレスバー用のプレースホルダー
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        # 統合処理実行
        if run_merge:
            st.session_state["merge_logs"] = []  # ログクリア
            add_merge_log(f"🔗 統合処理開始")
            add_merge_log(f"統合元: {', '.join(selected_collections)}")
            add_merge_log(f"統合先: {merge_collection_name}")

            # プログレスバーを表示
            progress_bar = progress_placeholder.progress(0)

            def progress_callback(message: str, current: int, total: int):
                """進捗コールバック"""
                add_merge_log(message)
                if total > 0:
                    progress_bar.progress(current / total)
                status_placeholder.text(message)

            try:
                # 統合処理を実行
                result = merge_collections(
                    client=client,
                    source_collections=selected_collections,
                    target_collection=merge_collection_name,
                    recreate=recreate_merge,
                    progress_callback=progress_callback,
                )

                if result["success"]:
                    add_merge_log("🎉 統合処理完了！")

                    # 各コレクションからの取得件数をログ
                    for src_name, count in result["points_per_collection"].items():
                        add_merge_log(f"  - {src_name}: {count:,} 件")

                    add_merge_log(f"合計: {result['total_points']:,} 件")

                    st.success(
                        f"✅ {result['total_points']:,}件のデータを '{merge_collection_name}' に統合しました"
                    )

                    # 統計情報を表示
                    try:
                        stats = get_collection_stats(client, merge_collection_name)
                        if stats:
                            st.divider()
                            st.subheader("📊 統合結果")
                            st.json(stats)
                    except Exception as e:
                        logger.warning(f"統計情報取得エラー: {e}")

                else:
                    add_merge_log(f"❌ エラー: {result['error']}")
                    st.error(f"統合エラー: {result['error']}")

            except Exception as e:
                add_merge_log(f"❌ エラー発生: {str(e)}")
                st.error(f"エラーが発生しました: {str(e)}")
                logger.error(f"コレクション統合エラー: {e}")

            finally:
                # プログレスバーをクリア
                progress_placeholder.empty()
                status_placeholder.empty()

        # ログ表示
        with merge_log_container:
            if st.session_state["merge_logs"]:
                log_text = "\n".join(st.session_state["merge_logs"])
                st.text_area("処理ログ", value=log_text, height=300, disabled=True)
            else:
                st.info("統合処理を開始するとここにログが表示されます")
# streamlit run a34_rag_search_cloud_vs.py --server.port=8503
# a34_rag_search_cloud_vs.py - 最新OpenAI Responses API完全対応版（動的Vector Store対応・重複問題修正版）
# OpenAI Responses API + file_search ツール + 環境変数APIキー対応 + 動的Vector Store ID管理
"""
🔍 最新RAG検索アプリケーション（動的Vector Store対応・重複問題修正版）

【前提条件】
1. OpenAI APIキーの環境変数設定（必須）:
   export OPENAI_API_KEY='your-api-key-here'

2. 必要なライブラリ（必須）:
   pip install streamlit openai
   pip install openai-agents

【実行方法】
streamlit run a34_rag_search_cloud_vs.py --server.port=8501

【主要機能】
✅ 最新Responses API使用
✅ file_search ツールでVector Store検索
✅ 動的Vector Store ID管理（vector_stores.json）
✅ 重複Vector Store対応（最新優先選択）
✅ ファイル引用表示
✅ 型安全実装（型エラー完全修正）
✅ 環境変数でAPIキー管理
✅ 英語/日本語質問対応
✅ カスタマイズ可能な検索オプション
✅ 最新Vector Store自動取得・更新機能

【Vector Store連携】
- a30_020_make_vsid.py で作成されたVector Storeを自動認識
- vector_stores.json ファイルで動的管理
- 同名Vector Store重複時は最新作成日時を優先
- OpenAI APIから最新状態を取得・更新
"""
import streamlit as st
import logging
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import traceback

# OpenAI SDK のインポート
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ OpenAI SDK がロードされました")
except ImportError as e:
    OPENAI_AVAILABLE = False
    st.error(f"OpenAI SDK が見つかりません: {e}")
    st.stop()

# Agent SDK のインポート（オプション）
try:
    from agents import Agent, Runner, SQLiteSession

    AGENT_SDK_AVAILABLE = True
    logger.info("✅ OpenAI Agent SDK もロードされました")
except ImportError as e:
    AGENT_SDK_AVAILABLE = False
    logger.info(f"Agent SDK は利用できません（オプション機能のため問題なし）: {e}")

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===================================================================
# Vector Store設定管理クラス（重複問題修正版）
# ===================================================================
class VectorStoreManager:
    """Vector Store設定の動的管理（重複問題修正版）"""

    CONFIG_FILE_PATH = Path("vector_stores.json")

    # デフォルトのVector Store設定（フォールバック用）
    # 注: 辞書は挿入順序を保持（Python 3.7+）
    DEFAULT_VECTOR_STORES = {
        # CC News データセット（3種類の生成方式）
        "CC News Q&A (a02_llm)"      : "vs_cc_news_a02_llm_placeholder",  # CC News LLM生成方式
        "CC News Q&A (a03_rule)"     : "vs_cc_news_a03_rule_placeholder",  # CC News ルールベース生成方式
        "CC News Q&A (a10_hybrid)"   : "vs_cc_news_a10_hybrid_placeholder",  # CC News ハイブリッド生成方式
        # Livedoor データセット（3種類の生成方式）
        "Livedoor Q&A (a02_llm)"     : "vs_livedoor_a02_llm_placeholder",  # Livedoor LLM生成方式
        "Livedoor Q&A (a03_rule)"    : "vs_livedoor_a03_rule_placeholder",  # Livedoor ルールベース生成方式
        "Livedoor Q&A (a10_hybrid)"  : "vs_livedoor_a10_hybrid_placeholder",  # Livedoor ハイブリッド生成方式
        # その他のデータセット
        "Customer Support FAQ"       : "vs_68c94da49c80819189dd42d6e941c4b5",
        "Science & Technology Q&A"   : "vs_68c94db932fc8191b6e17f86e6601bc1",
        "Medical Q&A"                : "vs_68c94daffc708191b3c561f4dd6b2af8",
        "Legal Q&A"                  : "vs_68c94dc1cc008191a197bdbc3947a67b",
        "Trivia Q&A"                 : "vs_68c94dc9e6b08191946d7cafcd9880a3",
        "Unified Knowledge Base"     : "vs_unified_placeholder",  # 統合ナレッジベース（プレースホルダー）
    }

    # a30_020_make_vsid.py のVectorStoreConfigと対応するマッピング
    STORE_NAME_MAPPING = {
        "customer_support_faq": "Customer Support FAQ Knowledge Base",
        "medical_qa"          : "Medical Q&A Knowledge Base",
        "sciq_qa"             : "Science & Technology Q&A Knowledge Base",
        "legal_qa"            : "Legal Q&A Knowledge Base",
        "trivia_qa"           : "Trivia Q&A Knowledge Base",
        "unified_all"         : "Unified Knowledge Base - All Domains"
    }

    # 表示名への逆マッピング（UI表示用）
    DISPLAY_NAME_MAPPING = {
        "Customer Support FAQ Knowledge Base"              : "Customer Support FAQ",
        "Medical Q&A Knowledge Base"                       : "Medical Q&A",
        "Science & Technology Q&A Knowledge Base"          : "Science & Technology Q&A",
        "Legal Q&A Knowledge Base"                         : "Legal Q&A",
        "Trivia Q&A Knowledge Base"                        : "Trivia Q&A",
        "Unified Knowledge Base - All Domains"             : "Unified Knowledge Base",
        # CC News Q&A関連のマッピング
        "CC News Q&A - Basic Generation (a02_make_qa)"     : "CC News Q&A (LLM)",
        "CC News Q&A - Coverage Improved (a03_coverage)"   : "CC News Q&A (Coverage)",
        "CC News Q&A - Hybrid Method (a10_hybrid)"         : "CC News Q&A (Hybrid)"
    }

    def __init__(self, openai_client: OpenAI = None):
        self.openai_client = openai_client
        self._cache = {}
        self._last_update = None

    def load_vector_stores(self) -> Dict[str, str]:
        """Vector Store設定を読み込み"""
        try:
            if self.CONFIG_FILE_PATH.exists():
                with open(self.CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 設定ファイルの形式確認
                if 'vector_stores' in data and isinstance(data['vector_stores'], dict):
                    stores = data['vector_stores']
                    logger.info(f"✅ Vector Store設定を読み込み: {len(stores)}件")
                    return stores
                else:
                    logger.warning("⚠️ 設定ファイル形式が不正です")
                    return self.DEFAULT_VECTOR_STORES.copy()
            else:
                logger.info("ℹ️ 設定ファイルが存在しません。デフォルト値を使用")
                return self.DEFAULT_VECTOR_STORES.copy()

        except Exception as e:
            logger.error(f"❌ Vector Store設定読み込みエラー: {e}")
            st.warning(f"設定ファイル読み込みエラー: {e}")
            return self.DEFAULT_VECTOR_STORES.copy()

    def save_vector_stores(self, stores: Dict[str, str]) -> bool:
        """Vector Store設定を保存"""
        try:
            config_data = {
                "vector_stores": stores,
                "last_updated" : datetime.now().isoformat(),
                "source"       : "a34_rag_search_cloud_vs.py",
                "version"      : "1.1"
            }

            with open(self.CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Vector Store設定を保存: {self.CONFIG_FILE_PATH}")
            return True

        except Exception as e:
            logger.error(f"❌ Vector Store設定保存エラー: {e}")
            st.error(f"設定保存エラー: {e}")
            return False

    def fetch_latest_vector_stores(self) -> Dict[str, str]:
        """OpenAI APIから最新のVector Store一覧を取得し、既知の名前とマッチング（重複問題修正版）"""
        if not self.openai_client:
            logger.warning("⚠️ OpenAI クライアントが未設定です")
            return self.load_vector_stores()

        try:
            # OpenAI APIからVector Store一覧を取得
            stores_response = self.openai_client.vector_stores.list()

            # Vector Storeを作成日時でソート（新しい順）
            sorted_stores = sorted(
                stores_response.data,
                key=lambda x: x.created_at if hasattr(x, 'created_at') else 0,
                reverse=True
            )

            api_stores = {}
            store_candidates = {}  # 同名Store候補を管理

            logger.info(f"📊 取得したVector Store数: {len(sorted_stores)}")

            for store in sorted_stores:
                store_name = store.name
                store_id = store.id
                created_at = getattr(store, 'created_at', 0)

                logger.info(f"🔍 処理中: '{store_name}' ({store_id}) - 作成日時: {created_at}")

                # 既知のstore_nameパターンとのマッチング
                matched_display_name = None

                # 完全一致確認
                if store_name in self.DISPLAY_NAME_MAPPING:
                    matched_display_name = self.DISPLAY_NAME_MAPPING[store_name]
                else:
                    # 部分一致確認（柔軟なマッチング）
                    for full_name, display_name in self.DISPLAY_NAME_MAPPING.items():
                        if (full_name.lower() in store_name.lower() or
                                any(keyword in store_name.lower() for keyword in full_name.lower().split())):
                            matched_display_name = display_name
                            break

                if matched_display_name:
                    # 同名の場合は最新のもの（作成日時が新しい）を優先
                    if matched_display_name not in store_candidates:
                        store_candidates[matched_display_name] = {
                            'id'        : store_id,
                            'name'      : store_name,
                            'created_at': created_at
                        }
                        logger.info(f"✅ 新規候補: '{matched_display_name}' -> '{store_name}' ({store_id})")
                    else:
                        # 既存候補と比較
                        existing = store_candidates[matched_display_name]
                        if created_at > existing['created_at']:
                            logger.info(
                                f"🔄 更新: '{matched_display_name}' -> '{store_name}' ({store_id}) [新: {created_at} > 旧: {existing['created_at']}]")
                            store_candidates[matched_display_name] = {
                                'id'        : store_id,
                                'name'      : store_name,
                                'created_at': created_at
                            }
                        else:
                            logger.info(
                                f"⏭️ スキップ: '{matched_display_name}' -> '{store_name}' ({store_id}) [新: {created_at} <= 既存: {existing['created_at']}]")
                else:
                    # 既知パターンにマッチしない場合
                    if store_name not in store_candidates:
                        store_candidates[store_name] = {
                            'id'        : store_id,
                            'name'      : store_name,
                            'created_at': created_at
                        }
                        logger.info(f"ℹ️ 新規店舗: '{store_name}' ({store_id})")

            # 最終的なapi_storesを構築（DEFAULT_VECTOR_STORESの順序を維持）
            # まずDEFAULT_VECTOR_STORESの順序でソート
            default_order = list(self.DEFAULT_VECTOR_STORES.keys())

            # DEFAULT_VECTOR_STORESに含まれるものを先に追加
            for display_name in default_order:
                if display_name in store_candidates:
                    candidate = store_candidates[display_name]
                    api_stores[display_name] = candidate['id']
                    logger.info(f"🎯 最終選択: '{display_name}' -> {candidate['id']} (作成日時: {candidate['created_at']})")

            # DEFAULT_VECTOR_STORESに含まれない新規Storeを後に追加
            for display_name, candidate in store_candidates.items():
                if display_name not in api_stores:
                    api_stores[display_name] = candidate['id']
                    logger.info(f"🎯 最終選択（新規）: '{display_name}' -> {candidate['id']} (作成日時: {candidate['created_at']})")

            if api_stores:
                logger.info(f"✅ OpenAI APIから{len(api_stores)}個のVector Storeを取得完了")
                return api_stores
            else:
                logger.warning("⚠️ APIから有効なVector Storeが見つかりませんでした")
                return self.load_vector_stores()

        except Exception as e:
            logger.error(f"❌ OpenAI API取得エラー: {e}")
            logger.error(traceback.format_exc())
            st.warning(f"最新情報の取得に失敗しました: {e}")
            return self.load_vector_stores()

    def get_vector_stores(self, force_refresh: bool = False) -> Dict[str, str]:
        """Vector Store一覧を取得（キャッシュ機能付き）"""
        now = datetime.now()

        # 強制リフレッシュまたは初回取得の場合
        if force_refresh or not self._cache:
            logger.info("🔄 Vector Store情報を更新中...")

            # APIからの最新情報を取得
            if self.openai_client and st.session_state.get('auto_refresh_stores', True):
                try:
                    api_stores = self.fetch_latest_vector_stores()
                    self._cache = api_stores
                    self._last_update = now
                    return api_stores
                except Exception as e:
                    logger.warning(f"⚠️ API取得に失敗、設定ファイルから読み込み: {e}")

            # 設定ファイルから読み込み（フォールバック）
            stores = self.load_vector_stores()
            self._cache = stores
            self._last_update = now
            return stores

        # キャッシュチェック（5分間有効）
        if self._last_update and (now - self._last_update).seconds >= 300:
            logger.info("⏰ キャッシュ有効期限切れ、更新中...")
            return self.get_vector_stores(force_refresh=True)

        logger.info("💾 キャッシュから取得")
        return self._cache

    def refresh_and_save(self) -> Dict[str, str]:
        """最新のVector Store情報を取得して保存"""
        if not self.openai_client:
            st.error("OpenAI クライアントが設定されていません")
            return self.load_vector_stores()

        try:
            # キャッシュクリア
            self._cache = {}
            self._last_update = None

            # 最新情報を強制取得
            latest_stores = self.get_vector_stores(force_refresh=True)

            # 設定ファイルに保存
            if self.save_vector_stores(latest_stores):
                st.success(f"✅ Vector Store設定を更新しました（{len(latest_stores)}件）")

                # 詳細情報を表示
                with st.expander("📊 更新されたVector Store一覧", expanded=True):
                    for name, store_id in latest_stores.items():
                        st.write(f"**{name}**: `{store_id}`")

                return latest_stores
            else:
                st.error("❌ 設定の保存に失敗しました")
                return self.load_vector_stores()

        except Exception as e:
            st.error(f"❌ 更新処理エラー: {e}")
            logger.error(f"更新処理エラー: {e}")
            logger.error(traceback.format_exc())
            return self.load_vector_stores()

    def debug_vector_stores(self) -> Dict[str, Any]:
        """デバッグ用：Vector Store情報の詳細取得"""
        debug_info = {
            "config_file_exists": self.CONFIG_FILE_PATH.exists(),
            "cached_stores"     : self._cache,
            "last_update"       : self._last_update.isoformat() if self._last_update else None,
            "api_stores"        : {}
        }

        if self.openai_client:
            try:
                stores_response = self.openai_client.vector_stores.list()
                for store in stores_response.data:
                    debug_info["api_stores"][store.name] = {
                        "id"         : store.id,
                        "created_at" : store.created_at,
                        "file_counts": getattr(store, 'file_counts', None),
                        "usage_bytes": getattr(store, 'usage_bytes', None)
                    }
            except Exception as e:
                debug_info["api_error"] = str(e)

        return debug_info


# グローバル Vector Store Manager インスタンス
@st.cache_resource
def get_vector_store_manager():
    """Vector Store Manager のシングルトン取得"""
    try:
        openai_client = OpenAI()
        return VectorStoreManager(openai_client)
    except Exception as e:
        logger.warning(f"OpenAI クライアント初期化失敗: {e}")
        return VectorStoreManager()


# ===================================================================
# 動的Vector Store取得
# ===================================================================
def get_current_vector_stores(force_refresh: bool = False) -> Tuple[Dict[str, str], List[str]]:
    """現在のVector Store設定を取得"""
    manager = get_vector_store_manager()
    stores = manager.get_vector_stores(force_refresh=force_refresh)
    store_list = list(stores.keys())
    return stores, store_list


# helper_ragからモデル関連のインポート
try:
    from helper_rag import AppConfig, select_model, show_model_info
    HELPER_AVAILABLE = True
except ImportError as e:
    HELPER_AVAILABLE = False
    logger.warning(f"ヘルパーモジュールのインポートに失敗: {e}")

# テスト用質問（英語版 - RAGデータに最適化）
test_questions_en = [
    "How do I create a new account?",
    "What payment methods are available?",
    "Can I return a product?",
    "I forgot my password",
    "How can I contact the support team?"
]

test_questions_2_en = [
    "What are the latest trends in artificial intelligence?",
    "What is the principle of quantum computing?",
    "What are the types and characteristics of renewable energy?",
    "What are the current status and challenges of gene editing technology?",
    "What are the latest technologies in space exploration?"
]

test_questions_3_en = [
    "How to prevent high blood pressure?",
    "What are the symptoms and treatment of diabetes?",
    "What are the risk factors for heart disease?",
    "What are the guidelines for healthy eating?",
    "What is the relationship between exercise and health?"
]

test_questions_4_en = [
    "What are the important clauses in contracts?",
    "How to protect intellectual property rights?",
    "What are the basic principles of labor law?",
    "What is an overview of personal data protection law?",
    "What is the scope of application of consumer protection law?"
]

# 統合ナレッジベース用のテスト質問（全ドメインから）
test_questions_unified_en = [
    "How do I reset my password?",  # Customer Support
    "What is machine learning?",  # Science & Technology
    "What are the symptoms of flu?",  # Medical
    "What is a non-disclosure agreement?",  # Legal
    "Who invented the telephone?",  # Trivia
]

# CC News Q&A用のテスト質問（英語）
test_questions_cc_news_en = [
    "What are the latest developments in artificial intelligence?",
    "What happened in the recent tech industry news?",
    "What are the current trends in global markets?",
    "What are the major political events this week?",
    "What innovations are emerging in technology?"
]

# 日本語テスト質問は削除（英語のみ使用）

# OpenAI APIキーの設定（環境変数から自動取得）
try:
    # 環境変数 OPENAI_API_KEY から自動的に読み取り
    openai_client = OpenAI()
    logger.info("✅ OpenAI APIキーが環境変数から正常に読み込まれました")
except Exception as e:
    st.error(f"OpenAI API キーの設定に問題があります: {e}")
    st.error("環境変数 OPENAI_API_KEY が設定されているか確認してください")
    st.code("export OPENAI_API_KEY='your-api-key-here'")
    st.stop()


class ModernRAGManager:
    """最新Responses API + file_search を使用したRAGマネージャー"""

    def __init__(self):
        self.agent_sessions = {}  # Agent SDK用セッション（オプション）

    def search_with_responses_api(self, query: str, store_name: str, store_id: str, **kwargs) -> Tuple[
        str, Dict[str, Any]]:
        """最新Responses API + file_search ツールを使用した検索"""
        try:
            # file_search ツールの設定（正しい型で定義）
            file_search_tool_dict: Dict[str, Any] = {
                "type"            : "file_search",
                "vector_store_ids": [store_id]
            }

            # オプション設定（型安全な方法）
            max_results = kwargs.get('max_results', 20)
            include_results = kwargs.get('include_results', True)
            filters = kwargs.get('filters', None)
            selected_model = kwargs.get('selected_model', 'gpt-4o-mini')  # モデル選択を受け取る

            # 型安全な辞書更新
            if max_results and isinstance(max_results, int):
                file_search_tool_dict["max_num_results"] = max_results
            if filters is not None:
                file_search_tool_dict["filters"] = filters

            # include パラメータの設定
            include_params = []
            if include_results:
                include_params.append("file_search_call.results")

            # Responses API呼び出し（型安全な方法）
            # 選択されたモデルを使用
            response = openai_client.responses.create(
                model=selected_model,
                input=query,
                tools=[file_search_tool_dict],  # type: ignore[arg-type]
                include=include_params if include_params else None
            )

            # レスポンステキストの抽出
            response_text = self._extract_response_text(response)

            # ファイル引用の抽出
            citations = self._extract_citations(response)

            # メタデータの構築（型を明示的に指定）
            metadata: Dict[str, Any] = {
                "store_name": store_name,
                "store_id"  : store_id,
                "query"     : query,
                "timestamp" : datetime.now().isoformat(),
                "model"     : selected_model,  # 選択されたモデルを記録
                "method"    : "responses_api_file_search",
                "citations" : citations,
                "tool_calls": self._extract_tool_calls(response)
            }

            # 使用統計があれば追加（型安全な方法）
            if hasattr(response, 'usage') and response.usage is not None:
                try:
                    # ResponseUsageオブジェクトを辞書に変換
                    if hasattr(response.usage, 'model_dump'):
                        metadata["usage"] = response.usage.model_dump()
                    elif hasattr(response.usage, 'dict'):
                        metadata["usage"] = response.usage.dict()
                    else:
                        # 手動で属性を抽出
                        usage_dict = {}
                        for attr in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
                            if hasattr(response.usage, attr):
                                usage_dict[attr] = getattr(response.usage, attr)
                        metadata["usage"] = usage_dict
                except Exception as e:
                    logger.warning(f"使用統計の変換に失敗: {e}")
                    metadata["usage"] = str(response.usage)

            return response_text, metadata

        except Exception as e:
            error_msg = f"Responses API検索でエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())

            # エラー時のメタデータ（型安全）
            error_metadata: Dict[str, Any] = {
                "error"     : str(e),
                "method"    : "responses_api_error",
                "store_name": store_name,
                "store_id"  : store_id,
                "query"     : query,
                "timestamp" : datetime.now().isoformat()
            }
            return error_msg, error_metadata

    def search_with_agent_sdk(self, query: str, store_name: str, store_id: str) -> Tuple[str, Dict[str, Any]]:
        """Agent SDKを使用した検索（簡易版 - file_searchはResponses APIで実行）"""
        try:
            if not AGENT_SDK_AVAILABLE:
                logger.info("Agent SDK利用不可、Responses APIにフォールバック")
                return self.search_with_responses_api(query, store_name, store_id)

            # 注意: Agent SDKでのfile_searchツール統合は複雑なため、
            # 現在は簡易版として通常のAgent実行のみ行い、
            # 実際のRAG機能はResponses APIに委譲

            # Agent SDKセッションの取得/作成
            session_key = f"{store_name}_agent"
            if session_key not in self.agent_sessions:
                self.agent_sessions[session_key] = SQLiteSession(session_key)

            session = self.agent_sessions[session_key]

            # 簡易Agent作成（file_searchなし）
            agent = Agent(
                name=f"RAG_Agent_{store_name.replace(' ', '_')}",
                instructions=f"""
                You are a helpful assistant specializing in {store_name}.
                Provide informative and accurate responses based on your knowledge.
                Be professional and helpful in your responses.
                """,
                model="gpt-4o-mini"
            )

            # Runner実行（セッション管理のみの利点）
            result = Runner.run_sync(
                agent,
                query,
                session=session
            )

            response_text = result.final_output if hasattr(result, 'final_output') else str(result)

            # メタデータの構築
            metadata: Dict[str, Any] = {
                "store_name": store_name,
                "store_id"  : store_id,
                "query"     : query,
                "timestamp" : datetime.now().isoformat(),
                "model"     : "gpt-4o-mini",
                "method"    : "agent_sdk_simple_session",
                "note"      : "Agent SDKセッション管理のみ、RAG機能なし"
            }

            logger.info("Agent SDK検索完了（簡易版）")
            return response_text, metadata

        except Exception as e:
            error_msg = f"Agent SDK検索でエラーが発生: {str(e)}"
            logger.error(error_msg)
            logger.warning("Agent SDKエラーによりResponses APIにフォールバック")
            # Agent SDKが失敗した場合はResponses APIにフォールバック
            return self.search_with_responses_api(query, store_name, store_id)

    def search(self, query: str, store_name: str, store_id: str, use_agent_sdk: bool = True, **kwargs) -> Tuple[
        str, Dict[str, Any]]:
        """統合検索メソッド"""
        if use_agent_sdk and AGENT_SDK_AVAILABLE:
            return self.search_with_agent_sdk(query, store_name, store_id)
        else:
            return self.search_with_responses_api(query, store_name, store_id, **kwargs)

    def _extract_response_text(self, response) -> str:
        """レスポンスからテキストを抽出"""
        try:
            # output_text属性がある場合
            if hasattr(response, 'output_text'):
                return response.output_text

            # output配列から抽出
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    if hasattr(item, 'type') and item.type == "message":
                        if hasattr(item, 'content') and item.content:
                            for content in item.content:
                                if hasattr(content, 'type') and content.type == "output_text":
                                    return content.text

            return "レスポンステキストの抽出に失敗しました"

        except Exception as e:
            logger.error(f"レスポンステキスト抽出エラー: {e}")
            return f"テキスト抽出エラー: {e}"

    def _extract_citations(self, response) -> List[Dict[str, Any]]:
        """ファイル引用情報を抽出"""
        citations: List[Dict[str, Any]] = []
        try:
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    if hasattr(item, 'type') and item.type == "message":
                        if hasattr(item, 'content') and item.content:
                            for content in item.content:
                                if hasattr(content, 'annotations'):
                                    for annotation in content.annotations:
                                        if hasattr(annotation, 'type') and annotation.type == "file_citation":
                                            citations.append({
                                                "file_id" : getattr(annotation, 'file_id', ''),
                                                "filename": getattr(annotation, 'filename', ''),
                                                "index"   : getattr(annotation, 'index', 0)
                                            })
        except Exception as e:
            logger.error(f"引用情報抽出エラー: {e}")

        return citations

    def _extract_tool_calls(self, response) -> List[Dict[str, Any]]:
        """ツール呼び出し情報を抽出"""
        tool_calls: List[Dict[str, Any]] = []
        try:
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    if hasattr(item, 'type') and item.type == "file_search_call":
                        tool_calls.append({
                            "id"     : getattr(item, 'id', ''),
                            "type"   : "file_search",
                            "status" : getattr(item, 'status', ''),
                            "queries": getattr(item, 'queries', [])
                        })
        except Exception as e:
            logger.error(f"ツール呼び出し情報抽出エラー: {e}")

        return tool_calls


# グローバルインスタンス
@st.cache_resource
def get_rag_manager():
    """RAGマネージャーのシングルトン取得"""
    return ModernRAGManager()


def initialize_session_state():
    """セッション状態の初期化"""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    if 'selected_store' not in st.session_state:
        # 動的に最初のVector Storeを選択
        _, store_list = get_current_vector_stores()
        st.session_state.selected_store = store_list[0] if store_list else "Customer Support FAQ"
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = AppConfig.DEFAULT_MODEL if HELPER_AVAILABLE else "gpt-4o-mini"  # デフォルトモデル
    if 'use_agent_sdk' not in st.session_state:
        st.session_state.use_agent_sdk = False  # デフォルトはResponses API直接使用
    if 'search_options' not in st.session_state:
        st.session_state.search_options = {
            'max_results'    : 20,
            'include_results': True,
            'show_citations' : True
        }
    if 'auto_refresh_stores' not in st.session_state:
        st.session_state.auto_refresh_stores = True


def display_search_history():
    """検索履歴の表示"""
    st.header("🕒 検索履歴")

    if not st.session_state.search_history:
        st.info("検索履歴がありません")
        return

    # 履歴をエクスパンダーで表示
    for i, item in enumerate(st.session_state.search_history[:10]):  # 最新10件
        with st.expander(f"履歴 {i + 1}: {item['query'][:50]}..."):
            st.markdown(f"**質問:** {item['query']}")
            st.markdown(f"**Vector Store:** {item['store_name']}")
            st.markdown(f"**Store ID:** `{item.get('store_id', 'N/A')}`")
            st.markdown(f"**実行時間:** {item['timestamp']}")
            st.markdown(f"**検索方法:** {item.get('method', 'unknown')}")

            # 引用情報表示
            if 'citations' in item and item['citations']:
                st.markdown("**引用ファイル:**")
                for citation in item['citations']:
                    st.markdown(f"- {citation.get('filename', 'Unknown file')}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("再実行", key=f"rerun_{i}"):
                    st.session_state.current_query = item['query']
                    st.session_state.selected_store = item['store_name']
                    st.rerun()
            with col2:
                if st.button("詳細表示", key=f"detail_{i}"):
                    st.json(item)


def get_selected_store_index(selected_store: str, store_list: List[str]) -> int:
    """選択されたVector Storeのインデックスを取得"""
    try:
        return store_list.index(selected_store)
    except ValueError:
        return 0  # デフォルトは最初のインデックス


def get_test_questions_by_store(store_name: str) -> List[str]:
    """Vector Storeに応じたテスト質問を取得（英語のみ）"""
    # 動的なVector Storeに対応するためのマッピング（英語のみ）
    store_question_mapping = {
        "Customer Support FAQ"    : test_questions_en,
        "Science & Technology Q&A": test_questions_2_en,
        "Medical Q&A"             : test_questions_3_en,
        "Legal Q&A"               : test_questions_4_en,
        "Unified Knowledge Base"  : test_questions_unified_en,  # 統合ナレッジベース
        "CC News Q&A (LLM)"       : test_questions_cc_news_en,  # CC News LLM生成
        "CC News Q&A (Coverage)"  : test_questions_cc_news_en,  # CC Newsカバレッジ改良
        "CC News Q&A (Hybrid)"    : test_questions_cc_news_en,  # CC Newsハイブリッド
    }

    # 完全一致確認
    if store_name in store_question_mapping:
        return store_question_mapping[store_name]

    # 部分一致確認（柔軟対応）
    for mapped_store, questions in store_question_mapping.items():
        if (mapped_store.lower() in store_name.lower() or
                any(word in store_name.lower() for word in mapped_store.lower().split())):
            return questions

    # デフォルト（Customer Support FAQ）
    return test_questions_en


def display_test_questions():
    """テスト用質問の表示（動的Vector Store対応）"""
    # 現在選択されているVector Storeを取得
    selected_store = st.session_state.get('selected_store', 'Customer Support FAQ')

    # 対応する質問を取得
    questions = get_test_questions_by_store(selected_store)

    # ヘッダーの動的生成
    header = f"Test Questions ({selected_store})"
    st.header(f"💡 {header}")

    # RAGデータが英語の場合の注意書き
    st.success("✅ 英語質問（RAGデータに最適化）")

    if not questions:
        st.info("No test questions available for this Vector Store")
        return

    # 質問ボタンの表示
    for i, question in enumerate(questions):
        button_key = f"test_q_{selected_store}_{i}_{hash(question)}"
        if st.button(f"Q{i + 1}: {question}", key=button_key):
            st.session_state.current_query = question
            st.session_state.selected_store = selected_store
            st.rerun()


def display_vector_store_management():
    """Vector Store管理UI（重複問題修正版）"""
    st.header("🗄️ Vector Store管理（最新ID優先）")

    manager = get_vector_store_manager()

    col1, col2 = st.columns(2)
    with col1:
        st.write("**現在の設定ファイル**")
        if manager.CONFIG_FILE_PATH.exists():
            file_stat = manager.CONFIG_FILE_PATH.stat()
            st.success(f"✅ 存在 ({file_stat.st_size} bytes)")
            modified_time = datetime.fromtimestamp(file_stat.st_mtime)
            st.write(f"最終更新: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.warning("⚠️ 設定ファイル未作成")

    with col2:
        st.write("**操作**")
        if st.button("🔄 最新情報に更新", type="primary"):
            with st.spinner("最新のVector Store情報を取得中..."):
                manager.refresh_and_save()
                st.session_state['vector_stores_updated'] = datetime.now().isoformat()
                # キャッシュクリア
                st.cache_resource.clear()
                st.rerun()

        if st.button("📊 デバッグ情報表示"):
            debug_info = manager.debug_vector_stores()
            with st.expander("🔍 デバッグ情報", expanded=True):
                st.json(debug_info)

        if st.button("📁 設定ファイル表示"):
            if manager.CONFIG_FILE_PATH.exists():
                try:
                    with open(manager.CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                        config_content = f.read()
                    st.code(config_content, language='json')
                except Exception as e:
                    st.error(f"ファイル読み込みエラー: {e}")
            else:
                st.warning("設定ファイルが存在しません")


def display_system_info():
    """システム情報の表示"""
    with st.expander("🔧 システム情報", expanded=False):
        st.write("**利用可能な機能:**")
        st.write(f"- OpenAI SDK: {'✅' if OPENAI_AVAILABLE else '❌'}")
        st.write("- Responses API: ✅")
        st.write("- file_search ツール: ✅")
        st.write(f"- Agent SDK: {'✅（簡易版）' if AGENT_SDK_AVAILABLE else '❌'}")
        st.write("- Vector Store RAG: ✅")
        st.write("- ファイル引用: ✅")
        st.write("- 検索結果詳細: ✅")
        st.write("- 型安全実装: ✅")
        st.write("- 環境変数APIキー: ✅")
        st.write("- 動的Vector Store管理: ✅")
        st.write("- 重複ID解決: ✅（最新優先）")

        st.write("**APIキー設定:**")
        st.write("- 環境変数 `OPENAI_API_KEY` から自動取得")
        st.write("- Streamlit secrets.toml 不要")
        st.code("export OPENAI_API_KEY='your-api-key-here'")

        # 動的Vector Store情報
        st.write("**Vector Stores（動的・最新優先）:**")
        stores, _ = get_current_vector_stores()
        for i, (name, store_id) in enumerate(stores.items(), 1):
            st.write(f"{i}. {name}: `{store_id}`")

        if st.session_state.search_history:
            st.write(f"**検索履歴:** {len(st.session_state.search_history)} 件")

        # Vector Store連動情報
        st.write("**設定情報:**")
        selected_store = st.session_state.get('selected_store', 'Customer Support FAQ')
        selected_model = st.session_state.get('selected_model', 'gpt-4o-mini')

        st.write(f"- 選択Vector Store: {selected_store}")
        st.write(f"- 選択モデル: {selected_model}")
        st.write(f"- Agent SDK使用: {'有効' if st.session_state.get('use_agent_sdk', False) else '無効'}")
        st.write(f"- 自動更新: {'有効' if st.session_state.get('auto_refresh_stores', True) else '無効'}")


def display_search_options():
    """検索オプションの表示"""
    with st.expander("⚙️ 検索オプション", expanded=False):
        # 最大結果数
        max_results = st.slider(
            "最大検索結果数",
            min_value=1,
            max_value=50,
            value=st.session_state.search_options['max_results'],
            help="Vector Storeから取得する最大結果数"
        )
        st.session_state.search_options['max_results'] = max_results

        # 検索結果詳細を含める
        include_results = st.checkbox(
            "検索結果詳細を含める",
            value=st.session_state.search_options['include_results'],
            help="file_search_call.resultsを含める"
        )
        st.session_state.search_options['include_results'] = include_results

        # 引用表示
        show_citations = st.checkbox(
            "ファイル引用を表示",
            value=st.session_state.search_options['show_citations'],
            help="レスポンスにファイル引用情報を表示"
        )
        st.session_state.search_options['show_citations'] = show_citations

        # Agent SDK使用設定
        if AGENT_SDK_AVAILABLE:
            use_agent_sdk = st.checkbox(
                "Agent SDKを使用",
                value=st.session_state.use_agent_sdk,
                help="Agent SDKを使用してセッション管理を有効化"
            )
            st.session_state.use_agent_sdk = use_agent_sdk

        # Vector Store自動更新設定
        auto_refresh = st.checkbox(
            "Vector Store自動更新",
            value=st.session_state.auto_refresh_stores,
            help="起動時にOpenAI APIから最新のVector Store情報を取得"
        )
        st.session_state.auto_refresh_stores = auto_refresh


def generate_enhanced_response(query: str, search_result: str, has_result: bool = True) -> Tuple[str, Dict[str, Any]]:
    """検索結果を基に、より自然な日本語回答を生成"""
    try:
        # モデル選択
        selected_model = st.session_state.get('selected_model', 'gpt-4o-mini')
        
        # プロンプトの構成
        if has_result and search_result and search_result.strip():
            # 検索結果がある場合
            system_prompt = """あなたは親切なアシスタントです。提供された検索結果を基に、
ユーザーの質問に対して正確で分かりやすい日本語の回答を生成してください。
検索結果から関連する情報を抽出し、自然な日本語で説明してください。"""
            
            user_prompt = f"""以下の検索結果を参考にして、質問に日本語で回答してください。

【検索結果】
{search_result}

【質問】
{query}

この検索結果から取り出して、日本語で回答してください。"""
        else:
            # 検索結果がない場合
            system_prompt = """あなたは親切なアシスタントです。
ユーザーの質問に対して、あなたの知識を基に正確で分かりやすい日本語の回答を生成してください。"""
            
            user_prompt = f"""Vector Storeからの検索結果が見つかりませんでした。
一般的な知識を基に、以下の質問に日本語で回答してください。

【質問】
{query}"""

        # ChatCompletion APIを呼び出し
        # モデルに応じてパラメータを調整
        model_lower = selected_model.lower()

        # 基本パラメータ
        completion_params = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        # o-seriesモデルと gpt-5-mini は temperature をサポートしない（デフォルト1.0のみ）
        no_temperature_models = any(prefix in model_lower for prefix in ['o1-', 'o3-', 'o4-', 'gpt-5-mini'])
        if not no_temperature_models:
            completion_params["temperature"] = 0.7

        # gpt-4.1, gpt-5, gpt-5-mini, o-seriesモデルは max_completion_tokens を使用
        if any(prefix in model_lower for prefix in ['gpt-5-mini', 'gpt-4.1', 'gpt-5', 'o1-', 'o3-', 'o4-']):
            completion_params["max_completion_tokens"] = 2000
        else:
            completion_params["max_tokens"] = 2000

        response = openai_client.chat.completions.create(**completion_params)
        
        # 回答の抽出
        enhanced_response = response.choices[0].message.content
        
        # メタデータの構築
        metadata = {
            "model": selected_model,
            "has_search_result": has_result,
            "timestamp": datetime.now().isoformat(),
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
        }
        
        return enhanced_response, metadata
        
    except Exception as e:
        error_msg = f"回答生成エラー: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return error_msg, {"error": str(e), "timestamp": datetime.now().isoformat()}


def display_search_results(response_text: str, metadata: Dict[str, Any], original_query: str):
    """検索結果の表示（日本語回答生成機能付き）"""
    st.markdown("### 🤖 回答")
    st.markdown(response_text)

    # ファイル引用の表示
    if metadata.get('citations') and st.session_state.search_options['show_citations']:
        st.markdown("### 📚 引用ファイル")
        citations = metadata['citations']
        for i, citation in enumerate(citations, 1):
            st.markdown(f"{i}. **{citation.get('filename', 'Unknown file')}** (ID: `{citation.get('file_id', '')}`)")

    # メタデータ表示
    st.markdown("---")
    st.markdown("### 📊 検索情報")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Vector Store:** {metadata.get('store_name', '')}")
        st.markdown(f"**Store ID:** `{metadata.get('store_id', '')}`")
        st.markdown(f"**検索方法:** {metadata.get('method', '')}")

    with col2:
        st.markdown(f"**モデル:** {metadata.get('model', '')}")
        st.markdown(f"**実行時間:** {metadata.get('timestamp', '')}")
        if 'tool_calls' in metadata and metadata['tool_calls']:
            st.markdown(f"**ツール呼び出し:** {len(metadata['tool_calls'])}回")

    # 詳細情報
    with st.expander("🔍 詳細情報", expanded=False):
        st.json(metadata)
    
    # 日本語での追加回答生成
    st.markdown("---")
    st.markdown("### 🇯🇵 日本語回答（検索結果を基に生成）")
    
    with st.spinner("日本語回答を生成中..."):
        # 検索結果の有無を判定
        has_result = bool(response_text and 
                         response_text.strip() and 
                         "エラー" not in response_text and
                         "見つかりません" not in response_text)
        
        # 日本語回答を生成
        enhanced_response, enhanced_metadata = generate_enhanced_response(
            original_query, 
            response_text,
            has_result
        )
        
        # 日本語回答を表示
        if not has_result:
            st.info("ℹ️ Vector Storeに関連情報が見つからなかったため、一般的な知識から回答します。")
        
        st.markdown(enhanced_response)
        
        # 生成情報を表示
        with st.expander("📊 日本語回答生成情報", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**使用モデル:** {enhanced_metadata.get('model', '')}")
                st.markdown(f"**検索結果利用:** {'あり' if enhanced_metadata.get('has_search_result') else 'なし'}")
            with col2:
                if 'usage' in enhanced_metadata:
                    usage = enhanced_metadata['usage']
                    st.markdown("**トークン使用量:**")
                    st.markdown(f"- 入力: {usage.get('prompt_tokens', 0):,}")
                    st.markdown(f"- 出力: {usage.get('completion_tokens', 0):,}")
                    st.markdown(f"- 合計: {usage.get('total_tokens', 0):,}")


def main():
    """メイン関数"""
    st.set_page_config(
        page_title="最新RAG検索アプリ（重複問題修正版）",
        page_icon="🔍",
        layout="wide"
    )

    # セッション状態の初期化（デフォルト値設定）
    initialize_session_state()

    # RAGマネージャーの取得
    rag_manager = get_rag_manager()

    # Vector Store設定の取得（強制リフレッシュは初回のみ）
    force_refresh = st.session_state.get('force_initial_refresh', True)
    if force_refresh:
        st.session_state['force_initial_refresh'] = False

    vector_stores, vector_store_list = get_current_vector_stores(force_refresh=force_refresh)

    # メインタイトルと使い方を最上部に配置
    st.header("🔍 RAG検索（Cloud:OpenAI Vector Store版）")
    
    # 使い方をExpanderで表示
    with st.expander("📖 使い方", expanded=False):
        st.markdown("""
        **RAG検索システムの使い方:**
        
        1. **Vector Storeの選択**
           - 左ペインで使いたい Vector Store を選択します
           - 各Vector Storeには異なるドメインの知識が格納されています
        
        2. **質問の入力方法**
           - **方法1**: 左ペインのテスト用質問から選択
           - **方法2**: 下の入力欄で直接質問を入力
        
        3. **検索の実行**
           - 質問を入力後、「🔍 検索実行」ボタンを押下
           - OpenAI Vector Storeから関連情報を検索し、回答を生成します
        
        **💡 ヒント:**
        - RAGデータは英語で作成されているため、英語での質問を推奨
        - 各Vector Storeに適した質問内容を選択すると精度が向上します
        """)
    
    # サブヘッダー
    st.write("🔍 最新RAG検索アプリケーション（重複問題修正・最新ID優先版）")

    # API状況表示
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ OpenAI Responses API 利用可能")
        st.success("✅ file_search ツール対応")
    with col2:
        if AGENT_SDK_AVAILABLE:
            st.success("✅ Agent SDK 利用可能")
        else:
            st.info("ℹ️ Agent SDK 未利用（Responses APIのみ）")
    with col3:
        st.success("✅ 動的Vector Store管理")
        st.success("🔄 重複ID解決（最新優先）")
        st.info(f"📊 利用可能店舗: {len(vector_stores)}件")

    st.markdown("---")

    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")

        # Vector Store選択（動的）
        if vector_store_list:
            # 現在の選択が利用可能かチェック
            current_selected = st.session_state.get('selected_store', vector_store_list[0])
            if current_selected not in vector_store_list:
                current_selected = vector_store_list[0]
                st.session_state.selected_store = current_selected

            selected_store = st.selectbox(
                "Vector Store を選択",
                options=vector_store_list,
                index=vector_store_list.index(current_selected),
                key="store_selection"
            )
            st.session_state.selected_store = selected_store

            # 選択されたVector Store IDを表示
            selected_store_id = vector_stores.get(selected_store, "未知のID")
            st.code(selected_store_id)

            # ID更新状況表示
            if st.session_state.get('vector_stores_updated'):
                update_time = st.session_state['vector_stores_updated']
                update_dt = datetime.fromisoformat(update_time)
                st.caption(f"最終更新: {update_dt.strftime('%H:%M:%S')}")
        else:
            st.error("❌ 利用可能なVector Storeがありません")
            st.stop()

        # モデル選択
        st.markdown("---")
        st.subheader("🤖 モデル選択")
        
        if HELPER_AVAILABLE:
            # helper_ragのselect_model関数をサイドバー外で使用
            models = AppConfig.AVAILABLE_MODELS
            default_model = st.session_state.get('selected_model', AppConfig.DEFAULT_MODEL)
            
            try:
                default_index = models.index(default_model)
            except ValueError:
                default_index = 0
            
            selected_model = st.selectbox(
                "使用モデル",
                models,
                index=default_index,
                key="model_selection",
                help="RAG検索に使用するOpenAIモデルを選択してください"
            )
            st.session_state.selected_model = selected_model
            
            # モデル情報表示
            limits = AppConfig.get_model_limits(selected_model)
            pricing = AppConfig.get_model_pricing(selected_model)
            
            st.info(f"""
            📄 **{selected_model}**
            - 最大トークン: {limits['max_tokens']:,}
            - 最大出力: {limits['max_output']:,}
            - 料金: ${pricing['input']:.4f} / ${pricing['output']:.4f} (input/output per 1K tokens)
            """)
        else:
            # フォールバック：シンプルなモデル選択
            models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"]
            selected_model = st.selectbox(
                "使用モデル",
                models,
                index=models.index(st.session_state.get('selected_model', 'gpt-4o-mini')),
                key="model_selection",
                help="RAG検索に使用するOpenAIモデルを選択してください"
            )
            st.session_state.selected_model = selected_model

        # 検索オプション
        display_search_options()

        # Vector Store管理
        st.markdown("---")
        with st.expander("🗄️ Vector Store管理", expanded=False):
            display_vector_store_management()

        # システム情報
        display_system_info()

        # テスト用質問（選択されたVector Storeに対応）
        with st.expander("💡 テスト用質問", expanded=True):
            display_test_questions()

    # 質問入力部分（横1段構成）
    st.header("❓ 質問入力")

    # 質問入力フォーム
    with st.form("search_form"):
        query = st.text_area(
            "質問を入力してください",
            value=st.session_state.current_query,
            height=100,
            key="query_input",
            help="英語での質問がRAGデータに最適化されています"
        )

        col_left, col_center, col_right = st.columns([1, 1, 1])
        with col_center:
            submitted = st.form_submit_button("🔍 検索実行", type="primary", use_container_width=True)
    
    # 質問例を追加
    with st.expander("💡 質問例", expanded=False):
        st.markdown("**質問例:**")
        example_questions = [
            "How can I create an account?",
            "新規アカウントは、どう作成すればよいか？　日本語で回答してください。",
            "What are the symptoms of diabetes?",
            "What is quantum computing?"
        ]
        for example_question in example_questions:
            if st.button(f"📝 {example_question}", use_container_width=True, key=f"example_{hash(example_question)}"):
                st.session_state.current_query = example_question
                st.rerun()

    # 検索結果表示部分
    if submitted and query.strip():
        st.session_state.current_query = query
        
        st.markdown("---")
        st.header("🤖 検索結果")

        with st.spinner("🔍 Vector Store検索中..."):
            # 選択されたVector StoreのIDを取得
            selected_store_id = vector_stores.get(selected_store, "")
            if not selected_store_id:
                st.error(f"❌ Vector Store ID が見つかりません: {selected_store}")
            else:
                # 検索オプションの取得
                search_options = st.session_state.search_options

                # 検索実行（store_idと選択されたモデルを渡す）
                final_result, final_metadata = rag_manager.search(
                    query,
                    selected_store,
                    selected_store_id,
                    use_agent_sdk=st.session_state.use_agent_sdk,
                    max_results=search_options['max_results'],
                    include_results=search_options['include_results'],
                    selected_model=st.session_state.selected_model  # 選択されたモデルを渡す
                )

                # 結果表示（元の質問も渡す）
                display_search_results(final_result, final_metadata, query)

                # 検索履歴に追加（型安全）
                history_item: Dict[str, Any] = {
                    "query"         : query,
                    "store_name"    : selected_store,
                    "store_id"      : selected_store_id,
                    "timestamp"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "method"        : final_metadata.get('method', 'unknown'),
                    "citations"     : final_metadata.get('citations', []),
                    "result_preview": final_result[:200] + "..." if len(final_result) > 200 else final_result
                }

                # 重複チェック
                if not any(item['query'] == query and item['store_name'] == selected_store
                           for item in st.session_state.search_history):
                    st.session_state.search_history.insert(0, history_item)
                    st.session_state.search_history = st.session_state.search_history[:50]  # 最新50件保持

    elif submitted and not query.strip():
        st.error("質問を入力してください")

    # 初期状態の表示
    if not st.session_state.current_query:
        st.info("質問を入力して検索を実行してください")

        # API機能説明
        st.markdown("### 🚀 利用可能な機能")
        st.markdown("""
        - **最新Responses API**: OpenAIの最新API
        - **file_search ツール**: Vector Storeからの高精度検索
        - **動的Vector Store管理**: 自動ID更新・設定ファイル連携
        - **重複ID解決**: 同名Vector Storeの最新作成日時優先
        - **ファイル引用**: 検索結果の出典表示
        - **カスタマイズ可能**: 結果数、フィルタリング等
        - **Agent SDK連携**: セッション管理（オプション）
        - **型安全実装**: 型エラー修正済み
        - **環境変数APIキー**: セキュアな設定方法
        """)

        # 重複問題修正の説明
        with st.expander("🔄 重複ID問題修正について", expanded=False):
            st.markdown("""
            **修正内容: 同名Vector Storeの重複問題解決**

            **問題:**
            - 同じ名前で複数のVector Storeが存在
            - 古いIDが選択されるバグ
            - 作成日時での優先度が未実装

            **修正:**
            - **作成日時ソート**: Vector Store一覧を作成日時順（新しい順）でソート
            - **最新優先選択**: 同名の場合は`created_at`が最新のものを優先
            - **詳細ログ出力**: どのIDが選択されたかをログで確認可能
            - **デバッグ機能**: サイドバーで詳細情報を確認可能

            **選択ロジック:**
            1. OpenAI APIからVector Store一覧を取得
            2. 作成日時(`created_at`)で降順ソート
            3. 同名Store候補の中から最新を選択
            4. 設定ファイルに保存・キャッシュ

            **確認方法:**
            - サイドバー「Vector Store管理」→「デバッグ情報表示」
            - ログでどのIDが選択されたかを確認
            """)

        # Vector Store動的管理の説明
        with st.expander("🗄️ 動的Vector Store管理について", expanded=False):
            st.markdown("""
            **新機能: 動的Vector Store管理**

            - **自動更新**: OpenAI APIから最新のVector Store一覧を取得
            - **設定ファイル連携**: `vector_stores.json` で永続化
            - **a30_020_make_vsid.py 連携**: 新規作成されたVector Storeを自動認識
            - **フォールバック**: 設定ファイルがない場合はデフォルト値を使用

            **設定ファイル形式:**
            ```json
            {
              "vector_stores": {
                "Customer Support FAQ": "vs_xxx...",
                "Medical Q&A": "vs_yyy...",
                ...
              },
              "last_updated": "2025-01-XX...",
              "source": "a34_rag_search_cloud_vs.py",
              "version": "1.1"
            }
            ```

            **更新方法:**
            1. サイドバーの「Vector Store管理」で「最新情報に更新」をクリック
            2. 自動でOpenAI APIから最新一覧を取得（重複解決済み）
            3. 設定ファイルに保存して永続化
            """)

        # トラブルシューティング
        with st.expander("🚨 トラブルシューティング", expanded=False):
            st.markdown("""
            **重複ID問題の場合:**
            - サイドバー「Vector Store管理」→「最新情報に更新」をクリック
            - 「デバッグ情報表示」で選択されたIDを確認
            - ログで最新作成日時のIDが選択されているかを確認

            **APIキーエラーの場合:**
            ```bash
            # 環境変数確認
            echo $OPENAI_API_KEY

            # 設定方法
            export OPENAI_API_KEY='your-api-key-here'

            # 永続化（.bashrc/.zshrcに追加）
            echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
            ```

            **Vector Store関連エラー:**
            - Vector Store IDが正しいか確認
            - 「最新情報に更新」ボタンで再取得
            - a30_020_make_vsid.py で新規作成後は更新が必要

            **その他のエラー:**
            - OpenAI SDKが最新版か確認: `pip install --upgrade openai`
            - インターネット接続を確認
            - vector_stores.json ファイルの形式を確認
            """)

    # 検索履歴セクション
    st.markdown("---")
    display_search_history()

    # フッター
    st.markdown("---")
    st.markdown("#### 最新RAG検索アプリケーション（重複問題修正・最新ID優先版）")
    st.markdown("🚀 **OpenAI Responses API + file_search ツール** による次世代RAG")
    st.markdown("✨ **修正機能**: 重複Vector Store ID問題解決、最新作成日時優先")
    st.markdown("🔗 **a30_020_make_vsid.py 連携**: 新規Vector Store自動認識")
    st.markdown("🔑 **セキュリティ**: 環境変数でのAPIキー管理")
    if AGENT_SDK_AVAILABLE:
        st.markdown("🔧 **Agent SDK**: セッション管理サポート（簡易版）")
    else:
        st.markdown("⚡ **高性能**: 直接Responses API使用")


if __name__ == "__main__":
    main()

# streamlit run a34_rag_search_cloud_vs.py --server.port=8501

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
explanation_page.py - システム説明ページ
========================================
README.md の内容を表示（Mermaid図対応）
"""

import re
import streamlit as st
from pathlib import Path

try:
    import streamlit_mermaid as stmd

    MERMAID_AVAILABLE = True
except ImportError:
    MERMAID_AVAILABLE = False


def render_markdown_with_mermaid(content: str):
    """
    Mermaid コードブロックを含む Markdown を表示する。
    Mermaid 部分は streamlit-mermaid で、それ以外は st.markdown() で表示。
    """
    # Mermaid コードブロックを検出するパターン
    mermaid_pattern = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)

    # コンテンツを分割
    last_end = 0
    for match in mermaid_pattern.finditer(content):
        # Mermaid の前のマークダウン部分を表示
        before_text = content[last_end : match.start()]
        if before_text.strip():
            st.markdown(before_text)

        # Mermaid 図を表示
        mermaid_code = match.group(1).strip()
        if MERMAID_AVAILABLE:
            try:
                stmd.st_mermaid(mermaid_code)
            except Exception as e:
                st.code(mermaid_code, language="mermaid")
                st.warning(f"Mermaid 図のレンダリングに失敗: {e}")
        else:
            st.code(mermaid_code, language="mermaid")
            st.info("Mermaid 図を表示するには: pip install streamlit-mermaid")

        last_end = match.end()

    # 残りのマークダウン部分を表示
    remaining_text = content[last_end:]
    if remaining_text.strip():
        st.markdown(remaining_text)


def show_system_explanation_page():
    """システム説明ページ - README.md を表示"""
    st.title("📖 システム説明")
    st.caption("プロジェクト README.md")

    st.markdown("---")

    # README.md のパスを取得
    readme_path = Path(__file__).parent.parent.parent / "README.md"

    if readme_path.exists():
        readme_content = readme_path.read_text(encoding="utf-8")
        render_markdown_with_mermaid(readme_content)
    else:
        st.error(f"README.md が見つかりません: {readme_path}")
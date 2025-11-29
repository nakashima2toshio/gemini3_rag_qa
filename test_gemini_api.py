"""
Gemini 3 API 動作確認スクリプト

Phase 0-3: API疎通確認用

使用方法:
    1. .env に GOOGLE_API_KEY を設定
    2. python test_gemini_api.py を実行

期待される出力:
    - テキスト生成の結果
    - Embedding生成の次元数（3072次元）
"""

import os
import sys
from dotenv import load_dotenv

# .env 読み込み
load_dotenv()

# APIキー確認
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY が設定されていません")
    print()
    print("設定方法:")
    print("  1. Google AI Studio (https://aistudio.google.com/) でAPIキーを取得")
    print("  2. .env ファイルに以下を追加:")
    print("     GOOGLE_API_KEY=AIza...")
    sys.exit(1)

print("=" * 60)
print("Gemini 3 API 動作確認")
print("=" * 60)
print()

try:
    from google import genai

    # クライアント初期化
    client = genai.Client(api_key=api_key)
    print("[OK] google-genai クライアント初期化成功")
    print()

    # -------------------------------------------
    # テスト1: テキスト生成
    # -------------------------------------------
    print("-" * 40)
    print("テスト1: テキスト生成 (gemini-2.0-flash)")
    print("-" * 40)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="こんにちは。あなたの名前と、今日の日付を教えてください。",
        config={
            "temperature": 1.0,
        }
    )

    print(f"[OK] テキスト生成成功")
    print(f"Response: {response.text[:200]}...")
    print()

    # -------------------------------------------
    # テスト2: Embedding生成（3072次元）
    # -------------------------------------------
    print("-" * 40)
    print("テスト2: Embedding生成 (gemini-embedding-001, 3072次元)")
    print("-" * 40)

    embed_response = client.models.embed_content(
        model="gemini-embedding-001",
        contents="これはテスト文章です。Gemini 3のEmbedding機能を検証します。",
        config={"output_dimensionality": 3072}
    )

    embedding = embed_response.embeddings[0].values
    print(f"[OK] Embedding生成成功")
    print(f"次元数: {len(embedding)} (期待値: 3072)")
    print(f"先頭5要素: {embedding[:5]}")
    print()

    # 次元数検証
    if len(embedding) == 3072:
        print("[OK] 3072次元の検証: PASS")
    else:
        print(f"[NG] 3072次元の検証: FAIL (実際: {len(embedding)})")
    print()

    # -------------------------------------------
    # テスト3: 構造化出力（JSONスキーマ）
    # -------------------------------------------
    print("-" * 40)
    print("テスト3: 構造化出力（JSONスキーマ）")
    print("-" * 40)

    from pydantic import BaseModel
    from typing import List
    import json

    class QAPair(BaseModel):
        question: str
        answer: str

    class QAPairsResponse(BaseModel):
        qa_pairs: List[QAPair]

    structured_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="以下の文章から、Q&Aペアを2つ作成してください。\n\n文章：富士山は日本で最も高い山で、標高は3,776メートルです。",
        config={
            "response_mime_type": "application/json",
            "response_schema": QAPairsResponse,
            "temperature": 1.0,
        }
    )

    print(f"[OK] 構造化出力生成成功")
    print(f"Raw Response: {structured_response.text}")

    # Pydanticでパース
    qa_data = json.loads(structured_response.text)
    parsed = QAPairsResponse(**qa_data)
    print(f"パース成功: {len(parsed.qa_pairs)} Q&Aペア生成")
    for i, qa in enumerate(parsed.qa_pairs, 1):
        print(f"  Q{i}: {qa.question}")
        print(f"  A{i}: {qa.answer}")
    print()

    # -------------------------------------------
    # 結果サマリー
    # -------------------------------------------
    print("=" * 60)
    print("全テスト完了: SUCCESS")
    print("=" * 60)
    print()
    print("Gemini 3 APIは正常に動作しています。")
    print("Phase 1（抽象化レイヤー設計）に進む準備ができました。")

except ImportError as e:
    print(f"[ERROR] google-genai パッケージが見つかりません: {e}")
    print("pip install google-genai を実行してください")
    sys.exit(1)

except Exception as e:
    print(f"[ERROR] API呼び出しエラー: {e}")
    print()
    print("考えられる原因:")
    print("  - APIキーが無効")
    print("  - ネットワーク接続の問題")
    print("  - APIクォータ超過")
    sys.exit(1)
#!/usr/bin/env python3
"""
sample_semantic_chunks.py - SemanticCoverage.create_semantic_chunks() のサンプル

helper_rag_qa.py の SemanticCoverage クラスを使用して、
テキストをセマンティック分割するデモンストレーション。

実行方法:
    python sample_semantic_chunks.py
"""

from helper_rag_qa import SemanticCoverage


def main():
    # テスト用テキスト
    test_text = """
    人工知能（AI）は、機械学習と深層学習を基盤として急速に発展しています。
    特に自然言語処理（NLP）の分野では、トランスフォーマーモデルが革命的な成果を上げました。
    BERTやGPTなどの大規模言語モデルは、文脈理解能力を大幅に向上させています。
    最新の研究では、小規模データセットでも高性能を実現する手法が開発されています。
    """

    print("=" * 70)
    print("SemanticCoverage.create_semantic_chunks() サンプル")
    print("=" * 70)

    # SemanticCoverageインスタンスを作成
    analyzer = SemanticCoverage(embedding_model="text-embedding-3-small")

    print("\n【入力テキスト】")
    print(test_text.strip())
    print("\n" + "-" * 70)

    # セマンティック分割を実行
    print("\n【セマンティック分割実行】")
    print("パラメータ: max_tokens=200, min_tokens=50, prefer_paragraphs=True")
    print("-" * 70)

    chunks = analyzer.create_semantic_chunks(
        document=test_text,
        max_tokens=200,
        min_tokens=50,
        prefer_paragraphs=True,
        verbose=True
    )

    # 結果を表示
    print("\n" + "=" * 70)
    print(f"【分割結果】チャンク数: {len(chunks)}")
    print("=" * 70)

    for i, chunk in enumerate(chunks):
        print(f"\n--- チャンク {i + 1} ---")
        print(f"ID: {chunk.get('id', 'N/A')}")
        print(f"Type: {chunk.get('type', 'N/A')}")
        print(f"Text: {chunk.get('text', '')[:100]}...")
        print(f"Sentences: {len(chunk.get('sentences', []))}文")

        # 各文を表示
        sentences = chunk.get('sentences', [])
        if sentences:
            print("  文リスト:")
            for j, sent in enumerate(sentences):
                print(f"    [{j + 1}] {sent[:50]}..." if len(sent) > 50 else f"    [{j + 1}] {sent}")

    # 追加テスト: 段落を含むテキスト
    print("\n" + "=" * 70)
    print("【追加テスト】段落を含むテキスト")
    print("=" * 70)

    test_text_with_paragraphs = """
人工知能の基礎
人工知能（AI）は、人間の知能を模倣するコンピュータシステムです。
機械学習、深層学習、自然言語処理などの技術が含まれます。

応用分野
AIは医療診断、自動運転、音声認識など幅広い分野で活用されています。
特に画像認識の精度は人間を超えるレベルに達しています。

将来の展望
AIの発展により、多くの産業が変革を迎えると予想されています。
倫理的な課題への対応も重要なテーマとなっています。
"""

    print("\n【入力テキスト（段落あり）】")
    print(test_text_with_paragraphs.strip())
    print("\n" + "-" * 70)

    chunks_para = analyzer.create_semantic_chunks(
        document=test_text_with_paragraphs,
        max_tokens=200,
        min_tokens=50,
        prefer_paragraphs=True,
        verbose=True
    )

    print("\n" + "=" * 70)
    print(f"【分割結果】チャンク数: {len(chunks_para)}")
    print("=" * 70)

    for i, chunk in enumerate(chunks_para):
        print(f"\n--- チャンク {i + 1} ---")
        print(f"ID: {chunk.get('id', 'N/A')}")
        print(f"Type: {chunk.get('type', 'N/A')}")
        print(f"Text ({len(chunk.get('text', ''))}文字):")
        print(f"  {chunk.get('text', '')}")


if __name__ == "__main__":
    main()

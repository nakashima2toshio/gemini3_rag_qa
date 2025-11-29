# MeCabを使用した日本語キーワード抽出の改良版
import collections
from typing import List
import re

def extract_keywords_mecab(text: str, top_n: int = 5, use_compound: bool = True) -> List[str]:
    """
    MeCabを使用してテキストから重要なキーワードを抽出（改良版）
    Args:
        text: 分析対象のテキスト
        top_n: 抽出するキーワード数
        use_compound: 複合名詞を抽出するか（True: 複合名詞、False: 単名詞）
    Returns:
        重要キーワードのリスト
    """
    try:
        import MeCab
    except ImportError:
        print("MeCabがインストールされていません。")
        print("インストール方法:")
        print("  pip install mecab-python3")
        print("  pip install unidic-lite  # 辞書")
        return extract_keywords_regex(text, top_n)  # フォールバック
    
    # MeCabの初期化（UniDic形式）
    try:
        tagger = MeCab.Tagger()
    except Exception as e:
        print(f"MeCab初期化エラー: {e}")
        return extract_keywords_regex(text, top_n)

    # 形態素解析の実行
    parsed = tagger.parse(text)

    # ストップワード（より詳細な定義）
    stopwords = {
        'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん', 'ところ', 'とき',
        'ます', 'です', 'ある', 'いる', 'する', 'なる', 'できる', 'やる', 'くる',
        'いう', '的', 'な', 'に', 'を', 'は', 'が', 'で', 'と', 'の', 'から', 'まで',
        'へ', 'より', 'など', 'ら', 'たち', 'ない', 'れる', 'られる', 'せる', 'させる'
    }

    # 名詞を抽出
    nouns = []
    compound_noun = []  # 複合名詞用のバッファ

    for line in parsed.split('\n'):
        if line == 'EOS' or not line:
            # 文末または空行の場合、複合名詞をフラッシュ
            if compound_noun and use_compound:
                compound = ''.join(compound_noun)
                if len(compound) > 1 and compound not in stopwords:
                    nouns.append(compound)
            compound_noun = []
            continue

        # タブで分割（UniDic形式: 表層形\t読み\t読み\t原形\t品詞情報...）
        parts = line.split('\t')
        if len(parts) < 5:
            continue

        surface = parts[0]  # 表層形（単語）

        # UniDic形式では4番目（インデックス4）のカラムに品詞情報がある
        pos_info = parts[4] if len(parts) > 4 else ''
        pos_features = pos_info.split('-')  # UniDicではハイフン区切り

        # 品詞を取得
        pos = pos_features[0] if pos_features else ''
        pos_sub1 = pos_features[1] if len(pos_features) > 1 else ''
        pos_sub2 = pos_features[2] if len(pos_features) > 2 else ''
        
        # 名詞の判定
        if pos == '名詞':
            # 数詞、接尾辞、非自立語を除外
            if pos_sub1 in ['数', '非自立', '代名詞', '接尾']:
                # 複合名詞のバッファをフラッシュ
                if compound_noun and use_compound:
                    compound = ''.join(compound_noun)
                    if len(compound) > 1 and compound not in stopwords:
                        nouns.append(compound)
                compound_noun = []
                continue
            
            # UniDicでは「普通名詞」として分類される
            # 固有名詞、一般名詞、サ変接続を収集
            if pos_sub1 in ['固有名詞', '普通名詞', '一般', 'サ変接続', '形容動詞語幹'] or \
               (pos_sub1 == '普通名詞' and pos_sub2 in ['一般', 'サ変可能', '形状詞可能']):
                if use_compound:
                    # 複合名詞として連結
                    compound_noun.append(surface)
                else:
                    # 単名詞として追加
                    if surface not in stopwords and len(surface) > 1:
                        nouns.append(surface)
        else:
            # 名詞以外が来たら複合名詞をフラッシュ
            if compound_noun and use_compound:
                compound = ''.join(compound_noun)
                if len(compound) > 1 and compound not in stopwords:
                    nouns.append(compound)
            compound_noun = []
    
    # 最後の複合名詞をフラッシュ
    if compound_noun and use_compound:
        compound = ''.join(compound_noun)
        if len(compound) > 1 and compound not in stopwords:
            nouns.append(compound)
    
    # 英数字とカタカナ語も追加で抽出（MeCabが分割してしまう場合の補完）
    pattern = r'[A-Z][A-Za-z0-9]*|[ァ-ヴー]{3,}'
    additional_words = re.findall(pattern, text)
    nouns.extend(additional_words)
    
    # 単語頻度をカウント
    word_freq = collections.Counter(nouns)
    
    # 頻出上位のキーワードを返す
    keywords = [word for word, freq in word_freq.most_common(top_n)]
    
    # キーワードが少ない場合は補完
    if len(keywords) < 3:
        # ユニークな名詞から補完
        unique_nouns = [n for n in set(nouns) if n not in keywords]
        keywords.extend(unique_nouns[:3-len(keywords)])
    
    return keywords


def extract_keywords_regex(text: str, top_n: int = 5) -> List[str]:
    """
    正規表現を使用したキーワード抽出（フォールバック用）
    MeCabが利用できない場合に使用
    """
    # ストップワード
    stopwords = {
        'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん',
        'ます', 'です', 'ある', 'いる', 'する', 'なる', 'できる',
        'いう', '的', 'な', 'に', 'を', 'は', 'が', 'で', 'と', 'の', 'から', 'まで'
    }
    
    # カタカナ語、漢字複合語、英数字を抽出
    pattern = r'[ァ-ヴー]+|[一-龥]{2,}|[A-Za-z]+[A-Za-z0-9]*'
    words = re.findall(pattern, text)
    
    # 単語頻度をカウント（ストップワードを除外）
    word_freq = collections.Counter(
        word for word in words 
        if word not in stopwords and len(word) > 1
    )
    
    # 頻出上位のキーワードを返す
    keywords = [word for word, freq in word_freq.most_common(top_n)]
    
    # キーワードが少ない場合は補完
    if len(keywords) < 3:
        keywords.extend([w for w in words if w not in keywords][:3-len(keywords)])
    
    return keywords


def extract_keywords_with_score(text: str, top_n: int = 5) -> List[tuple]:
    """
    MeCabを使用してキーワードとスコアを返す版
    
    Args:
        text: 分析対象のテキスト
        top_n: 抽出するキーワード数
    
    Returns:
        (キーワード, スコア)のタプルのリスト
    """
    try:
        import MeCab
    except ImportError:
        return extract_keywords_regex_with_score(text, top_n)
    
    # MeCabの初期化
    try:
        tagger = MeCab.Tagger()
    except Exception:
        return extract_keywords_regex_with_score(text, top_n)
    
    parsed = tagger.parse(text)
    
    # ストップワード
    stopwords = {
        'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん',
        'ます', 'です', 'ある', 'いる', 'する', 'なる', 'できる',
        'いう', '的', 'な', 'に', 'を', 'は', 'が', 'で', 'と', 'の'
    }
    
    # 名詞と品詞情報を収集
    word_info = {}
    
    for line in parsed.split('\n'):
        if line == 'EOS' or not line:
            continue

        # UniDic形式の解析
        parts = line.split('\t')
        if len(parts) < 5:
            continue

        surface = parts[0]
        pos_info = parts[4] if len(parts) > 4 else ''
        pos_features = pos_info.split('-')

        pos = pos_features[0] if pos_features else ''
        pos_sub1 = pos_features[1] if len(pos_features) > 1 else ''

        if pos == '名詞' and surface not in stopwords:
            # 1文字の名詞も許可（ただし重要度を下げる）
            if len(surface) == 1:
                score = 0.5
            else:
                # スコアリング（品詞サブタイプに基づく重み付け）
                score = 1.0
                if pos_sub1 == '固有名詞':
                    score = 2.0  # 固有名詞は重要度高
                elif pos_sub1 == 'サ変接続':
                    score = 1.5  # サ変接続（動作性名詞）も重要
                elif pos_sub1 == '一般':
                    score = 1.0  # 一般名詞
                else:
                    score = 0.8  # その他
            
            if surface in word_info:
                word_info[surface] += score
            else:
                word_info[surface] = score
    
    # 英数字も追加で抽出（MeCabが解析できない場合の補完）
    import re
    pattern = r'[A-Z][A-Za-z0-9]*|[ァ-ヴー]{2,}'
    additional_words = re.findall(pattern, text)
    for word in additional_words:
        if word not in stopwords and word not in word_info:
            word_info[word] = 1.5  # 英語やカタカナは重要
    
    # スコアでソート
    sorted_words = sorted(word_info.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_words[:top_n]


def extract_keywords_regex_with_score(text: str, top_n: int = 5) -> List[tuple]:
    """
    正規表現版のスコア付きキーワード抽出
    """
    stopwords = {'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん'}
    
    # パターン別に抽出と重み付け
    patterns = [
        (r'[A-Z][A-Za-z0-9]*', 2.0),  # 英語の固有名詞（重要度高）
        (r'[ァ-ヴー]{3,}', 1.5),      # カタカナ語（専門用語の可能性）
        (r'[一-龥]{3,}', 1.0),        # 漢字複合語
    ]
    
    word_scores = {}
    
    for pattern, weight in patterns:
        matches = re.findall(pattern, text)
        for word in matches:
            if word not in stopwords and len(word) > 1:
                if word in word_scores:
                    word_scores[word] += weight
                else:
                    word_scores[word] = weight
    
    # スコアでソート
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_words[:top_n]


def analyze_mecab_pos_tags(text: str) -> None:
    """
    MeCabの品詞分類を詳細に分析・表示する関数

    Args:
        text: 分析対象のテキスト
    """
    try:
        import MeCab
    except ImportError:
        print("MeCabがインストールされていません")
        return

    try:
        tagger = MeCab.Tagger()
    except Exception as e:
        print(f"MeCab初期化エラー: {e}")
        return

    parsed = tagger.parse(text)

    print("\n" + "=" * 80)
    print("【MeCab品詞分析結果】")
    print("=" * 80)
    print(f"分析対象テキスト: {text[:50]}...")
    print("-" * 80)
    print(f"{'表層形':<15} {'品詞':<10} {'品詞細分類1':<12} {'品詞細分類2':<12} {'原形':<15}")
    print("-" * 80)

    # 品詞別の統計
    pos_stats = collections.defaultdict(lambda: collections.defaultdict(int))

    for line in parsed.split('\n'):
        if line == 'EOS' or not line:
            continue

        # UniDic形式の解析
        parts = line.split('\t')
        if len(parts) < 5:
            continue

        surface = parts[0]
        base_form = parts[3] if len(parts) > 3 else surface
        pos_info = parts[4] if len(parts) > 4 else ''
        pos_features = pos_info.split('-')

        pos = pos_features[0] if pos_features else ''
        pos_sub1 = pos_features[1] if len(pos_features) > 1 else '*'
        pos_sub2 = pos_features[2] if len(pos_features) > 2 else '*'

        # 統計を記録
        pos_stats[pos][pos_sub1] += 1

        # 名詞のみ詳細表示
        if pos == '名詞':
            print(f"{surface:<15} {pos:<10} {pos_sub1:<12} {pos_sub2:<12} {base_form:<15}")

    # 統計情報の表示
    print("\n" + "=" * 80)
    print("【品詞統計】")
    print("=" * 80)
    for pos, sub_types in sorted(pos_stats.items()):
        print(f"\n{pos}:")
        for sub_type, count in sorted(sub_types.items()):
            print(f"  - {sub_type}: {count}件")


def debug_compound_noun_extraction(text: str) -> None:
    """
    複合名詞抽出の動作を可視化するデバッグ関数

    Args:
        text: 分析対象のテキスト
    """
    try:
        import MeCab
    except ImportError:
        print("MeCabがインストールされていません")
        return

    try:
        tagger = MeCab.Tagger()
    except Exception as e:
        print(f"MeCab初期化エラー: {e}")
        return

    parsed = tagger.parse(text)

    stopwords = {
        'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん', 'ところ', 'とき',
        'ます', 'です', 'ある', 'いる', 'する', 'なる', 'できる'
    }

    print("\n" + "=" * 80)
    print("【複合名詞抽出プロセスの可視化】")
    print("=" * 80)
    print(f"分析対象: {text[:100]}...")
    print("-" * 80)

    compound_noun = []
    extracted_compounds = []
    step = 0

    for line in parsed.split('\n'):
        if line == 'EOS' or not line:
            if compound_noun:
                compound = ''.join(compound_noun)
                print(f"[Step {step}] 文末/空行 → 複合名詞バッファをフラッシュ: '{compound}'")
                if len(compound) > 1 and compound not in stopwords:
                    extracted_compounds.append(compound)
                    print(f"         ✓ 追加: '{compound}'")
                else:
                    print("         ✗ スキップ（短すぎるかストップワード）")
                compound_noun = []
            continue

        # UniDic形式の解析
        parts = line.split('\t')
        if len(parts) < 5:
            continue

        surface = parts[0]
        pos_info = parts[4] if len(parts) > 4 else ''
        pos_features = pos_info.split('-')

        pos = pos_features[0] if pos_features else ''
        pos_sub1 = pos_features[1] if len(pos_features) > 1 else ''
        pos_sub2 = pos_features[2] if len(pos_features) > 2 else ''

        step += 1

        if pos == '名詞':
            if pos_sub1 in ['数', '非自立', '代名詞', '接尾']:
                print(f"[Step {step}] '{surface}' (名詞-{pos_sub1}) → 除外対象")
                if compound_noun:
                    compound = ''.join(compound_noun)
                    print(f"         複合名詞バッファをフラッシュ: '{compound}'")
                    if len(compound) > 1 and compound not in stopwords:
                        extracted_compounds.append(compound)
                        print(f"         ✓ 追加: '{compound}'")
                    compound_noun = []
            elif pos_sub1 in ['固有名詞', '普通名詞', '一般', 'サ変接続', '形容動詞語幹']:
                print(f"[Step {step}] '{surface}' (名詞-{pos_sub1}-{pos_sub2}) → バッファに追加")
                compound_noun.append(surface)
                print(f"         現在のバッファ: {compound_noun}")
        else:
            print(f"[Step {step}] '{surface}' ({pos}) → 名詞以外")
            if compound_noun:
                compound = ''.join(compound_noun)
                print(f"         複合名詞バッファをフラッシュ: '{compound}'")
                if len(compound) > 1 and compound not in stopwords:
                    extracted_compounds.append(compound)
                    print(f"         ✓ 追加: '{compound}'")
                compound_noun = []

    # 最後のバッファ処理
    if compound_noun:
        compound = ''.join(compound_noun)
        print(f"[Step {step+1}] 最終バッファフラッシュ: '{compound}'")
        if len(compound) > 1 and compound not in stopwords:
            extracted_compounds.append(compound)
            print(f"         ✓ 追加: '{compound}'")

    print("\n" + "=" * 80)
    print("【抽出された複合名詞】")
    print("=" * 80)
    for i, compound in enumerate(extracted_compounds, 1):
        print(f"  {i}. {compound}")


def compare_extraction_methods(text: str) -> None:
    """
    異なる抽出方法の結果を比較する関数
    Args:
        text: 分析対象のテキスト
    """
    print("\n" + "=" * 80)
    print("【抽出方法の詳細比較】")
    print("=" * 80)

    # 複合名詞版
    print("\n◆ MeCab複合名詞版")
    keywords_compound = extract_keywords_mecab(text, top_n=10, use_compound=True)
    for i, kw in enumerate(keywords_compound, 1):
        print(f"  {i:2d}. {kw}")

    # 単名詞版
    print("\n◆ MeCab単名詞版")
    keywords_single = extract_keywords_mecab(text, top_n=10, use_compound=False)
    for i, kw in enumerate(keywords_single, 1):
        print(f"  {i:2d}. {kw}")

    # スコア付き版
    print("\n◆ スコア付き版")
    keywords_score = extract_keywords_with_score(text, top_n=10)
    for i, (kw, score) in enumerate(keywords_score, 1):
        print(f"  {i:2d}. {kw:<20} (スコア: {score:.2f})")

    # 正規表現版
    print("\n◆ 正規表現版")
    keywords_regex = extract_keywords_regex(text, top_n=10)
    for i, kw in enumerate(keywords_regex, 1):
        print(f"  {i:2d}. {kw}")

    # 共通キーワード
    print("\n" + "=" * 80)
    print("【方法間の共通キーワード】")
    print("=" * 80)

    set_compound = set(keywords_compound)
    set_single = set(keywords_single)
    set_score = set([kw for kw, _ in keywords_score])
    set_regex = set(keywords_regex)

    common_all = set_compound & set_single & set_score & set_regex
    if common_all:
        print("全手法で共通:", ", ".join(common_all))

    common_mecab = set_compound & set_single
    if common_mecab:
        print("MeCab版で共通:", ", ".join(common_mecab))

    unique_compound = set_compound - (set_single | set_score | set_regex)
    if unique_compound:
        print("複合名詞版のみ:", ", ".join(unique_compound))


# 使用例とテスト
if __name__ == "__main__":
    # テストテキスト
    test_text = """
    人工知能（AI）は、機械学習と深層学習を基盤として急速に発展しています。
    特に自然言語処理（NLP）の分野では、トランスフォーマーモデルが革命的な成果を上げました。
    BERTやGPTなどの大規模言語モデルは、文脈理解能力を大幅に向上させています。
    """
    
    print("=" * 60)
    print("MeCab版とRegex版の比較")
    print("=" * 60)
    
    # MeCab版（複合名詞あり）
    print("\n【MeCab版 - 複合名詞】")
    keywords_mecab_compound = extract_keywords_mecab(test_text, top_n=5, use_compound=True)
    for i, keyword in enumerate(keywords_mecab_compound, 1):
        print(f"  {i}. {keyword}")
    
    # MeCab版（単名詞のみ）
    print("\n【MeCab版 - 単名詞】")
    keywords_mecab_single = extract_keywords_mecab(test_text, top_n=5, use_compound=False)
    for i, keyword in enumerate(keywords_mecab_single, 1):
        print(f"  {i}. {keyword}")
    
    # 正規表現版
    print("\n【Regex版（フォールバック）】")
    keywords_regex = extract_keywords_regex(test_text, top_n=5)
    for i, keyword in enumerate(keywords_regex, 1):
        print(f"  {i}. {keyword}")
    
    # スコア付き版
    print("\n【スコア付き版】")
    keywords_with_score = extract_keywords_with_score(test_text, top_n=5)
    for i, (keyword, score) in enumerate(keywords_with_score, 1):
        print(f"  {i}. {keyword} (スコア: {score:.2f})")

    print("\n" + "=" * 60)
    print("調査用関数のテスト")
    print("=" * 60)

    # 品詞分析
    analyze_mecab_pos_tags(test_text)

    # 複合名詞抽出プロセスの可視化
    debug_compound_noun_extraction(test_text)

    # 抽出方法の詳細比較
    compare_extraction_methods(test_text)
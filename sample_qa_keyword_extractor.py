"""
Q/Aペア作成のための最適化されたキーワード抽出モジュール
MeCab複合名詞版 + 改善版
"""
import collections
from typing import List, Dict, Tuple, Optional
import re


class QAKeywordExtractor:
    """Q/Aペア作成に特化したキーワード抽出クラス"""

    def __init__(self, domain_stopwords: Optional[List[str]] = None):
        """
        Args:
            domain_stopwords: ドメイン特化のストップワード（オプション）
        """
        # 基本ストップワード
        self.basic_stopwords = {
            'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん', 'ところ', 'とき',
            'ます', 'です', 'ある', 'いる', 'する', 'なる', 'できる', 'やる', 'くる',
            'いう', '的', 'な', 'に', 'を', 'は', 'が', 'で', 'と', 'の', 'から', 'まで',
            'へ', 'より', 'など', 'ら', 'たち', 'ない', 'れる', 'られる', 'せる', 'させる',
            '方法', '場合', 'とき', 'ところ', 'わけ', 'はず', 'つもり'
        }

        # ドメイン特化のストップワードを追加
        self.stopwords = self.basic_stopwords.copy()
        if domain_stopwords:
            self.stopwords.update(domain_stopwords)

        # 接頭辞リスト（複合名詞の先頭を補完）
        self.prefixes = {
            '大': '大',
            '小': '小',
            '新': '新',
            '旧': '旧',
            '全': '全',
            '各': '各',
            '主': '主',
            '副': '副',
            '総': '総',
            '高': '高',
            '低': '低',
            '最': '最'
        }

        # MeCabの初期化
        self.tagger = self._init_mecab()

    def _init_mecab(self):
        """MeCabの初期化"""
        try:
            import MeCab
            return MeCab.Tagger()
        except ImportError:
            print("警告: MeCabがインストールされていません")
            print("インストール: pip install mecab-python3 unidic-lite")
            return None

    def extract_qa_keywords(
        self,
        text: str,
        top_n: int = 10,
        min_length: int = 2,
        include_scores: bool = False
    ) -> List:
        """
        Q/Aペア作成に最適化されたキーワード抽出

        Args:
            text: 分析対象のテキスト
            top_n: 抽出するキーワード数
            min_length: 最小文字数（デフォルト2文字以上）
            include_scores: スコアを含めるか

        Returns:
            キーワードリスト、またはスコア付きタプルのリスト
        """
        if not self.tagger:
            return self._extract_fallback(text, top_n, min_length)

        # 形態素解析
        parsed = self.tagger.parse(text)

        # 複合名詞の抽出
        compound_nouns = self._extract_compound_nouns(parsed, min_length, text)

        # 頻度カウントとスコアリング
        if include_scores:
            return self._score_keywords(compound_nouns, text, top_n)
        else:
            word_freq = collections.Counter(compound_nouns)
            return [word for word, freq in word_freq.most_common(top_n)]

    def _extract_compound_nouns(self, parsed: str, min_length: int, original_text: str = "") -> List[str]:
        """
        複合名詞の抽出（改善版）

        Args:
            parsed: MeCabの解析結果
            min_length: 最小文字数
            original_text: 元のテキスト（英数字抽出用）

        Returns:
            複合名詞のリスト
        """
        compound_nouns = []
        compound_buffer = []
        last_prefix = None  # 直前の接頭辞を保持

        for line in parsed.split('\n'):
            if line == 'EOS' or not line:
                # バッファに複合名詞があればフラッシュ
                if compound_buffer:
                    compound = ''.join(compound_buffer)
                    if len(compound) >= min_length and compound not in self.stopwords:
                        compound_nouns.append(compound)
                compound_buffer = []
                last_prefix = None
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

            # 接頭辞の処理
            if pos == '接頭辞' or surface in self.prefixes:
                last_prefix = surface
                continue

            # 名詞の判定
            if pos == '名詞':
                # 除外する名詞タイプ
                if pos_sub1 in ['数', '非自立', '代名詞', '接尾']:
                    # バッファをフラッシュ
                    if compound_buffer:
                        compound = ''.join(compound_buffer)
                        if len(compound) >= min_length and compound not in self.stopwords:
                            compound_nouns.append(compound)
                    compound_buffer = []
                    last_prefix = None
                    continue

                # 収集する名詞タイプ
                if pos_sub1 in ['固有名詞', '普通名詞'] or \
                   (pos_sub1 == '普通名詞' and pos_sub2 in ['一般', 'サ変可能', '形状詞可能']):
                    # 接頭辞があれば先に追加
                    if last_prefix and not compound_buffer:
                        compound_buffer.append(last_prefix)
                    # 名詞を追加
                    compound_buffer.append(surface)
                    last_prefix = None
            else:
                # 名詞以外が来たらバッファをフラッシュ
                if compound_buffer:
                    compound = ''.join(compound_buffer)
                    if len(compound) >= min_length and compound not in self.stopwords:
                        compound_nouns.append(compound)
                compound_buffer = []
                last_prefix = None

        # 最後のバッファ処理
        if compound_buffer:
            compound = ''.join(compound_buffer)
            if len(compound) >= min_length and compound not in self.stopwords:
                compound_nouns.append(compound)

        # 英数字とカタカナ語も追加（MeCabが分割してしまう場合の補完）
        pattern = r'[A-Z][A-Za-z0-9]*|[ァ-ヴー]{3,}'
        additional_words = re.findall(pattern, original_text)
        for word in additional_words:
            if len(word) >= min_length and word not in self.stopwords:
                compound_nouns.append(word)

        return compound_nouns

    def _score_keywords(self, keywords: List[str], text: str, top_n: int) -> List[Tuple[str, float]]:
        """
        キーワードのスコアリング

        Args:
            keywords: キーワードリスト
            text: 元のテキスト
            top_n: 上位N件

        Returns:
            (キーワード, スコア)のタプルリスト
        """
        keyword_scores = {}

        for keyword in keywords:
            # 基本スコア（出現頻度）
            base_score = keywords.count(keyword)

            # 文字数によるボーナス
            length_bonus = min(len(keyword) / 10, 1.0)

            # 位置によるボーナス（文頭に近いほど高スコア）
            position_bonus = 1.0
            first_pos = text.find(keyword)
            if first_pos != -1:
                position_bonus = 1.0 - (first_pos / len(text)) * 0.3

            # カタカナ・英字によるボーナス（専門用語の可能性）
            special_bonus = 0
            if re.search(r'[A-Za-z]', keyword):
                special_bonus += 0.5
            if re.search(r'[ァ-ヴー]', keyword):
                special_bonus += 0.3

            # 総合スコア
            total_score = base_score * (1 + length_bonus + position_bonus + special_bonus)

            if keyword in keyword_scores:
                keyword_scores[keyword] = max(keyword_scores[keyword], total_score)
            else:
                keyword_scores[keyword] = total_score

        # スコアでソート
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_n]

    def _extract_fallback(self, text: str, top_n: int, min_length: int) -> List[str]:
        """
        MeCabが使えない場合のフォールバック処理
        """
        # カタカナ語、漢字複合語、英数字を抽出
        pattern = r'[ァ-ヴー]+|[一-龥]{2,}|[A-Za-z]+[A-Za-z0-9]*'
        words = re.findall(pattern, text)

        # フィルタリング
        filtered = [
            word for word in words
            if len(word) >= min_length and word not in self.stopwords
        ]

        # 頻度カウント
        word_freq = collections.Counter(filtered)
        return [word for word, freq in word_freq.most_common(top_n)]

    def generate_qa_examples(self, keywords: List[str]) -> List[Dict[str, str]]:
        """
        抽出されたキーワードからQ/Aペアの例を生成

        Args:
            keywords: キーワードリスト

        Returns:
            Q/Aペアの例のリスト
        """
        qa_templates = [
            {"q": "{keyword}とは何ですか？", "type": "definition"},
            {"q": "{keyword}の特徴を教えてください", "type": "feature"},
            {"q": "{keyword}のメリットは何ですか？", "type": "advantage"},
            {"q": "{keyword}の使い方を説明してください", "type": "usage"},
            {"q": "{keyword}について詳しく教えてください", "type": "detail"}
        ]

        qa_pairs = []
        for keyword in keywords:
            # キーワードの種類を推定
            if re.search(r'[A-Z]', keyword):  # 英語/略語
                templates = [qa_templates[0], qa_templates[1], qa_templates[4]]
            elif '学習' in keyword or '処理' in keyword:  # 技術用語
                templates = [qa_templates[0], qa_templates[1], qa_templates[3]]
            else:  # 一般用語
                templates = qa_templates[:3]

            for template in templates[:2]:  # 各キーワードにつき2つまで
                qa_pairs.append({
                    "keyword": keyword,
                    "question": template["q"].format(keyword=keyword),
                    "type": template["type"]
                })

        return qa_pairs


# 使用例とテスト
if __name__ == "__main__":
    # テストテキスト
    test_text = """
    人工知能（AI）は、機械学習と深層学習を基盤として急速に発展しています。
    特に自然言語処理（NLP）の分野では、トランスフォーマーモデルが革命的な成果を上げました。
    BERTやGPTなどの大規模言語モデルは、文脈理解能力を大幅に向上させています。
    最新の研究では、小規模データセットでも高性能を実現する手法が開発されています。
    """

    # ドメイン特化のストップワード例（AI分野）
    domain_stopwords = ['データ', 'システム', '技術', '方式']

    # 抽出器の初期化
    extractor = QAKeywordExtractor(domain_stopwords=domain_stopwords)

    print("=" * 80)
    print("Q/Aペア作成用キーワード抽出（最適化版）")
    print("=" * 80)

    # キーワード抽出
    keywords = extractor.extract_qa_keywords(test_text, top_n=10)
    print("\n【抽出されたキーワード】")
    for i, keyword in enumerate(keywords, 1):
        print(f"  {i:2d}. {keyword}")

    # スコア付きキーワード
    keywords_with_scores = extractor.extract_qa_keywords(
        test_text, top_n=10, include_scores=True
    )
    print("\n【スコア付きキーワード】")
    print("  ※スコア = 出現頻度 × (1 + 文字数/10 + 位置評価 + 専門用語度)")
    print("  ※評価: 5.0以上=最重要、3.0-5.0=重要、2.0-3.0=標準")
    print("-" * 80)

    for i, (keyword, score) in enumerate(keywords_with_scores, 1):
        # スコア評価
        if score >= 5.0:
            evaluation = "⭐⭐⭐"
        elif score >= 3.0:
            evaluation = "⭐⭐"
        elif score >= 2.0:
            evaluation = "⭐"
        else:
            evaluation = ""

        print(f"  {i:2d}. {keyword:<20} (スコア: {score:5.2f}) {evaluation}")

    # Q/Aペア例の生成
    print("\n" + "=" * 80)
    print("Q/Aペア生成")
    print("=" * 80)

    qa_examples = extractor.generate_qa_examples(keywords[:5])
    print("\n【生成されたQ/Aペア例】")
    print("-" * 80)

    # キーワードごとにグループ化して表示
    current_keyword = ""
    for i, qa in enumerate(qa_examples, 1):
        if current_keyword != qa['keyword']:
            current_keyword = qa['keyword']
            print(f"\n◆ キーワード: 「{current_keyword}」")
            print("-" * 40)

        print(f"  Q{i:2d}: {qa['question']}")
        print(f"       (タイプ: {qa['type']})")

    # Q/Aペア統計
    print("\n" + "-" * 80)
    print("【Q/Aペア生成統計】")
    print(f"  - 使用キーワード数: {len(set([qa['keyword'] for qa in qa_examples]))} 個")
    print(f"  - 生成された質問数: {len(qa_examples)} 個")
    print(f"  - 質問タイプ: {', '.join(set([qa['type'] for qa in qa_examples]))}")
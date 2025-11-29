"""MeCabの出力形式を確認するテストスクリプト"""
import MeCab

# 複数の出力形式で確認
test_text = "人工知能は機械学習の分野です。"

print("=" * 60)
print("デフォルト形式:")
print("=" * 60)
tagger = MeCab.Tagger()
result = tagger.parse(test_text)
print(result)
print("\n各行の詳細:")
for line in result.split('\n'):
    if line and line != 'EOS':
        print(f"行: '{line}'")
        parts = line.split('\t')
        print(f"  タブ区切り数: {len(parts)}")
        if len(parts) >= 2:
            print(f"  表層形: '{parts[0]}'")
            print(f"  残り: '{parts[1]}'")

print("\n" + "=" * 60)
print("Chasen形式:")
print("=" * 60)
tagger_chasen = MeCab.Tagger("-Ochasen")
result_chasen = tagger_chasen.parse(test_text)
print(result_chasen)

print("\n" + "=" * 60)
print("Wakati形式:")
print("=" * 60)
tagger_wakati = MeCab.Tagger("-Owakati")
result_wakati = tagger_wakati.parse(test_text)
print(result_wakati)

print("\n" + "=" * 60)
print("詳細な形態素情報の取得:")
print("=" * 60)
tagger = MeCab.Tagger()
node = tagger.parseToNode(test_text)
while node:
    if node.surface:
        print(f"表層形: {node.surface}")
        print(f"  品詞情報: {node.feature}")
        features = node.feature.split(',')
        print(f"  品詞: {features[0] if len(features) > 0 else '不明'}")
        print(f"  品詞細分類1: {features[1] if len(features) > 1 else '不明'}")
        print(f"  品詞細分類2: {features[2] if len(features) > 2 else '不明'}")
        print()
    node = node.next
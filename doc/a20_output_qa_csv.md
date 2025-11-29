# a20_output_qa_csv.py - 技術仕様書

## 最新バージョン情報
- **最終更新**: 2025-11-12
- **バージョン**: v1.1
- **主要機能**: Q/Aペア抽出、統一フォーマット変換、複数データセット対応

---

## 概要

`a20_output_qa_csv.py`は、各Q&A生成プログラム（a02, a03, a10）の最新出力CSVファイルから`question`と`answer`の列のみを抽出し、統一フォーマットのCSVファイルを作成するユーティリティスクリプトです。

## 対応データセット

以下の2種類のデータセットに対応しています:

1. **cc_news**: CC-News英語ニュース
2. **livedoor**: Livedoorニュースコーパス（日本語）

**主な用途**:
- 異なるプログラムの出力を統一フォーマットに変換
- 不要なメタデータカラムの除去
- 最新のQ/Aペアファイルの自動選択と抽出
- 複数データセット（cc_news、livedoor）対応

---

## 目次

1. [概要](#概要)
2. [主要機能](#主要機能)
3. [処理対象ファイル](#処理対象ファイル)
4. [関数仕様](#関数仕様)
5. [使用方法](#使用方法)
6. [出力ファイル](#出力ファイル)
7. [エラーハンドリング](#エラーハンドリング)
8. [使用例](#使用例)

---

## 主要機能

### 1. 最新ファイルの自動選択

各プログラムのディレクトリから最新のCSVファイルを自動的に選択：
- 更新時刻でソート（L85）
- 最新のファイルを処理対象に選定

### 2. Q/Aペアの抽出

元のCSVファイルから`question`と`answer`カラムのみを抽出：
- メタデータカラム（doc_id, title, text_lengthなど）を除去
- 統一された2カラム構成に変換

### 3. バッチ処理

複数のプログラム出力を一括処理：
- a02, a03, a10の出力を順次処理
- エラーが発生しても他のファイルの処理を継続

---

## 処理対象ファイル

### 入力ファイルパターン（L99-124）

| プログラム | データセット | 入力パターン | 説明 |
|----------|----------|------------|------|
| a02 | cc_news | `qa_output/a02/qa_pairs_cc_news_*.csv` | LLMバッチ処理版（英語） |
| a02 | livedoor | `qa_output/a02/qa_pairs_livedoor_*.csv` | LLMバッチ処理版（日本語） |
| a03 | cc_news | `qa_output/a03/qa_pairs_cc_news_*.csv` | ルールベース改良版（英語） |
| a03 | livedoor | `qa_output/a03/qa_pairs_livedoor_*.csv` | ルールベース改良版（日本語） |
| a10 | cc_news | `qa_output/a10/batch_qa_pairs_cc_news_gpt_5_mini_b*_*.csv` | ハイブリッドバッチ処理版（英語） |
| a10 | livedoor | `qa_output/a10/batch_qa_pairs_livedoor_gpt_5_mini_b*_*.csv` | ハイブリッドバッチ処理版（日本語） |

### 出力ファイル（L99-124）

| 出力ファイル | 説明 |
|------------|------|
| `qa_output/a02_qa_pairs_cc_news.csv` | a02の最新Q/Aペア（CC-News） |
| `qa_output/a02_qa_pairs_livedoor.csv` | a02の最新Q/Aペア（Livedoor） |
| `qa_output/a03_qa_pairs_cc_news.csv` | a03の最新Q/Aペア（CC-News） |
| `qa_output/a03_qa_pairs_livedoor.csv` | a03の最新Q/Aペア（Livedoor） |
| `qa_output/a10_qa_pairs_cc_news.csv` | a10の最新Q/Aペア（CC-News） |
| `qa_output/a10_qa_pairs_livedoor.csv` | a10の最新Q/Aペア（Livedoor） |

---

## 関数仕様

### 1. extract_qa_pairs()（L24-68）

**目的**: CSVファイルからquestionとanswerの列を抽出して新しいCSVファイルを作成

```python
def extract_qa_pairs(input_file: str, output_file: str) -> None:
    """
    Args:
        input_file: 入力CSVファイルのパス
        output_file: 出力CSVファイルのパス
    """
```

**処理フロー**:
1. **入力ファイル存在確認**（L33-35）:
   ```python
   if not os.path.exists(input_file):
       print(f"エラー: 入力ファイルが見つかりません: {input_file}")
       return
   ```

2. **出力ディレクトリ作成**（L38-40）:
   ```python
   output_dir = os.path.dirname(output_file)
   if output_dir and not os.path.exists(output_dir):
       os.makedirs(output_dir)
   ```

3. **ヘッダー確認**（L47-50）:
   ```python
   if 'question' not in reader.fieldnames or 'answer' not in reader.fieldnames:
       print(f"エラー: {input_file} にquestionまたはanswerカラムが見つかりません")
       print(f"利用可能なカラム: {reader.fieldnames}")
       return
   ```

4. **データ抽出**（L53-63）:
   ```python
   with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
       writer = csv.DictWriter(f_out, fieldnames=['question', 'answer'])
       writer.writeheader()

       row_count = 0
       for row in reader:
           writer.writerow({
               'question': row['question'],
               'answer': row['answer']
           })
           row_count += 1
   ```

5. **完了メッセージ**（L65）:
   ```python
   print(f"✓ {input_file} から {row_count} 件のQ&Aペアを抽出 → {output_file}")
   ```

**エラーハンドリング**（L67-68）:
```python
except Exception as e:
    print(f"エラー: {input_file} の処理中にエラーが発生しました: {e}")
```

### 2. get_latest_file()（L71-85）

**目的**: 指定されたパターンにマッチする最新のファイルを取得

```python
def get_latest_file(pattern: str) -> str | None:
    """
    Args:
        pattern: ファイルパターン (例: 'qa_output/a02/qa_pairs_cc_news_*.csv')

    Returns:
        最新のファイルパス。見つからない場合はNone
    """
```

**処理内容**:
1. **パターンマッチング**（L81）:
   ```python
   files = glob.glob(pattern)
   ```

2. **ファイル存在確認**（L82-83）:
   ```python
   if not files:
       return None
   ```

3. **最新ファイル選択**（L85）:
   ```python
   return max(files, key=os.path.getmtime)  # 更新時刻でソート
   ```

### 3. main()（L88-128）

**目的**: 各プログラムの最新出力ファイルからquestion/answerを抽出

**処理フロー**:
1. **処理対象定義**（L93-106）:
   ```python
   file_patterns = [
       {
           'pattern': 'qa_output/a02/qa_pairs_cc_news_*.csv',
           'output': 'qa_output/a02_qa_pairs_cc_news.csv'
       },
       # ... 他のパターン
   ]
   ```

2. **開始メッセージ**（L108-110）:
   ```python
   print("=" * 60)
   print("Q&Aペア抽出処理を開始します")
   print("=" * 60)
   ```

3. **バッチ処理ループ**（L112-124）:
   ```python
   for mapping in file_patterns:
       pattern = mapping['pattern']
       output_file = mapping['output']

       # 最新のファイルを取得
       latest_file = get_latest_file(pattern)

       if latest_file is None:
           print(f"警告: パターン '{pattern}' にマッチするファイルが見つかりません")
           continue

       print(f"最新ファイル: {latest_file}")
       extract_qa_pairs(latest_file, output_file)
   ```

4. **完了メッセージ**（L126-128）:
   ```python
   print("=" * 60)
   print("処理が完了しました")
   print("=" * 60)
   ```

---

## 使用方法

### 基本実行

```bash
python a20_output_qa_csv.py
```

### 実行例

```bash
$ python a20_output_qa_csv.py
============================================================
Q&Aペア抽出処理を開始します
============================================================
最新ファイル: qa_output/a02/qa_pairs_cc_news_gpt_5_mini_20241029_141030.csv
✓ qa_output/a02/qa_pairs_cc_news_gpt_5_mini_20241029_141030.csv から 525 件のQ&Aペアを抽出 → qa_output/a02_qa_pairs_cc_news.csv
最新ファイル: qa_output/a03/qa_pairs_cc_news_20241029_141530.csv
✓ qa_output/a03/qa_pairs_cc_news_20241029_141530.csv から 7308 件のQ&Aペアを抽出 → qa_output/a03_qa_pairs_cc_news.csv
最新ファイル: qa_output/a10/batch_qa_pairs_cc_news_gpt_5_mini_b25_20241029_142000.csv
✓ qa_output/a10/batch_qa_pairs_cc_news_gpt_5_mini_b25_20241029_142000.csv から 1491 件のQ&Aペアを抽出 → qa_output/a10_qa_pairs_cc_news.csv
============================================================
処理が完了しました
============================================================
```

---

## 出力ファイル

### ファイル構成

```
qa_output/
├── a02_qa_pairs_cc_news.csv      # a02の最新Q/Aペア（CC-News）
├── a02_qa_pairs_livedoor.csv     # a02の最新Q/Aペア（Livedoor）
├── a03_qa_pairs_cc_news.csv      # a03の最新Q/Aペア（CC-News）
├── a03_qa_pairs_livedoor.csv     # a03の最新Q/Aペア（Livedoor）
├── a10_qa_pairs_cc_news.csv      # a10の最新Q/Aペア（CC-News）
└── a10_qa_pairs_livedoor.csv     # a10の最新Q/Aペア（Livedoor）
```

### 出力フォーマット

すべての出力ファイルは以下の統一フォーマット：

```csv
question,answer
"What is the main topic of passage 1?","This passage discusses the latest developments in artificial intelligence..."
"パッセージ2において、「機械学習」について何が述べられていますか？","機械学習は人工知能の一分野であり..."
```

**カラム構成**:
| カラム | 説明 |
|-------|------|
| question | 質問文 |
| answer | 回答文 |

### 元のファイル形式との比較

**元のファイル（a10の例）**:
```csv
doc_id,question,answer,doc_title,text_length
cc_news_0,"What is...","This...",CNN Breaking News,1250
cc_news_1,"How does...","It...",BBC World News,980
```

**変換後のファイル**:
```csv
question,answer
"What is...","This..."
"How does...","It..."
```

---

## エラーハンドリング

### 1. ファイルが見つからない場合（L119-121）

```python
if latest_file is None:
    print(f"警告: パターン '{pattern}' にマッチするファイルが見つかりません")
    continue  # 次のパターンへ
```

**動作**: 警告を表示して次のファイルパターンの処理を続行

### 2. 必須カラムが存在しない場合（L47-50）

```python
if 'question' not in reader.fieldnames or 'answer' not in reader.fieldnames:
    print(f"エラー: {input_file} にquestionまたはanswerカラムが見つかりません")
    print(f"利用可能なカラム: {reader.fieldnames}")
    return
```

**動作**: エラーメッセージを表示して処理を中断

### 3. 読み込みエラー（L67-68）

```python
except Exception as e:
    print(f"エラー: {input_file} の処理中にエラーが発生しました: {e}")
```

**動作**: エラー内容を表示してプログラムは継続

---

## 使用例

### ケース1: すべてのプログラムが実行済み

```bash
$ python a20_output_qa_csv.py
============================================================
Q&Aペア抽出処理を開始します
============================================================
最新ファイル: qa_output/a02/qa_pairs_cc_news_gpt_5_mini_20241029_141030.csv
✓ qa_output/a02/qa_pairs_cc_news_gpt_5_mini_20241029_141030.csv から 525 件のQ&Aペアを抽出 → qa_output/a02_qa_pairs_cc_news.csv
最新ファイル: qa_output/a03/qa_pairs_cc_news_20241029_141530.csv
✓ qa_output/a03/qa_pairs_cc_news_20241029_141530.csv から 7308 件のQ&Aペアを抽出 → qa_output/a03_qa_pairs_cc_news.csv
最新ファイル: qa_output/a10/batch_qa_pairs_cc_news_gpt_5_mini_b25_20241029_142000.csv
✓ qa_output/a10/batch_qa_pairs_cc_news_gpt_5_mini_b25_20241029_142000.csv から 1491 件のQ&Aペアを抽出 → qa_output/a10_qa_pairs_cc_news.csv
============================================================
処理が完了しました
============================================================
```

### ケース2: 一部のプログラムのみ実行済み

```bash
$ python a20_output_qa_csv.py
============================================================
Q&Aペア抽出処理を開始します
============================================================
最新ファイル: qa_output/a02/qa_pairs_cc_news_gpt_5_mini_20241029_141030.csv
✓ qa_output/a02/qa_pairs_cc_news_gpt_5_mini_20241029_141030.csv から 525 件のQ&Aペアを抽出 → qa_output/a02_qa_pairs_cc_news.csv
警告: パターン 'qa_output/a03/qa_pairs_cc_news_*.csv' にマッチするファイルが見つかりません
警告: パターン 'qa_output/a10/batch_qa_pairs_cc_news_gpt_5_mini_b25_*.csv' にマッチするファイルが見つかりません
============================================================
処理が完了しました
============================================================
```

### ケース3: カラム名が不正なファイル

```bash
$ python a20_output_qa_csv.py
============================================================
Q&Aペア抽出処理を開始します
============================================================
最新ファイル: qa_output/a02/qa_pairs_cc_news_invalid.csv
エラー: qa_output/a02/qa_pairs_cc_news_invalid.csv にquestionまたはanswerカラムが見つかりません
利用可能なカラム: ['q', 'a', 'doc_id']
...
```

---

## プログラム間の連携

### 実行順序

```
1. データ前処理
   ↓
2. Q/A生成（a02, a03, a10のいずれか）
   ↓
3. a20_output_qa_csv.py（統一フォーマット変換）← ここ
   ↓
4. 後続処理（比較分析、評価など）
```

### 典型的なワークフロー

```bash
# Step 1: データ前処理
python a01_load_non_qa_rag_data.py

# Step 2: Q/A生成（複数実行可能）
python a02_make_qa.py --dataset cc_news --analyze-coverage
python a03_rag_qa_coverage_improved.py --input OUTPUT/preprocessed_cc_news.csv --dataset cc_news --analyze-coverage
python a10_qa_optimized_hybrid_batch.py --dataset cc_news --batch-size 25

# Step 3: 統一フォーマット変換
python a20_output_qa_csv.py

# Step 4: 結果の確認
ls -lh qa_output/a*_qa_pairs_cc_news.csv
```

---

## 技術的詳細

### ファイル選択ロジック（L85）

```python
return max(files, key=os.path.getmtime)
```

- `os.path.getmtime()`: ファイルの最終更新時刻を取得
- `max()`: 更新時刻が最も新しいファイルを選択

### CSV処理の安全性

- **エンコーディング**: UTF-8を明示的に指定（L43, L53）
- **改行処理**: `newline=''`でクロスプラットフォーム対応（L53）
- **ヘッダー検証**: 必須カラムの存在を確認してからデータ抽出（L47-50）

### エラー耐性

- ファイルが見つからない場合も処理を継続（L119-121）
- 各ファイルの処理エラーは独立（try-except内で処理）（L42-68）

---

## ベストプラクティス

### 実行タイミング

- **推奨**: Q/A生成プログラムの実行直後
- 複数のプログラムを実行した後にまとめて実行も可能

### 出力ファイルの確認

```bash
# 各ファイルの行数を確認
wc -l qa_output/a*_qa_pairs_cc_news.csv

# 各ファイルの内容をプレビュー
head -5 qa_output/a02_qa_pairs_cc_news.csv
head -5 qa_output/a03_qa_pairs_cc_news.csv
head -5 qa_output/a10_qa_pairs_cc_news.csv
```

### トラブルシューティング

1. **ファイルが生成されない**
   - 該当するプログラム（a02/a03/a10）を先に実行する
   - `qa_output/a02/`, `qa_output/a03/`, `qa_output/a10/`ディレクトリの存在を確認

2. **カラムエラー**
   - 元のCSVファイルに`question`と`answer`カラムが存在するか確認
   - ファイルが破損していないか確認

3. **エンコーディングエラー**
   - すべてのCSVファイルがUTF-8エンコーディングであることを確認

---

## 今後の拡張案

1. **データセット対応の拡張**
   - japanese_text、wikipedia_jaにも対応
   - コマンドライン引数でデータセット指定

2. **統計情報の出力**
   - 各ファイルのQ/A数
   - 平均質問長・回答長

3. **フィルタリング機能**
   - 空の質問・回答の除外
   - 重複Q/Aペアの除去

4. **バックアップ機能**
   - 既存の出力ファイルを上書きする前にバックアップ

---

## 変更履歴

### v1.1 (2025-11-12)
- **Livedoorデータセット対応追加**
- 合計6つの出力ファイルに対応（cc_news × 3プログラム + livedoor × 3プログラム）
- ドキュメント更新

### v1.0 (2024-10-29)
- 初版リリース
- a02, a03, a10の出力に対応
- 最新ファイル自動選択機能
- 統一フォーマット変換機能

---

**最終更新日**: 2025年11月12日
**バージョン**: 1.1
**作成者**: OpenAI RAG Q&A JP開発チーム
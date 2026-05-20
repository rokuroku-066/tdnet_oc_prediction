# 仕様書

## 1. 概要 🌟

### 1.1 モジュール名

```text
tdnet_oc_prediction
```

### 1.2 目的

本モジュールは、適時開示テキストを入力として、翌営業日の終値が始値を上回るかどうかを予測する2値分類モデルを構築・評価・シミュレーションするための研究用基盤である。

### 1.3 対象フェーズ

本モジュールは以下の4フェーズを対象とする。

| フェーズ       | 目的                         |
| ---------- | -------------------------- |
| データ収集      | 適時開示テキストと株価OHLCデータを取得・整形する |
| モデル構築      | テキスト分類モデルを学習する             |
| 評価         | 分類性能を評価する                  |
| 運用シミュレーション | 予測結果を用いた仮想売買を検証する          |

---

# 2. タスク定義 🎯

## 2.1 入力

```text
適時開示のタイトル + 本文テキスト
```

同一銘柄が同一日に複数の適時開示を出した場合は、テキストを結合して1サンプルにする。

```text
1銘柄 × 1開示日 = 1サンプル
```

## 2.2 出力ラベル

開示日の翌営業日を `target_date` とする。

```text
y = 1 if target_close > target_open
y = 0 if target_close < target_open
```

`target_close == target_open` のサンプルは除外する。

## 2.3 予測対象

モデルは以下を出力する。

```text
p_up = P(target_close > target_open)
```

分類ラベルは原則として以下で決める。

```text
pred_label = 1 if p_up >= threshold else 0
```

デフォルトの `threshold` は `0.5`。

---

# 3. 全体アーキテクチャ 🧱

## 3.1 ディレクトリ構成

```text
tdnet_oc_prediction/
  spec.md
  README.md
  pyproject.toml
  configs/
    default.yaml
    tfidf_logreg.yaml
    transformer.yaml
    simulation.yaml

  data/
    raw/
      disclosures/
      prices/
    interim/
    processed/
    splits/
    predictions/
    simulations/

  models/
    baselines/
    transformers/
    checkpoints/

  reports/
    metrics/
    figures/
    simulation/

  src/
    tdnet_oc_prediction/
      __init__.py

      config/
        schema.py
        loader.py

      data/
        disclosure_client.py
        price_client.py
        text_extractor.py
        dataset_builder.py
        calendar.py
        validators.py

      features/
        text_preprocessor.py
        tfidf_vectorizer.py
        tokenizer.py

      models/
        base.py
        majority.py
        tfidf_logreg.py
        transformer_classifier.py
        registry.py

      training/
        trainer.py
        losses.py
        callbacks.py

      evaluation/
        metrics.py
        evaluator.py
        calibration.py

      simulation/
        signal.py
        backtester.py
        cost.py
        risk.py

      utils/
        logging.py
        seed.py
        io.py
        time.py

  scripts/
    collect_data.py
    build_dataset.py
    train.py
    evaluate.py
    simulate.py
    run_all.py

  tests/
    test_dataset_builder.py
    test_labeling.py
    test_metrics.py
    test_simulation.py
```

---

# 4. 共通仕様 ⚙️

## 4.1 実行環境

```text
Language: Python
Recommended Python version: 3.10+
```

主要ライブラリ候補：

```text
pandas
numpy
scikit-learn
torch
transformers
datasets
lightgbm optional
pyyaml
pydantic
matplotlib
tqdm
joblib
```

## 4.2 設定ファイル

全フェーズは YAML 設定ファイルで制御する。

例：

```yaml
project:
  name: tdnet_oc_prediction
  seed: 42
  timezone: Asia/Tokyo

data:
  start_date: "2020-04-01"
  end_date: "2025-04-30"
  disclosure_source: "local_or_api"
  price_source: "local_or_api"
  text_scope: "title_body"
  aggregate_unit: "stock_date"

label:
  target: "next_business_day_open_close"
  positive_condition: "close_gt_open"
  drop_equal_open_close: true

split:
  method: "time_series"
  train_end: "2022-12-31"
  valid_start: "2023-01-01"
  valid_end: "2023-12-31"
  test_start: "2024-01-01"
  test_end: "2025-04-30"

model:
  name: "tfidf_logreg"
  threshold: 0.5

evaluation:
  metrics:
    - accuracy
    - balanced_accuracy
    - roc_auc
    - log_loss

simulation:
  enabled: true
  threshold_long: 0.55
  threshold_short: 0.45
  allow_short: true
  transaction_cost_bps: 10
```

---

# 5. フェーズ1：データ収集 📥

## 5.1 目的

適時開示データと株価データを取得し、モデル学習に使える形式へ整形する。

---

## 5.2 入力データ

### 5.2.1 適時開示データ

最低限必要なカラム：

| カラム名            |    型 | 内容               |
| --------------- | ---: | ---------------- |
| disclosure_id   |  str | 開示ID             |
| stock_code      |  str | 銘柄コード            |
| company_name    |  str | 会社名              |
| disclosure_date | date | 開示日              |
| disclosure_time |  str | 開示時刻             |
| title           |  str | 開示タイトル           |
| body_text       |  str | 本文テキスト           |
| source_url      |  str | 元データURLまたはファイルパス |

今回の主入力は `title` と `body_text` のみ。

`disclosure_time` やカテゴリは保存してよいが、モデル特徴量としては使わない。

---

### 5.2.2 株価データ

最低限必要なカラム：

| カラム名       |     型 | 内容    |
| ---------- | ----: | ----- |
| stock_code |   str | 銘柄コード |
| date       |  date | 取引日   |
| open       | float | 始値    |
| high       | float | 高値    |
| low        | float | 安値    |
| close      | float | 終値    |
| volume     | float | 出来高   |

---

## 5.3 データ収集モジュール

### `DisclosureClient`

```python
class DisclosureClient:
    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """指定期間の適時開示データを取得する。"""
```

責務：

```text
- 適時開示メタデータの取得
- PDFまたはHTML本文の取得
- 本文テキストの抽出
- rawデータとして保存
```

---

### `PriceClient`

```python
class PriceClient:
    def fetch(self, stock_codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """指定銘柄・期間のOHLCVデータを取得する。"""
```

責務：

```text
- 株価OHLCVの取得
- 欠損値チェック
- rawデータとして保存
```

---

### `TextExtractor`

```python
class TextExtractor:
    def extract(self, file_path: str) -> str:
        """PDFまたはHTMLから本文テキストを抽出する。"""
```

責務：

```text
- PDF本文テキスト抽出
- 文字化け除去
- 余分な空白や改行の正規化
```

---

## 5.4 サンプル作成

### `DatasetBuilder`

```python
class DatasetBuilder:
    def build(
        self,
        disclosures: pd.DataFrame,
        prices: pd.DataFrame,
        calendar: pd.DataFrame,
    ) -> pd.DataFrame:
        """開示データと株価データから学習用データセットを作成する。"""
```

処理手順：

```text
1. 適時開示データを読み込む
2. title と body_text を結合して text を作る
3. stock_code × disclosure_date 単位で集約する
4. 各 disclosure_date に対して翌営業日 target_date を付与する
5. target_date の open と close を株価データから結合する
6. close > open なら y=1、close < open なら y=0 を付与する
7. close == open のサンプルを除外する
8. 欠損データを除外する
9. processed dataset として保存する
```

---

## 5.5 集約仕様

同一銘柄・同一開示日の複数開示は以下の形式で結合する。

```text
[DISCLOSURE_1_TITLE]
...
[DISCLOSURE_1_BODY]
...

[DISCLOSURE_2_TITLE]
...
[DISCLOSURE_2_BODY]
...
```

集約後のカラム：

| カラム名            |     型 | 内容         |
| --------------- | ----: | ---------- |
| sample_id       |   str | サンプルID     |
| stock_code      |   str | 銘柄コード      |
| disclosure_date |  date | 開示日        |
| target_date     |  date | 翌営業日       |
| text            |   str | 結合済み開示テキスト |
| target_open     | float | 翌営業日の始値    |
| target_close    | float | 翌営業日の終値    |
| y               |   int | 目的変数       |
| num_disclosures |   int | 同日に結合した開示数 |

---

## 5.6 ラベル生成仕様

```python
def make_label(open_price: float, close_price: float) -> int | None:
    if close_price > open_price:
        return 1
    if close_price < open_price:
        return 0
    return None
```

`None` の場合は学習データから除外する。

---

## 5.7 データ分割仕様

ランダム分割は禁止。
時系列分割を使う。

```python
class TimeSeriesSplitter:
    def split(self, dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """disclosure_date に基づいて train / valid / test に分割する。"""
```

出力：

```text
train.parquet
valid.parquet
test.parquet
```

---

# 6. フェーズ2：モデル構築 🧠

## 6.1 目的

適時開示テキストから `y` を予測する2値分類モデルを構築する。

---

## 6.2 対象モデル

今回の最小スコープでは、以下の3モデルを対象にする。

| モデルID | モデル名                            | 目的          |
| ----- | ------------------------------- | ----------- |
| M0    | Majority Baseline               | 最低基準        |
| M1    | TF-IDF + Logistic Regression    | 古典NLPベースライン |
| M2    | Japanese Transformer Classifier | 主モデル        |

---

## 6.3 共通インターフェース

すべてのモデルは同じインターフェースを持つ。

```python
class BaseModel:
    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame | None = None) -> None:
        raise NotImplementedError

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(df)
        return (proba >= threshold).astype(int)

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        raise NotImplementedError
```

---

## 6.4 M0：Majority Baseline

### 概要

学習データで多数派だったラベルを常に予測する。

```python
class MajorityBaseline(BaseModel):
    majority_label: int
    positive_rate: float
```

### 仕様

```text
fit:
  train_df["y"] の多数派を記録する

predict_proba:
  全サンプルに positive_rate を返す

predict:
  全サンプルに majority_label を返す
```

---

## 6.5 M1：TF-IDF + Logistic Regression

### 概要

テキストをTF-IDFでベクトル化し、ロジスティック回帰で分類する。

```python
class TfidfLogRegModel(BaseModel):
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
```

### 入力

```text
df["text"]
```

### 出力

```text
P(y=1)
```

### 推奨設定

```yaml
model:
  name: "tfidf_logreg"
  tfidf:
    max_features: 100000
    ngram_range: [1, 2]
    min_df: 3
    max_df: 0.95
  classifier:
    C: 1.0
    class_weight: "balanced"
    max_iter: 1000
```

### 学習処理

```text
1. train_df["text"] で TF-IDF を fit
2. train_df["y"] で LogisticRegression を fit
3. valid_df に対して predict_proba
4. モデルとvectorizerを保存
```

---

## 6.6 M2：Japanese Transformer Classifier

### 概要

日本語事前学習Transformerを用いて、適時開示テキストを2値分類する。

```python
class TransformerClassifier(BaseModel):
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
```

### 入力形式

```text
df["text"]
```

### 最大長

初期設定では、テキストの先頭から最大トークン数までを使う。

```yaml
model:
  name: "transformer"
  pretrained_model_name: "cl-tohoku/bert-base-japanese-v3"
  max_length: 512
  batch_size: 16
  learning_rate: 2.0e-5
  epochs: 3
  weight_decay: 0.01
  early_stopping_patience: 2
```

### 長文処理

今回の最小仕様では、長文分割や階層型モデルは使わない。

```text
title + body の先頭 max_length トークンを使用する
```

理由：

```text
- 研究軸を増やさない
- 実装を単純に保つ
- タイトルと冒頭情報だけで予測力があるかを確認する
```

---

## 6.7 学習出力

各モデルは以下を保存する。

```text
models/
  {run_id}/
    config.yaml
    model.pkl or pytorch_model.bin
    tokenizer/
    metrics_valid.json
    train_log.json
```

---

# 7. フェーズ3：評価 📊

## 7.1 目的

テストデータに対する分類性能を評価し、モデル間比較を行う。

---

## 7.2 評価対象

```text
M0: Majority Baseline
M1: TF-IDF + Logistic Regression
M2: Transformer Classifier
```

---

## 7.3 評価指標

| 指標                | 内容                    |
| ----------------- | --------------------- |
| Accuracy          | 正解率                   |
| Balanced Accuracy | ラベル偏りを補正した正解率         |
| Precision         | 1と予測した中で実際に1だった割合     |
| Recall            | 実際に1のうち1と予測できた割合      |
| F1                | PrecisionとRecallの調和平均 |
| ROC-AUC           | 閾値に依存しない分類性能          |
| PR-AUC            | 陽性ラベルに対する性能           |
| Log Loss          | 確率予測の損失               |
| Brier Score       | 確率予測の較正度              |

最小レポートでは以下を必須にする。

```text
Accuracy
Balanced Accuracy
ROC-AUC
Log Loss
```

---

## 7.4 評価モジュール

### `Evaluator`

```python
class Evaluator:
    def evaluate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> dict:
        """分類性能を計算する。"""
```

出力例：

```json
{
  "accuracy": 0.523,
  "balanced_accuracy": 0.518,
  "precision": 0.541,
  "recall": 0.602,
  "f1": 0.570,
  "roc_auc": 0.536,
  "pr_auc": 0.552,
  "log_loss": 0.691,
  "brier_score": 0.249
}
```

---

## 7.5 予測結果保存

評価時には、各サンプルの予測結果を保存する。

```text
data/predictions/
  {run_id}_test_predictions.parquet
```

カラム：

| カラム名            |     型 | 内容      |
| --------------- | ----: | ------- |
| sample_id       |   str | サンプルID  |
| stock_code      |   str | 銘柄コード   |
| disclosure_date |  date | 開示日     |
| target_date     |  date | 取引日     |
| y_true          |   int | 正解ラベル   |
| y_proba         | float | P(y=1)  |
| y_pred          |   int | 予測ラベル   |
| target_open     | float | 翌営業日の始値 |
| target_close    | float | 翌営業日の終値 |

---

## 7.6 レポート出力

```text
reports/metrics/
  {run_id}_metrics.json
  model_comparison.csv
```

`model_comparison.csv` の例：

| model        | accuracy | balanced_accuracy | roc_auc | log_loss |
| ------------ | -------: | ----------------: | ------: | -------: |
| majority     |    0.512 |             0.500 |   0.500 |    0.693 |
| tfidf_logreg |    0.526 |             0.521 |   0.538 |    0.690 |
| transformer  |    0.534 |             0.529 |   0.551 |    0.687 |

---

# 8. フェーズ4：運用シミュレーション 💹

## 8.1 目的

モデルの予測確率を使った場合に、翌営業日の始値から終値までの仮想売買でどのような損益になるかを検証する。

これは研究上の補助評価であり、実運用成績を保証するものではない。

---

## 8.2 売買ルール

デフォルトでは、予測確率に応じて以下の売買を行う。

```text
p_up >= threshold_long:
  翌営業日の始値で買い
  翌営業日の終値で売り

p_up <= threshold_short:
  翌営業日の始値で空売り
  翌営業日の終値で買い戻し

それ以外:
  取引しない
```

デフォルト設定：

```yaml
simulation:
  threshold_long: 0.55
  threshold_short: 0.45
  allow_short: true
  transaction_cost_bps: 10
```

---

## 8.3 リターン計算

### ロング

[
r = \frac{Close - Open}{Open}
]

### ショート

[
r = \frac{Open - Close}{Open}
]

### 取引コスト控除後

往復コストを `transaction_cost_bps` とする。

[
r_{net} = r - \frac{transaction_cost_bps}{10000}
]

---

## 8.4 シミュレーションモジュール

### `SignalGenerator`

```python
class SignalGenerator:
    def generate(
        self,
        predictions: pd.DataFrame,
        threshold_long: float,
        threshold_short: float,
        allow_short: bool = True,
    ) -> pd.DataFrame:
        """予測確率から売買シグナルを生成する。"""
```

出力シグナル：

```text
1  = long
-1 = short
0  = no trade
```

---

### `Backtester`

```python
class Backtester:
    def run(
        self,
        signal_df: pd.DataFrame,
        transaction_cost_bps: float = 0.0,
    ) -> pd.DataFrame:
        """翌営業日の始値・終値を用いて仮想売買損益を計算する。"""
```

出力カラム：

| カラム名         |     型 | 内容         |
| ------------ | ----: | ---------- |
| sample_id    |   str | サンプルID     |
| stock_code   |   str | 銘柄コード      |
| trade_date   |  date | 売買日        |
| signal       |   int | 1, -1, 0   |
| entry_price  | float | 始値         |
| exit_price   | float | 終値         |
| gross_return | float | コスト控除前リターン |
| net_return   | float | コスト控除後リターン |
| y_true       |   int | 正解ラベル      |
| y_proba      | float | 予測確率       |

---

## 8.5 シミュレーション評価指標

| 指標                | 内容            |
| ----------------- | ------------- |
| num_trades        | 取引数           |
| trade_rate        | 全サンプルに対する取引割合 |
| win_rate          | 勝率            |
| mean_return       | 平均リターン        |
| median_return     | 中央値リターン       |
| cumulative_return | 累積リターン        |
| annualized_return | 年率換算リターン      |
| volatility        | リターン標準偏差      |
| sharpe_ratio      | シャープレシオ       |
| max_drawdown      | 最大ドローダウン      |
| long_trades       | ロング取引数        |
| short_trades      | ショート取引数       |

最小レポートでは以下を必須にする。

```text
num_trades
win_rate
mean_return
cumulative_return
sharpe_ratio
max_drawdown
```

---

## 8.6 運用上の制約

研究用シミュレーションでは、まず以下の単純化を置く。

```text
- 約定価格は翌営業日の始値とする
- 決済価格は同日の終値とする
- 注文遅延は考慮しない
- 板流動性は考慮しない
- 空売り規制や貸株可否は考慮しない
- スリッページは transaction_cost_bps に含める
```

このため、結果はあくまで**モデル比較用の仮想シミュレーション**として扱う。

---

# 9. CLI仕様 🖥️

## 9.1 データ収集

```bash
python scripts/collect_data.py \
  --config configs/default.yaml
```

出力：

```text
data/raw/disclosures/
data/raw/prices/
```

---

## 9.2 データセット作成

```bash
python scripts/build_dataset.py \
  --config configs/default.yaml
```

出力：

```text
data/processed/dataset.parquet
data/splits/train.parquet
data/splits/valid.parquet
data/splits/test.parquet
```

---

## 9.3 学習

```bash
python scripts/train.py \
  --config configs/tfidf_logreg.yaml
```

```bash
python scripts/train.py \
  --config configs/transformer.yaml
```

出力：

```text
models/{run_id}/
```

---

## 9.4 評価

```bash
python scripts/evaluate.py \
  --config configs/transformer.yaml \
  --model-dir models/{run_id}
```

出力：

```text
data/predictions/{run_id}_test_predictions.parquet
reports/metrics/{run_id}_metrics.json
```

---

## 9.5 運用シミュレーション

```bash
python scripts/simulate.py \
  --config configs/simulation.yaml \
  --predictions data/predictions/{run_id}_test_predictions.parquet
```

出力：

```text
data/simulations/{run_id}_trades.parquet
reports/simulation/{run_id}_simulation_metrics.json
reports/simulation/{run_id}_equity_curve.png
```

---

## 9.6 一括実行

```bash
python scripts/run_all.py \
  --config configs/default.yaml
```

実行順序：

```text
1. collect_data
2. build_dataset
3. train
4. evaluate
5. simulate
```

---

# 10. データ品質チェック 🔍

## 10.1 開示データチェック

必須チェック：

```text
- stock_code が欠損していない
- disclosure_date が欠損していない
- title または body_text が存在する
- text の文字数が一定以上ある
- 同一 disclosure_id の重複がない
```

---

## 10.2 株価データチェック

必須チェック：

```text
- open, close が欠損していない
- open > 0
- close > 0
- target_date が取引日である
- target_open と target_close が取得できる
```

---

## 10.3 ラベルチェック

```text
- y は 0 または 1 のみ
- close == open のサンプルが除外されている
- train / valid / test で日付が重複しない
- test が train より未来になっている
```

---

# 11. リーク防止仕様 🛡️

## 11.1 時系列リーク禁止

データ分割は必ず `disclosure_date` に基づく。

```text
train < valid < test
```

ランダム分割は使用しない。

---

## 11.2 未来情報の使用禁止

モデル入力に使用できるのは以下のみ。

```text
- title
- body_text
```

使用禁止：

```text
- target_open
- target_close
- y
- 翌営業日のニュース
- 翌営業日の追加開示
- 株価特徴量
- 業種
- 開示カテゴリ
- ネガポジラベル
```

---

## 11.3 前処理のfit範囲

TF-IDFなどの前処理は、学習データのみに `fit` する。

```text
OK:
train で fit
valid/test は transform のみ

NG:
全データで fit してから分割
```

---

# 12. ログ・再現性 🧾

## 12.1 乱数固定

```python
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

---

## 12.2 ログ保存

各runで以下を保存する。

```text
run_id
実行日時
git commit hash
config
データ件数
ラベル分布
学習ログ
評価指標
シミュレーション指標
```

保存先：

```text
reports/
models/{run_id}/
```

---

# 13. 主要データスキーマ 📦

## 13.1 `dataset.parquet`

| カラム名            |     型 |  必須 | 内容      |
| --------------- | ----: | --: | ------- |
| sample_id       |   str | yes | サンプルID  |
| stock_code      |   str | yes | 銘柄コード   |
| disclosure_date |  date | yes | 開示日     |
| target_date     |  date | yes | 翌営業日    |
| text            |   str | yes | 入力テキスト  |
| target_open     | float | yes | 翌営業日の始値 |
| target_close    | float | yes | 翌営業日の終値 |
| y               |   int | yes | 目的変数    |
| num_disclosures |   int | yes | 同日開示数   |

---

## 13.2 `predictions.parquet`

| カラム名            |     型 |  必須 | 内容     |
| --------------- | ----: | --: | ------ |
| sample_id       |   str | yes | サンプルID |
| stock_code      |   str | yes | 銘柄コード  |
| disclosure_date |  date | yes | 開示日    |
| target_date     |  date | yes | 売買対象日  |
| y_true          |   int | yes | 正解ラベル  |
| y_proba         | float | yes | 予測確率   |
| y_pred          |   int | yes | 予測ラベル  |
| target_open     | float | yes | 始値     |
| target_close    | float | yes | 終値     |

---

## 13.3 `trades.parquet`

| カラム名         |     型 |  必須 | 内容         |
| ------------ | ----: | --: | ---------- |
| sample_id    |   str | yes | サンプルID     |
| stock_code   |   str | yes | 銘柄コード      |
| trade_date   |  date | yes | 売買日        |
| signal       |   int | yes | 1, -1, 0   |
| entry_price  | float | yes | エントリー価格    |
| exit_price   | float | yes | 決済価格       |
| gross_return | float | yes | コスト控除前リターン |
| net_return   | float | yes | コスト控除後リターン |
| y_true       |   int | yes | 正解ラベル      |
| y_proba      | float | yes | 予測確率       |

---

# 14. テスト仕様 🧪

## 14.1 単体テスト

必須テスト：

```text
test_labeling.py
- close > open なら 1
- close < open なら 0
- close == open なら None

test_dataset_builder.py
- 同一銘柄・同一日の開示が1サンプルに集約される
- 翌営業日が正しく付与される
- 欠損価格データが除外される

test_metrics.py
- Accuracy, Balanced Accuracy, ROC-AUC, Log Loss が計算できる

test_simulation.py
- long のリターンが正しく計算される
- short のリターンが正しく計算される
- transaction cost が控除される
```

---

# 15. 最小実装ロードマップ 🚀

## Step 1：データセット作成

```text
- disclosure.csv を読み込む
- price.csv を読み込む
- stock_code × disclosure_date で集約
- target_date を付与
- y を作る
- train / valid / test に分割
```

## Step 2：TF-IDFモデル

```text
- TF-IDF + Logistic Regression を学習
- テストデータで評価
- predictions.parquet を保存
```

## Step 3：Transformerモデル

```text
- tokenizer で text をエンコード
- SequenceClassification モデルを学習
- テストデータで評価
- predictions.parquet を保存
```

## Step 4：運用シミュレーション

```text
- predictions.parquet から signal を作る
- open で建てて close で閉じる
- リターンと指標を計算
```

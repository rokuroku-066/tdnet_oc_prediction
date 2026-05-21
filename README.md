# tdnet_oc_prediction

## Smoke test (spec 9.6 integration flow)

仕様 9.6 のパイプライン順序 `collect_data -> build_dataset -> train -> evaluate -> simulate` を一気通貫で確認する最小手順です。

```bash
python scripts/run_all.py --config configs/example.yaml
```

期待される主要生成物:

- `models/<run_id>/`（学習モデル）
- `data/predictions/<run_id>_test_predictions.parquet`（評価予測）
- `data/simulations/<run_id>_trades.parquet`（シミュレーション結果）

`run_all.py` は各ステップ失敗時に即時停止し、失敗ステップ名と実行コマンドを含むエラーメッセージを出力します。

## TDnet + Stooq 実データ取得手順（`configs/tdnet_stooq.yaml`）

### 1) 実データ収集（collect_data）

```bash
python scripts/collect_data.py --config configs/tdnet_stooq.yaml
```

- `data.start_date`〜`data.end_date` は **開示取得期間** に使われます。
- 価格はラベル作成で翌営業日が必要なため、`collect_data.py` は価格取得期間だけ後ろに延長します。
  - 既定: `data.end_date + 10日`（`price_extra_days` で変更可）
  - 明示指定: `price_end_date` を指定するとその日付まで取得

### 2) データセット作成（build_dataset）

```bash
python scripts/build_dataset.py --config configs/tdnet_stooq.yaml
```

### 3) PDF抽出を無効化した疎通確認

TDnetのHTMLメタ情報のみで収集可否を確認したい場合、`configs/tdnet_stooq.yaml` の以下を切り替えます。

```yaml
data:
  tdnet_extract_pdf: false
```

その後、再度 collect_data を実行してください。

```bash
python scripts/collect_data.py --config configs/tdnet_stooq.yaml
```

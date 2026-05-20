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

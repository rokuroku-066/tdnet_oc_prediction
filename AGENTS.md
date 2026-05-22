# AGENTS.md

このリポジトリで作業するエージェント向けの簡易ガイドです。

## 1. 環境構築

### 前提
- Python 3.11 以上
- `pip` が利用可能

### セットアップ
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

開発時にテスト補助ツールも入れる場合:
```bash
pip install -e .[dev]
```

## 2. テスト方法

### 全テスト
```bash
pytest -q
```

### 個別テスト
```bash
pytest -q tests/test_dataset_builder.py
```

### スモークテスト（README準拠）
```bash
python scripts/run_all.py --config configs/example.yaml
```

## 3. コーディング規約（Google Python Style Guide準拠）

- 基本方針は **Google Python Style Guide** に準拠する。
- 重要ポイント（要約）:
  - 命名: 関数/変数は `snake_case`、クラスは `PascalCase`、定数は `UPPER_SNAKE_CASE`。
  - Docstring: 公開関数・クラスには簡潔なdocstringを付ける。
  - 型ヒント: 新規/変更コードは可能な限り型ヒントを明示する。
  - 関数分割: 長大な関数を避け、責務ごとに分離する。
  - 例外処理: 握りつぶしを避け、必要に応じて文脈付きで再送出する。
  - import: 標準ライブラリ / サードパーティ / ローカルの順で整理する。
- 既存実装との整合性を優先し、差分は最小に保つ。

## 4. 実装時の補足
- 設定は `configs/*.yaml` と `src/tdnet_oc_prediction/config/` を優先確認する。
- 仕様変更時は、関連する `tests/` を更新してから実装を確定する。

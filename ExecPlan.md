# ExecPlan
## Current Task: TDnet/Stooq実データ対応の指摘反映
- [x] 指摘事項（設定不足・TDnetパース堅牢性・検証処理）を確認。
- [x] `configs/tdnet_stooq.yaml` に model/simulation を追加し、run_all系でも安全に実行可能化。
- [x] `TDnetPublicClient` のリンク選択・文字コード・ページ上限・company_name推定・PDF抽出フラグを改善。
- [x] `StooqClient` のNo data判定・UA設定を改善。
- [x] `TextExtractor` をTDnet側から利用して重複実装を削減。
- [x] `validate_disclosures` / `validate_prices` の日時正規化とnan文字列対策を実装。
- [x] 既存/追加テストを更新して実行。
- [x] コミット作成とPR記録。

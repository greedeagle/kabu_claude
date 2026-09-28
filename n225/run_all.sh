#!/usr/bin/env bash
# 日経225版パイプライン：データ取得からランキング出力までを順に実行する（出力は n225/out/）。
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
# データ取得（取得済みなら SKIP_FETCH=1）。構成銘柄履歴 PDF・構成一覧は data/raw/n225/ に保存済み
[ "${SKIP_FETCH:-0}" = 1 ] || $PY -m src.fetch_missing
$PY -m src.constituents             # 構成銘柄の履歴（PDF 解析・検証）→ out/constituents.csv
$PY -m src.repro                    # environment.txt, data_hash.txt
$PY -m src.panel                    # Step 1-2: 品質確認・ユニバース
$PY -m src.features                 # Step 4: 特徴量 + 切り詰め一致テスト
$PY -m src.univ_trunc_test          # Step 4: U(t) の切り詰め一致テスト
$PY -m src.labels                   # Step 3: ラベル + 単体テスト
$PY -m src.cv                       # 第6章: 分割
$PY -m src.ic                       # Step 5: 単独特徴量の評価
$PY -m src.models tune              # ハイパーパラメータ探索（最初のフォールドの学習期間内）
$PY -m src.models M1,M2a,M2b,M3,M4r,M4c,M5,M7   # Step 7: WF スコア
$PY -m src.grid baselines           # Step 6
$PY -m src.grid grid                # Step 9-10
$PY -m src.overfit                  # Step 11（構成銘柄リークテストを含む）
$PY -m src.holdout                  # Step 12（1回限り。合格候補なしなら未評価として記録）
$PY -m src.rank                     # Step 13
$PY -m src.report_stats             # 第15章の追加集計

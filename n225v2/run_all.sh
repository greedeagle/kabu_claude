#!/usr/bin/env bash
# 日経225 v2 パイプライン（開発 2016-09-29〜2025-08、テスト 2025-09-29〜2026-09-28）。出力は n225v2/out/。
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
if [ "${SKIP_FETCH:-0}" != 1 ]; then
  $PY -m src.fetch meta             # 構成銘柄履歴 PDF・現在の構成・JPX 一覧
  $PY -m src.constituents           # 構成銘柄の履歴 → out/constituents.csv
  $PY -m src.fetch prices           # 価格（取得済みの銘柄は飛ばす）
else
  $PY -m src.constituents
fi
$PY -m src.repro                    # environment.txt, data_hash.txt
$PY -m src.panel                    # Step 1-2: 品質確認・ユニバース（data_end で打ち切り）
$PY -m src.features                 # Step 4: 特徴量 + 切り詰め一致テスト
$PY -m src.univ_trunc_test          # Step 4: U(t) の切り詰め一致テスト
$PY -m src.labels                   # Step 3: ラベル（LM を含む）+ 単体テスト
$PY -m src.cv                       # 分割
$PY -m src.ic                       # Step 5: 単独特徴量の評価（LM）
$PY -m src.models tune              # ハイパーパラメータ探索（最初のフォールドの学習期間内）
$PY -m src.models M2r,M2b,M4r,M5    # Step 7: WF スコア
$PY -m src.grid baselines           # Step 6
$PY -m src.grid grid                # Step 9-10（台地 Sharpe で選定）
$PY -m src.overfit                  # Step 11
$PY -m src.holdout                  # Step 12: テスト（1回限り）
$PY -m src.rank                     # Step 13

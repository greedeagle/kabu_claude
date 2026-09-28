#!/usr/bin/env bash
# データ取得からランキング出力までを順に実行する。各ステップは結果ファイルを out/ に出力する。
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ "${SKIP_FETCH:-0}" = 1 ] || $PY -m src.fetch all          # データ取得（Yahoo / JPX）
$PY -m src.repro                    # environment.txt, data_hash.txt
$PY -m src.panel                    # Step 1-2: 品質確認・ユニバース
$PY -m src.features                 # Step 4: 特徴量 + 切り詰め一致テスト
$PY -m src.labels                   # Step 3: ラベル + 単体テスト
$PY -m src.cv                       # 第6章: 分割
$PY -m src.ic                       # Step 5: 単独特徴量の評価
$PY -m src.models tune              # ハイパーパラメータ探索（最初のフォールドの学習期間内）
$PY -m src.models M1,M2a,M2b,M3,M4r,M4c,M5,M7   # Step 7: WF スコア
$PY -m src.grid baselines           # Step 6
$PY -m src.grid grid                # Step 9-10
$PY -m src.overfit                  # Step 11
$PY -m src.holdout                  # Step 12（1回限り）
$PY -m src.rank                     # Step 13

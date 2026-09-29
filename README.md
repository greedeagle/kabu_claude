# 翌営業日寄り付き購入候補ランキング

日本株の日足データから、翌営業日の寄り付きで買う候補銘柄のランキングを作る研究用パイプラインです。
データ取得、品質確認、特徴量、ラベル、ウォークフォワード（WF）検証、過学習チェック、ホールドアウト評価、ランキング出力までを順に実行します。

> **研究目的の分析であり、投資助言ではありません。**

## 構成

| パス | 内容 |
|---|---|
| `run_all.sh` / `src/` / `config.yaml` / `out/` | 東証上場銘柄（プライム・スタンダード・グロース）版 |
| `n225/run_all.sh` / `n225/src/` / `n225/config.yaml` / `n225/out/` | 日経225採用銘柄版（構成銘柄の履歴をポイントインタイムで反映） |
| `n225/REPORT.md` | 日経225版の構築と検証の最終レポート |
| `n225v2/` / `n225v2/REPORT.md` | 日経225 改善版（開発 2016-09〜2025-08、テスト 2025-09〜2026-09）とその最終レポート |
| `data/raw/` | 取得した生データ（git 管理外） |

- データ入手元は Yahoo Finance（yfinance）と JPX 上場銘柄一覧です。APIキーは不要です。
- 各ステップの結果は `out/`（日経225版は `n225/out/`）に JSON / CSV で出力されます。
- 最終的なランキングは `out/step13_ranking.json`（日経225版は `n225/out/step13_ranking.json`）です。

## セットアップ

Python 3.14 で動作を確認しています（使用パッケージの一覧とバージョンは `n225/out/environment.txt`）。

```bash
cd ~/claudeCode_kabu2
python3 -m venv .venv
.venv/bin/pip install pandas numpy pyarrow pyyaml requests yfinance openpyxl \
    lightgbm scikit-learn scipy statsmodels holidays pdfplumber
chmod +x run_all.sh n225/run_all.sh
```

仮想環境は必ずリポジトリ直下の `.venv` に作ってください。両方の `run_all.sh` がこのパスの Python を直接使います。

## 手動で実行する

```bash
# 東証上場銘柄版（データ取得から）
./run_all.sh

# 取得済みデータを使う場合
SKIP_FETCH=1 ./run_all.sh

# 日経225版
cd n225 && SKIP_FETCH=1 ./run_all.sh
```

- スクリプトは自分のディレクトリに `cd` してから実行するので、どこから呼んでも動きます。
- `set -euo pipefail` なので、途中のステップが失敗するとそこで止まります。
- WF グリッド（数千通り）を含むため、全体の実行には時間がかかります。

## crontab -e で定期実行する

`crontab -e` でユーザーの crontab を開き、次のような行を追加します。
例は平日 18:00 に東証上場銘柄版を実行し、ログを `logs/` に日付つきで残す設定です。

```bash
crontab -e
```

```cron
# 分 時 日 月 曜日  コマンド
0 18 * * 1-5  cd /home/yusuke/claudeCode_kabu2 && mkdir -p logs && /usr/bin/flock -n /tmp/kabu2.lock ./run_all.sh >> logs/run_all_$(date +\%Y\%m\%d).log 2>&1
```

日経225版を実行する場合：

```cron
30 18 * * 1-5  cd /home/yusuke/claudeCode_kabu2/n225 && mkdir -p ../logs && /usr/bin/flock -n /tmp/kabu2_n225.lock ./run_all.sh >> ../logs/n225_$(date +\%Y\%m\%d).log 2>&1
```

登録内容の確認：

```bash
crontab -l
```

### cron で実行するときの注意

- **`%` はエスケープが必要**です。crontab では `%` が改行扱いになるため、`date +\%Y\%m\%d` のように `\%` と書きます。
- **パスは絶対パスで書きます**。cron の環境変数（`PATH` など）はログインシェルと異なります。`run_all.sh` は `.venv/bin/python` を直接使うので、venv の activate は不要です。
- **時刻はサーバーのタイムゾーン**で解釈されます。`timedatectl` で `Asia/Tokyo` になっているか確認してください。Yahoo の当日分データは大引け（15:30）後しばらくしてから反映されるため、夕方以降の実行を推奨します。
- **`flock -n`** で多重起動を防いでいます。前回の実行が終わっていなければ、その回はスキップされます。
- **ログ**は `logs/`（git 管理外）に出力されます。失敗時はまずログの末尾を確認してください。
- 実行されたかどうかは `grep CRON /var/log/syslog`（または `journalctl -u cron`）で確認できます。

### 定期実行の前に知っておくべき現状の制約

現在のスクリプトは「一度きりの検証」を前提に書かれているため、そのまま毎日実行しても最新のランキングにはなりません。

1. **ホールドアウトは1回限り**：`out/step12_holdout.json` が存在すると、`src.holdout` は「再評価は禁止」として終了コード 1 で終了します。`set -e` のため、2回目以降の実行は Step 12 で止まり、`src.rank`（Step 13）まで到達しません。これは検証の規定（ホールドアウトの再評価禁止）による意図的な動作です。
2. **データ取得は差分更新しない**：`src.fetch` と `n225/src/fetch_missing.py` は、価格ファイル（`data/raw/prices/<コード>.parquet`）がまだない銘柄だけを取得します。取得済み銘柄の新しい日足は追加されません。
3. **日経225版の前提ファイル**：`n225/data/todo_codes.txt` と `data/raw/n225/`（構成銘柄履歴の PDF など）が必要です。これらは git 管理外です。

毎日のランキング更新を cron で回すには、「最新データの再取得 → Step 1〜4 の再計算 → 固定済みモデルでの Step 13」だけを行う日次用スクリプトを別に用意する必要があります。
なお、日経225版は現時点で「運用可能モデルなし」の結論です（`n225/REPORT.md`）。ランキングは常に「購入候補なし（検証不合格）」になります。

## 再現性

- 乱数シードは `config.yaml` の `seed: 42` です。
- 実行環境は `out/environment.txt`、入力データのハッシュは `out/data_hash.txt` に記録されます。
- Yahoo の調整後価格は取得時点で変わることがあるため、再取得した場合は `data_hash.txt` で入力の同一性を確認してください。
- 設定変更は `trial_log.csv` に試行として記録されます。合格ゲート（`gates`）は Step 0 で確定しており、変更しません。

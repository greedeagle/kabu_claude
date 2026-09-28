"""日経225（過去の構成銘柄を含む）のうち、価格ファイルがない現役上場銘柄と指数を Yahoo から取得する。
上場廃止銘柄は Yahoo にデータがないため取得できない（欠落として記録）。"""
import datetime as dt, json, time, sys
from pathlib import Path
import pandas as pd, yfinance as yf
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "prices"
todo = [c for c in (ROOT / "n225/data/todo_codes.txt").read_text().split() if not (RAW / f"{c}.parquet").exists()]
todo += ["^N225"]
failed = []
for i in range(0, len(todo), 10):
    batch = todo[i:i + 10]
    syms = [c if c.startswith("^") else f"{c}.T" for c in batch]
    for attempt in range(5):
        try:
            df = yf.download(syms, start="2000-01-01", auto_adjust=False, actions=True,
                             group_by="ticker", threads=False, progress=False)
            break
        except Exception as e:
            print("retry", e, flush=True); time.sleep(30 * (attempt + 1))
    for c, s in zip(batch, syms):
        try:
            sub = df[s].dropna(how="all")
        except KeyError:
            sub = pd.DataFrame()
        if sub.empty:
            failed.append(c); continue
        sub = sub.rename(columns={"Stock Splits": "Splits", "Adj Close": "AdjCloseYahoo"})
        sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
        sub.index.name = "Date"
        sub = sub.reset_index(); sub.insert(1, "Code", c.replace("^", "IDX_"))
        sub.to_parquet(RAW / f"{c.replace('^', 'IDX_')}.parquet")
    print(i + len(batch), "/", len(todo), "failed", failed, flush=True)
    time.sleep(2)
with (ROOT / "n225/data/fetch_log.jsonl").open("a") as fh:
    fh.write(json.dumps({"source": "Yahoo Finance via yfinance " + yf.__version__, "codes": todo,
                         "failed": failed, "fetched_at": dt.datetime.now().isoformat(timespec="seconds")}) + "\n")

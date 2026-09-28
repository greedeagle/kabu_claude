"""APIキー不要のデータ取得。

- 銘柄一覧: JPX「東証上場銘柄一覧」(data_j.xlsx, 現時点の上場銘柄のみ)
- 日足: Yahoo Finance (yfinance, auto_adjust=False)
    Open/High/Low/Close/Volume は分割調整済み、Dividends は分割調整済みの1株配当、
    Stock Splits は権利落ち日の分割比率。
- 指数: TOPIX 連動ETF 1306.T

使い方: python -m src.fetch [list|prices|all]
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
DOMESTIC = {"プライム（内国株式）", "スタンダード（内国株式）", "グロース（内国株式）"}
START = "2000-01-01"
BATCH = 50
INDEX_TICKER = "1306"


def fetch_list():
    r = requests.get(JPX_URL, timeout=60)
    r.raise_for_status()
    (RAW / "data_j.xlsx").write_bytes(r.content)
    d = pd.read_excel(RAW / "data_j.xlsx", dtype=str)
    d.to_parquet(RAW / "jpx_list.parquet")
    print("jpx list", len(d), "domestic", d["市場・商品区分"].isin(DOMESTIC).sum())


def _tickers() -> list[str]:
    d = pd.read_parquet(RAW / "jpx_list.parquet")
    codes = d.loc[d["市場・商品区分"].isin(DOMESTIC), "コード"].astype(str).tolist()
    return sorted(set(codes)) + [INDEX_TICKER]


def _to_long(df: pd.DataFrame, code: str) -> pd.DataFrame:
    df = df.dropna(how="all")
    if df.empty:
        return df
    df = df.rename(columns={"Stock Splits": "Splits", "Adj Close": "AdjCloseYahoo"})
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    df = df.reset_index()
    df.insert(1, "Code", code)
    return df


def fetch_prices():
    out = RAW / "prices"
    out.mkdir(parents=True, exist_ok=True)
    tickers = _tickers()
    todo = [c for c in tickers if not (out / f"{c}.parquet").exists()]
    failed: list[str] = []
    print(f"tickers {len(tickers)}, remaining {len(todo)}", flush=True)
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        syms = [f"{c}.T" for c in batch]
        for attempt in range(5):
            try:
                df = yf.download(syms, start=START, auto_adjust=False, actions=True,
                                 group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:  # ネットワーク・レート制限
                print("retry", attempt, e, flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            failed += batch
            continue
        for c, s in zip(batch, syms):
            try:
                sub = df[s] if isinstance(df.columns, pd.MultiIndex) else df
            except KeyError:
                failed.append(c)
                continue
            long = _to_long(sub.copy(), c)
            if long.empty:
                failed.append(c)
                continue
            long.to_parquet(out / f"{c}.parquet")
        print(f"{i + len(batch)}/{len(todo)} done, failed so far {len(failed)}", flush=True)
        time.sleep(1.0)
    (RAW / "fetch_failed.json").write_text(json.dumps(failed, ensure_ascii=False))
    with (RAW / "fetch_log.jsonl").open("a") as fh:
        fh.write(json.dumps({"source": "Yahoo Finance via yfinance " + yf.__version__,
                             "list_source": JPX_URL, "start": START,
                             "fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
                             "n_tickers": len(tickers), "n_failed": len(failed)}) + "\n")


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("list", "all"):
        fetch_list()
    if what in ("prices", "all"):
        fetch_prices()

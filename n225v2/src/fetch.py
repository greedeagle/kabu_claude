"""n225v2 のデータ取得（APIキー不要）。

1. 日経『日経平均株価銘柄変更履歴』PDF → data/raw/n225/
2. 現在の構成銘柄一覧 → data/raw/n225/current_members.csv
   日経の構成銘柄ページは Cloudflare のチャレンジで自動取得できないため、
   ~/python-kehai が同じページから取得した一覧（constituents.json）を複写する。
   9/29 に一時追加されたスピンオフ銘柄 646A は data_end（9/28）時点の構成に含まれないので除く。
3. JPX 上場銘柄一覧（銘柄名・33業種）→ data/raw/data_j.xlsx, jpx_list.parquet
4. 価格（yfinance, auto_adjust=False）: 構成銘柄履歴に出てくる全コード + ^N225 + 1321 + 1306
   → data/raw/prices/<code>.parquet（取得済みのファイルは上書きしない）

使い方: python -m src.fetch [meta|prices|all]
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

from .common import OUT, RAW, cfg

UA = {"User-Agent": "Mozilla/5.0 (compatible; market-research/1.0)"}
PDF_URL = "https://indexes.nikkei.co.jp/nkave/archives/file/history_of_nikkei_stock_average_component_changes_jp.pdf"
JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
KEHAI_MEMBERS = Path.home() / "python-kehai" / "nikkei225_cache" / "adjusted_ohlcv" / "constituents.json"
ADDED_AFTER_DATA_END = {"646A": "2026-09-29 にスピンオフで一時追加（data_end より後）"}


def fetch_meta():
    n = RAW / "n225"
    n.mkdir(parents=True, exist_ok=True)
    r = requests.get(PDF_URL, headers=UA, timeout=60)
    r.raise_for_status()
    assert r.content[:4] == b"%PDF", "history PDF not returned"
    (n / "history_of_nikkei_stock_average_component_changes_jp.pdf").write_bytes(r.content)
    items = json.loads(KEHAI_MEMBERS.read_text())
    mem = pd.DataFrame(items, columns=["code", "name"])
    mem = mem[~mem["code"].isin(ADDED_AFTER_DATA_END)]
    assert len(mem) == 225, len(mem)
    mem.to_csv(n / "current_members.csv", index=False)
    r = requests.get(JPX_URL, headers=UA, timeout=60)
    r.raise_for_status()
    (RAW / "data_j.xlsx").write_bytes(r.content)
    pd.read_excel(RAW / "data_j.xlsx", dtype=str).to_parquet(RAW / "jpx_list.parquet")
    (n / "fetched_at.txt").write_text(json.dumps({
        "pdf": PDF_URL, "members_source": str(KEHAI_MEMBERS),
        "members_file_mtime": dt.datetime.fromtimestamp(KEHAI_MEMBERS.stat().st_mtime).isoformat(timespec="seconds"),
        "excluded_from_members": ADDED_AFTER_DATA_END, "jpx": JPX_URL,
        "fetched_at": dt.datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False, indent=1))
    print("meta ok: members", len(mem))


def fetch_prices():
    c = cfg()
    out = RAW / "prices"
    out.mkdir(parents=True, exist_ok=True)
    iv = pd.read_csv(OUT / "constituents.csv", dtype={"code": str}, parse_dates=["out_date"])
    since = pd.Timestamp(c["data"]["panel_start"])
    codes = sorted(set(iv.loc[iv["out_date"].isna() | (iv["out_date"] > since), "code"]))
    codes += [c["data"]["etf_code"], c["data"]["topix_proxy_code"], "^N225"]
    todo = [x for x in codes if not (out / f"{x.replace('^', 'IDX_')}.parquet").exists()]
    failed = []
    print("codes", len(codes), "todo", len(todo), flush=True)
    for i in range(0, len(todo), 20):
        batch = todo[i:i + 20]
        syms = [x if x.startswith("^") else f"{x}.T" for x in batch]
        df = None
        for attempt in range(5):
            try:
                df = yf.download(syms, start=c["data"]["fetch_start"], auto_adjust=False, actions=True,
                                 group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:  # ネットワーク・レート制限
                print("retry", attempt, e, flush=True)
                time.sleep(30 * (attempt + 1))
        for x, s in zip(batch, syms):
            try:
                sub = df[s].dropna(how="all") if df is not None else pd.DataFrame()
            except KeyError:
                sub = pd.DataFrame()
            if sub.empty:
                failed.append(x)
                continue
            sub = sub.rename(columns={"Stock Splits": "Splits", "Adj Close": "AdjCloseYahoo"})
            sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
            sub.index.name = "Date"
            sub = sub.reset_index()
            code = x.replace("^", "IDX_")
            sub.insert(1, "Code", code)
            sub.to_parquet(out / f"{code}.parquet")
        print(i + len(batch), "/", len(todo), "failed", failed, flush=True)
        time.sleep(1)
    with (RAW / "n225v2_fetch_log.jsonl").open("a") as fh:
        fh.write(json.dumps({"source": "Yahoo Finance via yfinance " + yf.__version__, "n_codes": len(codes),
                             "failed": failed, "fetched_at": dt.datetime.now().isoformat(timespec="seconds")}) + "\n")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("meta", "all"):
        fetch_meta()
    if what in ("prices", "all"):
        fetch_prices()

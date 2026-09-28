"""第6章: データ分割（開発期間／ホールドアウト、拡大型ウォークフォワード、パージ）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import OUT, PROC, cfg, load_json, save_json


def calendar() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.read_parquet(PROC / "calendar.parquet")["Date"])


def splits() -> dict:
    """境界日を計算して out/splits.json に保存。"""
    c = cfg()
    cal = calendar()
    info = load_json(OUT / "step1_info.json")
    hs = pd.Timestamp(info["holdout_start"])
    nmax = max(c["split"]["n_candidates"])
    ph = cal.get_loc(hs)
    # 開発期間の最終シグナル日: t+1+N_max < holdout_start
    dev_end = cal[ph - nmax - 2]
    # 分析開始日: 特徴量（指数の SMA200 を含む）が揃い、ユニバースが 100 銘柄以上ある最初の日
    # 分析開始日: 指数の SMA200 と 250日系の特徴量が揃い、U(t) が 150 銘柄以上ある最初の日
    f = pd.read_parquet(PROC / "features.parquet", columns=["Date", "mk_above200", "dist_hi_250"])
    ok = f.groupby("Date").agg(n=("mk_above200", "size"), m=("mk_above200", lambda s: s.notna().mean()),
                               h=("dist_hi_250", lambda s: s.notna().mean()))
    start = ok.index[(ok["n"] >= 150) & (ok["m"] > 0.99) & (ok["h"] > 0.9)][0]
    v0 = cal[cal >= start + pd.DateOffset(years=c["split"]["first_train_years"])][0]
    folds = []
    v = v0
    while v <= dev_end:
        nxt = v + pd.DateOffset(months=c["split"]["fold_months"])
        end = min(cal[cal < nxt][-1], dev_end)
        folds.append({"fold": len(folds), "val_start": v.date().isoformat(), "val_end": end.date().isoformat()})
        later = cal[cal >= nxt]
        if len(later) == 0:
            break
        v = later[0]
    out = {"analysis_start": start.date().isoformat(), "dev_end_signal": dev_end.date().isoformat(),
           "holdout_start": hs.date().isoformat(), "last_date": cal[-1].date().isoformat(),
           "holdout_purge_days": nmax + 1, "folds": folds}
    save_json(out, OUT / "splits.json")
    return out


def load_splits() -> dict:
    return load_json(OUT / "splits.json")


def train_end_for(val_start: pd.Timestamp, N: int) -> pd.Timestamp:
    """学習データの最終シグナル日: v の (N+1) 営業日前より前（t+1+N < v）。"""
    cal = calendar()
    pv = cal.get_loc(pd.Timestamp(val_start))
    return cal[pv - N - 2]


def purge_check(train_dates: pd.Series, val_start, N: int) -> bool:
    cal = calendar()
    pos = cal.get_indexer(pd.DatetimeIndex(train_dates))
    pv = cal.get_loc(pd.Timestamp(val_start))
    return bool((pos + 1 + N < pv).all())


if __name__ == "__main__":
    s = splits()
    print({k: v for k, v in s.items() if k != "folds"}, "n_folds", len(s["folds"]))

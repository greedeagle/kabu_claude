"""Step 1（品質確認）と Step 2（ユニバース構築）。

出力:
  data/proc/panel.parquet   — 営業日カレンダーに揃えた銘柄×日のパネル（2007年以降）
  data/proc/calendar.parquet
  data/proc/dense.npz       — バックテスト用の日×銘柄の行列
  out/step1_quality.csv, out/step2_universe_counts.csv
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from .common import MAX_DATE, OUT, PROC, RAW, cfg, price_limit_width, save_json

PX = ["Open", "High", "Low", "Close"]


def load_raw() -> pd.DataFrame:
    files = sorted((RAW / "prices").glob("*.parquet"))
    dfs = []
    for f in files:
        d = pd.read_parquet(f)
        keep = [c for c in ["Date", "Code", "Open", "High", "Low", "Close", "Volume", "Dividends", "Splits"] if c in d]
        dfs.append(d[keep])
    df = pd.concat(dfs, ignore_index=True)
    df["Code"] = df["Code"].astype(str)
    if MAX_DATE:
        df = df[df["Date"] <= pd.Timestamp(MAX_DATE)]
    for c in ["Dividends", "Splits"]:
        if c not in df:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)
    return df


NICE_RATIOS = np.array([1.1, 1.2, 1.25, 1.5, 2, 2.5, 3, 4, 5, 10, 20, 25, 50, 100, 200, 300, 500, 1000])


def _nice(r: float) -> float:
    """分割比率（>1）を一般的な比率に丸める。5% 以内に無ければそのまま。"""
    k = NICE_RATIOS[np.argmin(np.abs(NICE_RATIOS / r - 1))]
    return float(k) if abs(k / r - 1) < 0.05 else float(r)


def clean_raw(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Yahoo データの補正。
    値幅制限の2倍を超える終値の変化は通常の取引では起こりえないため、データ誤りとみなす。
      - 3営業日以内に元の水準（±30%）へ戻る → 一時的な異常値: その行の OHLC を欠損（取引不可）
      - 新しい水準が続く → 分割の調整漏れ: 比率を推定し、それ以前の価格を調整、分割記録を補う
    始値だけが制限の2倍を超えて乖離している行は、始値を欠損（その日は約定不可）。"""
    stats = {"spike_rows": 0, "inferred_splits": 0, "bad_open": 0, "inferred_split_list": []}
    out = []
    for code, g in df.groupby("Code", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        C = g["Close"].to_numpy(dtype=float).copy()
        # 当日の株数ベースに換算する係数（当日より後の分割の累積積）
        F = split_factor_after(g).to_numpy()
        Cprev = np.r_[np.nan, C[:-1]]
        flagged = np.flatnonzero(np.abs(C - Cprev) * F > 2 * price_limit_width(Cprev * F) + 1e-9)
        nxt = 0
        for i in flagged:
            if i < nxt:
                continue
            if not (np.isfinite(C[i]) and np.isfinite(C[i - 1]) and C[i - 1] > 0):
                continue
            w = price_limit_width(np.array([C[i - 1] * F[i]]))[0]
            if abs(C[i] - C[i - 1]) * F[i] <= 2 * w + 1e-9:
                continue
            x = C[i] / C[i - 1]
            back = None
            for j in range(i + 1, min(i + 4, len(g))):
                if np.isfinite(C[j]) and abs(C[j] / C[i - 1] - 1) < 0.3:
                    back = j
                    break
            if back is not None:
                g.loc[i:back - 1, PX] = np.nan
                C[i:back] = np.nan
                stats["spike_rows"] += back - i
                nxt = back + 1
                continue
            # 分割の調整漏れ: それ以前を x 倍（価格）・1/x 倍（出来高）して連続にする
            r = _nice(1 / x) if x < 1 else 1 / _nice(x)
            g.loc[: i - 1, PX] = g.loc[: i - 1, PX] * (1 / r)
            g.loc[: i - 1, "Volume"] = g.loc[: i - 1, "Volume"] * r
            g.loc[: i - 1, "Dividends"] = g.loc[: i - 1, "Dividends"] * (1 / r)
            C[:i] = C[:i] / r
            prev = g.at[i, "Splits"]
            g.at[i, "Splits"] = r if not (prev > 0 and abs(prev / r - 1) < 0.05) else prev
            stats["inferred_splits"] += 1
            stats["inferred_split_list"].append([code, str(g.at[i, "Date"].date()), round(r, 4)])
        # 始値の異常（前日終値から制限の2倍超）
        Cp = pd.Series(C).shift(1).to_numpy()
        fa_all = split_factor_after(g).to_numpy()
        wv = price_limit_width(Cp * fa_all)
        badO = np.isfinite(Cp) & (np.abs(g["Open"].to_numpy() - Cp) * fa_all > 2 * wv)
        g.loc[badO, "Open"] = np.nan
        stats["bad_open"] += int(badO.sum())
        out.append(g)
    return pd.concat(out, ignore_index=True), stats


def split_factor_after(df: pd.DataFrame) -> pd.Series:
    """各行について、その日より後（翌日以降）に権利落ちとなる分割比率の累積積。
    未調整価格 = 調整後価格 × この係数。"""
    r = df["Splits"].where(df["Splits"] > 0, 1.0)
    # 同じ銘柄内で、日付の降順に累積積 → 自身を除くため shift
    rev = r.iloc[::-1].groupby(df["Code"].iloc[::-1]).cumprod()
    incl = rev.iloc[::-1]
    return (incl / r).astype(float)


def quality(df: pd.DataFrame, cal: pd.DatetimeIndex, holdout_start: pd.Timestamp | None) -> pd.DataFrame:
    rows = []

    def add(item, mask, action):
        m = np.asarray(mask)
        dev = m & (df["Date"] < holdout_start).to_numpy() if holdout_start is not None else m
        rows.append({"項目": item, "件数_全期間": int(m.sum()), "件数_開発期間": int(dev.sum()), "対処": action})

    add("OHLC 欠損", df[PX].isna().any(axis=1), "その日を取引不可フラグ（評価は前日終値で前方埋め）")
    add("重複行 (Date,Code)", df.duplicated(["Date", "Code"]), "先頭を残して削除")
    add("価格 <= 0", (df[PX] <= 0).any(axis=1), "取引不可フラグ・価格を欠損扱い")
    hi_bad = df["High"] < df[["Open", "Close"]].max(axis=1) * (1 - 1e-9)
    lo_bad = df["Low"] > df[["Open", "Close"]].min(axis=1) * (1 + 1e-9)
    add("高値 < max(始値,終値)", hi_bad, "高値 = max(O,H,C) に補正")
    add("安値 > min(始値,終値)", lo_bad, "安値 = min(O,L,C) に補正")
    add("出来高 0", df["Volume"].fillna(0) <= 0, "取引不可フラグ（売買停止の可能性）")
    ret = df.groupby("Code")["Close"].pct_change()
    big = ret.abs() > 0.5
    add("|日次リターン| > 50%（調整後終値）", big, "分割比率と照合（下表）。調整漏れ以外は保持")
    split_day = df["Splits"] > 0
    # 調整漏れの疑い: 分割日の前後で、比率に近い価格ジャンプが残る
    ratio = df["Splits"].where(split_day)
    jump = (1 + ret)
    miss = split_day & ((jump - 1 / ratio).abs() / (1 / ratio) < 0.15) & (ratio != 1)
    add("分割日に比率どおりのジャンプが残る（調整漏れ疑い）", miss, "該当銘柄の分割日前後を取引不可として扱う")
    wk = df["Date"].dt.weekday >= 5
    jp = holidays.Japan(years=range(2000, 2027))
    hol = df["Date"].dt.date.map(lambda d: d in jp or (d.month == 12 and d.day == 31) or (d.month == 1 and d.day in (1, 2, 3)))
    add("週末・祝日の日付の行", wk | hol, "カレンダー外の行として削除")
    add("営業日カレンダー外の日付", ~df["Date"].isin(cal), "削除")
    return pd.DataFrame(rows)


MARKET_CLOSURES = {pd.Timestamp("2020-10-01")}  # 東証システム障害による終日売買停止


def build_calendar(df: pd.DataFrame) -> pd.DatetimeIndex:
    """期待営業日（平日・祝日以外・12/31と1/1〜1/3以外）から既知の休場日を除いたもの。
    Yahoo にデータが無い営業日も営業日として残し、その日は全銘柄を取引不可として扱う。"""
    first, last = df["Date"].min(), df.loc[df["Volume"].fillna(0) > 0, "Date"].max()
    jp = holidays.Japan(years=range(first.year, last.year + 1))
    cal = [d for d in pd.bdate_range(first, last) if d.date() not in jp and not (d.month == 12 and d.day == 31)
           and not (d.month == 1 and d.day in (1, 2, 3)) and d not in MARKET_CLOSURES]
    return pd.DatetimeIndex(cal)


def data_holes(df: pd.DataFrame, cal: pd.DatetimeIndex) -> list[str]:
    """営業日のうち、出来高のある銘柄数が周辺の中央値の5%未満の日（Yahoo のデータ欠落）。"""
    cnt = df.loc[df["Volume"].fillna(0) > 0].groupby("Date").size().reindex(cal, fill_value=0)
    med = cnt.rolling(21, center=True, min_periods=5).median()
    return [d.date().isoformat() for d in cal[(cnt < 0.05 * med).to_numpy()]]


def calendar_mismatch(cal: pd.DatetimeIndex) -> dict:
    jp = holidays.Japan(years=range(cal[0].year, cal[-1].year + 1))
    exp = pd.bdate_range(cal[0], cal[-1])
    exp = pd.DatetimeIndex([d for d in exp if d.date() not in jp and not (d.month == 12 and d.day == 31)
                            and not (d.month == 1 and d.day in (1, 2, 3))])
    missing = exp.difference(cal)
    extra = cal.difference(exp)
    return {"expected_business_days": len(exp), "calendar_days": len(cal),
            "missing_vs_expected": [d.date().isoformat() for d in missing],
            "extra_vs_expected": [d.date().isoformat() for d in extra]}


def main():
    c = cfg()
    df = load_raw()
    n_raw = len(df)
    df = df.sort_values(["Code", "Date"]).reset_index(drop=True)
    cal_all = build_calendar(df)
    last = cal_all[-1]
    hs = last - pd.DateOffset(months=c["split"]["holdout_months"])
    holdout_start = cal_all[cal_all > hs][0]

    q = quality(df, cal_all, holdout_start)

    # --- 補正 ---
    df = df.drop_duplicates(["Date", "Code"])
    df = df[df["Date"].isin(cal_all)].reset_index(drop=True)
    df, clean_stats = clean_raw(df)
    df = df.sort_values(["Code", "Date"]).reset_index(drop=True)
    df["F"] = split_factor_after(df)
    save_json(clean_stats, OUT / "step1_clean_stats.json")
    df.loc[(df[PX] <= 0).any(axis=1), PX] = np.nan
    df["High"] = df[["Open", "High", "Close"]].max(axis=1, skipna=False)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1, skipna=False)
    ratio = df["Splits"].where(df["Splits"] > 0)
    ret = df.groupby("Code")["Close"].pct_change()
    miss = (df["Splits"] > 0) & (((1 + ret) - 1 / ratio).abs() / (1 / ratio) < 0.15) & (ratio != 1)
    bad_codes_dates = df.loc[miss, ["Code", "Date"]]

    # 大きな変動の内訳（分割日かどうか）
    big = df.loc[ret.abs() > 0.5, ["Code", "Date", "Splits"]].copy()
    big["ret"] = ret[ret.abs() > 0.5]
    big.to_csv(OUT / "step1_big_moves.csv", index=False)

    start = pd.Timestamp(c["data"]["panel_start"])
    cal = cal_all[cal_all >= start]
    pd.DataFrame({"Date": cal}).to_parquet(PROC / "calendar.parquet")

    # --- 営業日カレンダーに揃える ---
    parts = []
    for code, g in df.groupby("Code", sort=True):
        g = g.set_index("Date")
        first = g.index.min()
        idx = cal_all[(cal_all >= first) & (cal_all <= g.index.max())]
        g = g.reindex(idx)
        g["Code"] = code
        g["listed_days"] = np.arange(1, len(g) + 1)
        g["F"] = g["F"].bfill().ffill()
        g = g[g.index >= start]
        if len(g):
            parts.append(g)
    p = pd.concat(parts)
    p.index.name = "Date"
    p = p.reset_index()
    p["Volume"] = p["Volume"].fillna(0.0)
    p["Dividends"] = p["Dividends"].fillna(0.0)
    p["Splits"] = p["Splits"].fillna(0.0)
    p["tradable"] = p[PX].notna().all(axis=1) & (p["Volume"] > 0)
    if len(bad_codes_dates):
        key = set(zip(bad_codes_dates["Code"], bad_codes_dates["Date"]))
        m = [(a, b) in key for a, b in zip(p["Code"], p["Date"])]
        p.loc[m, "tradable"] = False
    # 取引不可日は O/H/L を欠損、C は前方埋め（評価用のみ）
    p.loc[~p["tradable"], ["Open", "High", "Low"]] = np.nan
    p["Close"] = p.groupby("Code")["Close"].ffill()
    p.loc[~p["tradable"], "Volume"] = 0.0

    # 未調整値（その日の株数ベース）
    p["C_u"] = p["Close"] * p["F"]
    p["O_u"] = p["Open"] * p["F"]
    p["TV"] = p["Close"] * p["Volume"]  # 売買代金の近似（調整後×調整後 = 未調整×未調整）
    p["ADV20"] = p.groupby("Code")["TV"].transform(lambda s: s.rolling(20, min_periods=20).mean())

    # 値幅制限（基準値 = 前日の未調整終値を当日の株数ベースに換算）
    prevC = p.groupby("Code")["Close"].shift(1)
    base = prevC * p["F"]
    w = price_limit_width(base.to_numpy())
    up = base + w
    dn = base - w
    p["stop_high_open"] = p["tradable"] & (p["O_u"] >= up * 0.999)
    p["stop_low_open"] = p["tradable"] & (p["O_u"] <= dn * 1.001)

    uc = c["universe"]
    p["in_univ"] = (p["tradable"] & (p["ADV20"] >= uc["min_adv20_yen"]) & (p["C_u"] >= uc["min_close_unadj"])
                    & (p["listed_days"] >= uc["min_listed_days"]))
    idx_code = c["data"]["index_ticker"]
    p.loc[p["Code"] == idx_code, "in_univ"] = False

    p.to_parquet(PROC / "panel.parquet")

    # --- 品質レポート ---
    q.to_csv(OUT / "step1_quality.csv", index=False)
    cm = calendar_mismatch(cal_all)
    ix = p[p["Code"] == idx_code]
    info = {
        "raw_rows": n_raw, "panel_rows": len(p), "n_codes_panel": int(p["Code"].nunique()),
        "calendar_first": cal_all[0].date().isoformat(), "calendar_last": last.date().isoformat(),
        "holdout_start": holdout_start.date().isoformat(),
        "calendar_check": {k: (v if not isinstance(v, list) else {"n": len(v), "sample": v[:20]}) for k, v in cm.items()},
        "data_holes_all_untradable": data_holes(df, cal_all),
        "index_rows": int(len(ix)), "index_first": ix["Date"].min().date().isoformat(),
        "index_missing_open_days": int((~ix["tradable"]).sum()),
        "split_miss_suspects_after_clean": int(len(bad_codes_dates)),
        "clean": {k: v for k, v in clean_stats.items() if k != "inferred_split_list"},
        "first_date_by_code_summary": p.groupby("Code")["Date"].min().dt.year.value_counts().sort_index().to_dict(),
    }
    save_json(info, OUT / "step1_info.json")

    # --- Step 2: ユニバースの日次銘柄数 ---
    uc_cnt = p[p["in_univ"]].groupby("Date").size().rename("n_universe")
    uc_cnt.to_csv(OUT / "step2_universe_counts.csv")

    # --- バックテスト用の行列 ---
    codes = np.array(sorted(p["Code"].unique()))
    dates = cal
    def wide(col, fill=np.nan):
        m = p.pivot(index="Date", columns="Code", values=col).reindex(index=dates, columns=codes)
        return m.astype(float).fillna(fill).to_numpy()

    def wide_bool(col):
        return wide(col, fill=0.0) > 0.5

    np.savez_compressed(
        PROC / "dense.npz", dates=dates.values.astype("datetime64[D]"), codes=codes,
        O=wide("Open"), C=wide("Close"), O_u=wide("O_u"), C_u=wide("C_u"),
        tradable=wide_bool("tradable"), shopen=wide_bool("stop_high_open"), slopen=wide_bool("stop_low_open"),
        div=wide("Dividends", fill=0.0), adv=wide("ADV20"), inuniv=wide_bool("in_univ"),
    )
    print("panel", len(p), "codes", len(codes), "dates", len(dates), "holdout_start", holdout_start.date())


if __name__ == "__main__":
    main()

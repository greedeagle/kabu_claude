"""Step 4: 特徴量（t 日の終値時点までのデータのみ）。

銘柄別特徴量は各銘柄の時系列だけから計算する（過去方向のローリング／EWM のみ）。
クロスセクション特徴量（順位・騰落比率・セクター）は同じ日のユニバース内のみで計算する。
出力: data/proc/features.parquet（ユニバース内の行のみ）
"""
from __future__ import annotations

import multiprocessing as mp
import sys

import numpy as np
import pandas as pd

from .common import OUT, PROC, RAW, cfg, save_json

EPS = 1e-12


def _sma(s, w):
    return s.rolling(w, min_periods=w).mean()


def _ema(s, span):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _streak(x: np.ndarray) -> np.ndarray:
    out = np.zeros(len(x))
    run = 0
    for i, v in enumerate(x):
        run = run + 1 if v else 0
        out[i] = run
    return out


def _since_sign_change(sign: np.ndarray, cap=250) -> np.ndarray:
    out = np.full(len(sign), np.nan)
    last_sign, cnt = 0, 0
    for i, s in enumerate(sign):
        if not np.isfinite(s) or s == 0:
            last_sign, cnt = 0, 0
            continue
        if s != last_sign:
            cnt = 0
            last_sign = s
        else:
            cnt += 1
        out[i] = s * min(cnt, cap)
    return out


def stock_features(g: pd.DataFrame, idx_close: pd.Series) -> pd.DataFrame:
    """g: 1銘柄の時系列（Date 昇順, 調整後 Open/High/Low/Close/Volume, tradable）。"""
    C = g["Close"].astype(float)
    O = g["Open"].fillna(C).astype(float)
    H = g["High"].fillna(C).astype(float)
    L = g["Low"].fillna(C).astype(float)
    V = g["Volume"].fillna(0).astype(float)
    Cp = C.shift(1)
    f = {}
    # トレンド
    sma = {w: _sma(C, w) for w in (5, 25, 75, 200)}
    for w in (5, 25, 75, 200):
        f[f"sma_dev_{w}"] = C / sma[w] - 1
        f[f"ema_dev_{w}"] = C / _ema(C, w) - 1
        f[f"sma_slope_{w}"] = sma[w] / sma[w].shift(5) - 1
    f["ma_5_25"] = np.sign(sma[5] - sma[25])
    f["ma_25_75"] = np.sign(sma[25] - sma[75])
    f["ma_75_200"] = np.sign(sma[75] - sma[200])
    f["cross_days"] = _since_sign_change(np.sign((sma[25] - sma[75]).to_numpy()))
    # モメンタム
    d = C.diff()
    au, ad = _wilder(d.clip(lower=0), 14), _wilder((-d).clip(lower=0), 14)
    f["rsi14"] = 100 - 100 / (1 + au / (ad + EPS))
    macd = _ema(C, 12) - _ema(C, 26)
    sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd"] = macd / C
    f["macd_sig"] = sig / C
    f["macd_hist"] = (macd - sig) / C
    for w in (60, 120):
        f[f"roc_{w}"] = C / C.shift(w) - 1
    hh14, ll14 = H.rolling(14, min_periods=14).max(), L.rolling(14, min_periods=14).min()
    k = (C - ll14) / (hh14 - ll14 + EPS) * 100
    f["stoch_k14"] = k
    f["stoch_d3"] = k.rolling(3, min_periods=3).mean()
    f["willr14"] = (hh14 - C) / (hh14 - ll14 + EPS) * -100
    # ボラティリティ
    tr = pd.concat([H - L, (H - Cp).abs(), (L - Cp).abs()], axis=1).max(axis=1)
    f["atr14p"] = _wilder(tr, 14) / C
    f["atr20p"] = _wilder(tr, 20) / C
    m20, s20 = _sma(C, 20), C.rolling(20, min_periods=20).std(ddof=0)
    f["bb_pb"] = (C - (m20 - 2 * s20)) / (4 * s20 + EPS)
    f["bb_bw"] = 4 * s20 / m20
    lr = np.log(C / Cp)
    f["hv20"] = lr.rolling(20, min_periods=20).std() * np.sqrt(252)
    f["hv60"] = lr.rolling(60, min_periods=60).std() * np.sqrt(252)
    f["range_p"] = (H - L) / C
    f["gap"] = O / Cp - 1
    # 出来高
    f["vr5"] = V / (_sma(V, 5) + EPS)
    f["vr20"] = V / (_sma(V, 20) + EPS)
    f["adv20_log"] = np.log1p(_sma(C * V, 20))
    obv = (np.sign(d).fillna(0) * V).cumsum()
    f["obv20"] = (obv - obv.shift(20)) / (V.rolling(20, min_periods=20).sum() + EPS)
    tp = (H + L + C) / 3
    mf = tp * V
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(14, min_periods=14).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(14, min_periods=14).sum()
    f["mfi14"] = 100 - 100 / (1 + pos / (neg + EPS))
    # 値動き
    for w in (1, 3, 5, 20):
        f[f"ret{w}"] = C / C.shift(w) - 1
    for w in (20, 60, 250):
        f[f"dist_hi_{w}"] = C / H.rolling(w, min_periods=w).max() - 1
        f[f"dist_lo_{w}"] = C / L.rolling(w, min_periods=w).min() - 1
    f["new_hi250"] = (H >= H.rolling(250, min_periods=250).max()).astype(float).where(H.rolling(250, min_periods=250).count() == 250)
    f["new_lo250"] = (L <= L.rolling(250, min_periods=250).min()).astype(float).where(L.rolling(250, min_periods=250).count() == 250)
    f["up_streak"] = _streak((d > 0).to_numpy())
    f["dn_streak"] = _streak((d < 0).to_numpy())
    f["pullback"] = ((C > sma[75]) & (sma[25] > sma[25].shift(5)) & (C < C.shift(3))).astype(float).where(sma[75].notna())
    # ローソク足
    rng = (H - L).replace(0, np.nan)
    f["body"] = (C - O) / rng
    f["upper_sh"] = (H - np.maximum(O, C)) / rng
    f["lower_sh"] = (np.minimum(O, C) - L) / rng
    f["doji"] = ((C - O).abs() / rng < 0.1).astype(float).where(rng.notna())
    Op = O.shift(1)
    bull_eng = (C > O) & (Cp < Op) & (O <= Cp) & (C >= Op)
    bear_eng = (C < O) & (Cp > Op) & (O >= Cp) & (C <= Op)
    f["engulf"] = bull_eng.astype(float) - bear_eng.astype(float)
    hi_b, lo_b = np.maximum(Op, Cp), np.minimum(Op, Cp)
    har = (np.maximum(O, C) < hi_b) & (np.minimum(O, C) > lo_b)
    f["harami"] = har.astype(float) * np.sign(C - O)
    f["gap_flag"] = (L > H.shift(1)).astype(float) - (H < L.shift(1)).astype(float)
    # 相対（日経平均の終値。日付で揃える）
    ic = idx_close.reindex(g["Date"]).to_numpy()
    icS = pd.Series(ic, index=C.index)
    f["rs20"] = (C / C.shift(20)) / (icS / icS.shift(20)) - 1
    f["rs60"] = (C / C.shift(60)) / (icS / icS.shift(60)) - 1
    out = pd.DataFrame(f, index=g.index)
    out = out.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    out.insert(0, "Date", g["Date"].to_numpy())
    out.insert(1, "Code", g["Code"].to_numpy())
    return out


def market_features(idx: pd.DataFrame, topix: pd.Series | None = None) -> pd.DataFrame:
    """日経平均の市場環境。topix: TOPIX代替（1306 の終値, Date index）→ NT倍率の20日変化。"""
    C = idx["Close"].astype(float)
    f = pd.DataFrame({"Date": idx["Date"].to_numpy()}, index=idx.index)
    if topix is not None:
        tp = topix.reindex(idx["Date"]).to_numpy()
        nt = pd.Series(C.to_numpy() / tp, index=idx.index)
        f["mk_nt20"] = nt / nt.shift(20) - 1
    f["mk_ret1"] = C / C.shift(1) - 1
    f["mk_ret5"] = C / C.shift(5) - 1
    f["mk_ret20"] = C / C.shift(20) - 1
    f["mk_above200"] = (C > _sma(C, 200)).astype(float).where(_sma(C, 200).notna())
    f["mk_vol20"] = np.log(C / C.shift(1)).rolling(20, min_periods=20).std() * np.sqrt(252)
    return f


STOCK_RANK_EXCLUDE = {"Date", "Code"}


def cross_section(feat: pd.DataFrame, sector: pd.Series) -> pd.DataFrame:
    """ユニバース行のみの feat に、同日クロスセクションの特徴量を付与。"""
    feat = feat.copy()
    feat["sector33"] = feat["Code"].map(sector).fillna("NA")
    n_sec = feat.groupby(["Date", "sector33"])["Code"].transform("size")
    # 騰落比率（同日のユニバース内）の5日平均
    up = feat.groupby("Date")["ret1"].apply(lambda s: (s > 0).sum())
    dn = feat.groupby("Date")["ret1"].apply(lambda s: (s < 0).sum())
    ratio = up / dn.replace(0, np.nan)
    # 各窓を独立に計算（逐次加算の丸め誤差で切り詰め計算と値がずれないように）
    br = (sum(ratio.shift(k) for k in range(5)) / 5).rename("breadth5")
    feat = feat.merge(br, left_on="Date", right_index=True, how="left")
    # セクター20日リターン（同日のユニバース内のセクター平均）と、セクター内順位
    feat["sec_ret20"] = feat.groupby(["Date", "sector33"])["ret20"].transform("mean").where(n_sec >= 3)
    feat["sec_rank20"] = feat.groupby(["Date", "sector33"])["ret20"].rank(pct=True).where(n_sec >= 3)
    return feat


def rank_cols(feat: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    r = feat.groupby("Date")[cols].rank(pct=True).astype(np.float32)
    r.columns = [f"r_{c}" for c in cols]
    return r


STOCK_COLS: list[str] = []
TOPIX: pd.Series | None = None


def _worker(args):
    g, idx_close = args
    return stock_features(g, idx_close)


def load_inputs():
    c = cfg()
    p = pd.read_parquet(PROC / "panel.parquet", columns=["Date", "Code", "Open", "High", "Low", "Close", "Volume", "tradable", "in_univ"])
    idx = p[p["Code"] == c["data"]["index_code"]].sort_values("Date").reset_index(drop=True)
    idx_close = idx.set_index("Date")["Close"]
    global TOPIX
    TOPIX = p[p["Code"] == c["data"]["topix_proxy_code"]].set_index("Date")["Close"]
    jl = pd.read_parquet(RAW / "jpx_list.parquet")
    sector = jl.set_index(jl["コード"].astype(str))["33業種区分"]
    return c, p, idx, idx_close, sector


def compute_all(p, idx_close, codes=None, until=None, n_proc=4):
    q = p if until is None else p[p["Date"] <= until]
    if codes is not None:
        q = q[q["Code"].isin(codes)]
    groups = [(g.reset_index(drop=True), idx_close if until is None else idx_close[idx_close.index <= until])
              for _, g in q.groupby("Code", sort=True)]
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = pool.map(_worker, groups, chunksize=16)
    return pd.concat(res, ignore_index=True)


def finalize(stock: pd.DataFrame, p: pd.DataFrame, idx: pd.DataFrame, sector: pd.Series) -> pd.DataFrame:
    univ = p.loc[p["in_univ"], ["Date", "Code"]]
    feat = univ.merge(stock, on=["Date", "Code"], how="left")
    stock_cols = [c for c in stock.columns if c not in STOCK_RANK_EXCLUDE]
    feat = cross_section(feat, sector)
    feat = feat.merge(market_features(idx, TOPIX[TOPIX.index <= idx["Date"].max()]), on="Date", how="left")
    rk = rank_cols(feat, stock_cols + ["sec_ret20"])
    feat = pd.concat([feat.reset_index(drop=True), rk.reset_index(drop=True)], axis=1)
    return feat


def truncation_test(p, idx, idx_close, sector, feat_full, n_dates=20, n_codes=20, seed=42):
    """無作為の20日付×20銘柄: t で切り詰めたデータから再計算し、全期間計算の値と完全一致するか。
    クロスセクション特徴量（順位など）は、その日のユニバース全銘柄を切り詰めデータで再計算して照合する。"""
    rng = np.random.default_rng(seed)
    dates = np.sort(feat_full["Date"].unique())
    dates = dates[len(dates) // 10:]  # 特徴量が揃った期間から
    pick = rng.choice(dates, size=n_dates, replace=False)
    mism = []
    n_checked = 0
    for t in pick:
        t = pd.Timestamp(t)
        # 騰落比率の5日平均のため、直近5営業日のユニバース銘柄をすべて再計算する
        prev5 = dates[max(0, np.searchsorted(dates, t.to_datetime64()) - 4): np.searchsorted(dates, t.to_datetime64()) + 1]
        day_codes = feat_full.loc[feat_full["Date"].isin(prev5), "Code"].unique()
        stock_tr = compute_all(p, idx_close, codes=set(day_codes), until=t)
        ft = finalize(stock_tr, p[p["Date"] <= t], idx[idx["Date"] <= t], sector)
        a = ft[ft["Date"] == t].set_index("Code").sort_index()
        b = feat_full[feat_full["Date"] == t].set_index("Code").sort_index()
        sample = rng.choice(a.index.to_numpy(), size=min(n_codes, len(a)), replace=False)
        cols = [c for c in b.columns if c not in ("Date",)]
        for code in sample:
            for col in cols:
                x, y = a.at[code, col], b.at[code, col]
                n_checked += 1
                same = (pd.isna(x) and pd.isna(y)) or x == y
                if not same:
                    mism.append({"Date": t.date().isoformat(), "Code": code, "feature": col, "truncated": x, "full": y})
    return n_checked, mism


def main():
    c, p, idx, idx_close, sector = load_inputs()
    stock = compute_all(p, idx_close)
    feat = finalize(stock, p, idx, sector)
    feat.to_parquet(PROC / "features.parquet")
    print("features", feat.shape, flush=True)
    if "--no-test" not in sys.argv:
        n, mism = truncation_test(p, idx, idx_close, sector, feat)
        save_json({"n_values_checked": n, "n_mismatch": len(mism), "mismatches": mism[:200]}, OUT / "step4_truncation_test.json")
        print("truncation test checked", n, "mismatch", len(mism))
        if mism:
            sys.exit("TRUNCATION TEST FAILED")


if __name__ == "__main__":
    main()

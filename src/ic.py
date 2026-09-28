"""Step 5: 単独特徴量の評価（最初のWFフォールドの学習期間（パージ後）のみ）。

- 日次 Rank IC（スピアマン）の平均、IC_IR、t 値（Newey-West, ラグ N）
- 年別 IC 符号の一致率、十分位（上位−下位）の L1 リターン
- 採用: 年別符号一致率 >= 60% かつ |t| > 2（主ラベル L3）
- 相関 > 0.9 の特徴量は |IC_IR| の大きい方を残す
注: 日次の順位相関は L1・L2・L3 で同一（同じ日に一定値を引くだけのため）。L3 で代表させる。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import OUT, PROC, cfg, save_json
from .cv import load_splits, train_end_for

MARKET_COLS = ["mk_ret1", "mk_ret5", "mk_ret20", "mk_above200", "mk_vol20", "breadth5"]


def rank_feature_cols(feat_cols) -> list[str]:
    return [c for c in feat_cols if c.startswith("r_")]


def newey_west_t(x: np.ndarray, lag: int) -> float:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return np.nan
    m = x.mean()
    e = x - m
    s = (e @ e) / n
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        s += 2 * w * (e[L:] @ e[:-L]) / n
    return m / np.sqrt(s / n)


def daily_ic(df: pd.DataFrame, fcols: list[str], lab: str) -> pd.DataFrame:
    """各日の順位相関（特徴量の日次パーセンタイル順位 × ラベルの日次順位）。"""
    d = df[["Date", lab] + fcols].dropna(subset=[lab])
    y = d.groupby("Date")[lab].rank(pct=True)
    out = {}
    for f in fcols:
        m = d[f].notna()
        x = d.loc[m, f].astype(float)
        yy = y[m]
        g = d.loc[m, "Date"]
        xr = x.groupby(g).rank(pct=True)
        yr = yy.groupby(g).rank(pct=True)
        xc = xr - xr.groupby(g).transform("mean")
        yc = yr - yr.groupby(g).transform("mean")
        num = (xc * yc).groupby(g).sum()
        den = np.sqrt((xc ** 2).groupby(g).sum() * (yc ** 2).groupby(g).sum())
        n = g.groupby(g).size()
        ic = (num / den).where(n >= 30)
        out[f] = ic
    return pd.DataFrame(out)


def decile_spread(df, f, lab="L1_1"):
    d = df[["Date", f, lab]].dropna()
    q = d.groupby("Date")[f].rank(pct=True)
    top = d[lab][q > 0.9].groupby(d["Date"][q > 0.9]).mean()
    bot = d[lab][q <= 0.1].groupby(d["Date"][q <= 0.1]).mean()
    return float((top - bot).mean())


def run():
    c = cfg()
    Ns = c["split"]["n_candidates"]
    sp = load_splits()
    v0 = pd.Timestamp(sp["folds"][0]["val_start"])
    start = pd.Timestamp(sp["analysis_start"])
    feat = pd.read_parquet(PROC / "features.parquet")
    lab = pd.read_parquet(PROC / "labels.parquet")
    fcols = rank_feature_cols(feat.columns)
    rows, adopted = [], {}
    for N in Ns:
        te = train_end_for(v0, N)
        fm = feat[(feat["Date"] >= start) & (feat["Date"] <= te)]
        df = fm[["Date", "Code"] + fcols].merge(lab[["Date", "Code", f"L1_{N}", f"L3_{N}", f"L4_{N}", f"L5_{N}"]], on=["Date", "Code"])
        for L in ("L3", "L4", "L5"):
            ics = daily_ic(df, fcols, f"{L}_{N}")
            yr = ics.index.year
            for f in fcols:
                s = ics[f].dropna()
                if len(s) < 50:
                    continue
                mean, sd = s.mean(), s.std()
                ym = s.groupby(s.index.year).mean()
                cons = float((np.sign(ym) == np.sign(mean)).mean())
                t = newey_west_t(s.to_numpy(), lag=max(1, N))
                rows.append({"N": N, "label": L, "feature": f, "ic_mean": mean, "ic_std": sd,
                             "ic_ir": mean / sd if sd > 0 else np.nan, "t_nw": t, "year_sign_consistency": cons,
                             "n_days": len(s), "decile_spread_L1": decile_spread(df, f, f"L1_{N}") if L == "L3" else np.nan,
                             "pass": bool(cons >= 0.6 and abs(t) > 2)})
        res = pd.DataFrame([r for r in rows if r["N"] == N and r["label"] == "L3"])
        cand = res[res["pass"]].sort_values("ic_ir", key=np.abs, ascending=False)
        # 相関 > 0.9 の除去（学習期間のプール相関、スピアマン＝順位のピアソン）
        sample = df.sample(n=min(200000, len(df)), random_state=42)
        corr = sample[cand["feature"].tolist()].corr()
        keep = []
        for f in cand["feature"]:
            if all(abs(corr.at[f, k]) <= 0.9 for k in keep):
                keep.append(f)
        adopted[N] = {"features": keep, "sign": {f: float(np.sign(cand.set_index("feature").at[f, "ic_mean"])) for f in keep},
                      "ic_ir": {f: float(cand.set_index("feature").at[f, "ic_ir"]) for f in keep},
                      "dropped_by_corr": [f for f in cand["feature"] if f not in keep],
                      "train_period": [start.date().isoformat(), te.date().isoformat()]}
        print("N", N, "pass", len(cand), "adopted", len(keep), flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "step5_ic_table.csv", index=False)
    save_json(adopted, OUT / "step5_adopted.json")


if __name__ == "__main__":
    run()

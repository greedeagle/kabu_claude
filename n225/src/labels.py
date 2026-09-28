"""Step 3: ラベル。R_N(t) = O(t+1+N)/O(t+1) - 1（調整後始値）。

ラベルは学習用。O(t+1) または O(t+1+N) が存在しない（取引不可）場合は欠損。
各ラベルに参照日付 buy_date=t+1, sell_date_N=t+1+N を保持する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import OUT, PROC, cfg, save_json, slippage_rate


def build():
    c = cfg()
    Ns = c["split"]["n_candidates"]
    p = pd.read_parquet(PROC / "panel.parquet", columns=["Date", "Code", "Open", "tradable", "in_univ", "ADV20"])
    cal = pd.read_parquet(PROC / "calendar.parquet")["Date"]
    pos = pd.Series(np.arange(len(cal)), index=cal.values)
    p = p.sort_values(["Code", "Date"]).reset_index(drop=True)
    p["Oe"] = p["Open"].where(p["tradable"])
    g = p.groupby("Code")
    idx = p[p["Code"] == c["data"]["etf_code"]].set_index("Date")["Oe"]
    out = p.loc[:, ["Date", "Code", "in_univ", "ADV20"]].copy()
    ti = pos.reindex(p["Date"]).to_numpy()
    out["buy_date"] = cal.reindex(ti + 1).to_numpy()
    o1 = g["Oe"].shift(-1)
    fee = c["trading"]["fee_oneway"]
    rt_cost = 2 * (fee + slippage_rate(p["ADV20"].to_numpy(), c["trading"]["slippage"]))
    atr = pd.read_parquet(PROC / "features.parquet", columns=["Date", "Code", "atr20p"])
    for N in Ns:
        oN = g["Oe"].shift(-1 - N)
        # 銘柄の時系列がカレンダーに揃っているので shift = 営業日のずれ（単体テストで確認）
        R = oN / o1 - 1
        out[f"sell_date_{N}"] = cal.reindex(ti + 1 + N).to_numpy()
        ib = idx.reindex(out["buy_date"]).to_numpy()
        isell = idx.reindex(out[f"sell_date_{N}"]).to_numpy()
        out[f"L1_{N}"] = R.astype(np.float32)
        out[f"L2_{N}"] = (R - (isell / ib - 1)).astype(np.float32)
        out[f"L4_{N}"] = (R > rt_cost + 0.005).astype(np.float32).where(R.notna())
    out = out[out["in_univ"]].drop(columns=["in_univ"]).reset_index(drop=True)
    out = out.merge(atr, on=["Date", "Code"], how="left")
    for N in Ns:
        med = out.groupby("Date")[f"L1_{N}"].transform("median")
        out[f"L3_{N}"] = (out[f"L1_{N}"] - med).astype(np.float32)
        out[f"L5_{N}"] = (out[f"L1_{N}"] / out["atr20p"]).astype(np.float32)
    out = out.drop(columns=["atr20p"])
    out.to_parquet(PROC / "labels.parquet")
    print("labels", out.shape)
    return out


def unit_test(n=2000, seed=42):
    """参照日付と値が第4章の定義に一致するかを、無作為の行で検証する。"""
    c = cfg()
    lab = pd.read_parquet(PROC / "labels.parquet")
    p = pd.read_parquet(PROC / "panel.parquet", columns=["Date", "Code", "Open", "tradable"]).set_index(["Code", "Date"]).sort_index()
    cal = pd.read_parquet(PROC / "calendar.parquet")["Date"].to_numpy()
    rng = np.random.default_rng(seed)
    rows = lab.iloc[rng.choice(len(lab), size=min(n, len(lab)), replace=False)]
    errs = []
    for _, r in rows.iterrows():
        i = np.searchsorted(cal, r["Date"].to_datetime64())
        assert cal[i] == r["Date"].to_datetime64()
        if i + 1 < len(cal) and pd.Timestamp(cal[i + 1]) != r["buy_date"]:
            errs.append(("buy_date", r["Code"], r["Date"]))
        for N in c["split"]["n_candidates"]:
            if i + 1 + N >= len(cal):
                if pd.notna(r[f"L1_{N}"]):
                    errs.append((f"label beyond data N={N}", r["Code"], r["Date"]))
                continue
            sd = pd.Timestamp(cal[i + 1 + N])
            if sd != r[f"sell_date_{N}"]:
                errs.append((f"sell_date_{N}", r["Code"], r["Date"]))
            try:
                a, b = p.loc[(r["Code"], r["buy_date"])], p.loc[(r["Code"], sd)]
            except KeyError:
                if pd.notna(r[f"L1_{N}"]):
                    errs.append((f"label w/o price N={N}", r["Code"], r["Date"]))
                continue
            exp = b["Open"] / a["Open"] - 1 if (a["tradable"] and b["tradable"]) else np.nan
            got = r[f"L1_{N}"]
            if not ((pd.isna(exp) and pd.isna(got)) or abs(exp - got) < 1e-5 * max(1, abs(exp))):
                errs.append((f"value N={N}", r["Code"], r["Date"], exp, got))
            # 未来の価格を参照していないこと: 参照日 > t
            if not (r["buy_date"] > r["Date"] and sd > r["buy_date"]):
                errs.append(("order", r["Code"], r["Date"]))
    res = {"n_rows_checked": len(rows), "n_errors": len(errs), "errors": [list(map(str, e)) for e in errs[:50]]}
    save_json(res, OUT / "step3_label_unittest.json")
    print("label unit test", res["n_rows_checked"], "errors", res["n_errors"])
    if errs:
        raise SystemExit("LABEL UNIT TEST FAILED")


if __name__ == "__main__":
    build()
    unit_test()

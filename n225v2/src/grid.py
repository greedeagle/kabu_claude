"""Step 6（ベースライン）, Step 9（グリッドバックテスト）, Step 10（合格ゲートと選定）。WF検証期間のみ。"""
from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np
import pandas as pd

from . import bt
from .common import OUT, PROC, RAW, cfg, load_json, save_json, slippage_rate
from .cv import load_splits
from .metrics import evaluate, gates, sharpe
from .triallog import log_many

J_EXTRA = 40
PRED_RULE: dict = {}   # v2: 目的変数が順位なので予測値の条件（b）は使わない
FAMILIES = ["M2r", "M2b", "M4r", "M5"]
SIMPLICITY = {"M2r": 1, "M2b": 2, "M4r": 4, "M5": 4}


class Ctx:
    def __init__(self, holdout=False):
        self.c = cfg()
        self.holdout = holdout
        self.sp = load_splits()
        self.D = bt.Dense()
        di = self.D.date_idx
        if not holdout:
            self.sig_start = di[pd.Timestamp(self.sp["folds"][0]["val_start"])]
            self.sig_end = di[pd.Timestamp(self.sp["dev_end_signal"])]
            self.end_day = di[pd.Timestamp(self.sp["holdout_start"])] - 1
        else:
            self.sig_start = di[pd.Timestamp(self.sp["holdout_start"])]
            self.end_day = len(self.D.dates) - 1
            self.sig_end = self.end_day - 1
        mk = pd.read_parquet(PROC / "features.parquet", columns=["Date", "mk_above200", "mk_vol20"]).drop_duplicates("Date").set_index("Date")
        mk = mk.reindex(self.D.dates)
        self.above = (mk["mk_above200"] > 0.5).to_numpy()
        # ボラティリティ閾値: 各フォールドの検証開始日より前の期間（学習部）の 80 パーセンタイル
        vol = mk["mk_vol20"]
        thr = np.full(len(self.D.dates), np.nan)
        start = pd.Timestamp(self.sp["analysis_start"])
        folds = self.sp["folds"] + ([{"val_start": self.sp["holdout_start"], "val_end": self.sp["last_date"]}] if holdout else [])
        for f in folds:
            a, b = pd.Timestamp(f["val_start"]), pd.Timestamp(f["val_end"])
            q = vol[(vol.index >= start) & (vol.index < a)].quantile(0.8)
            thr[(self.D.dates >= a) & (self.D.dates <= b)] = q
        self.vol_ok = (vol.to_numpy() < thr)
        jl = pd.read_parquet(RAW / "jpx_list.parquet")
        sec = dict(zip(jl["コード"].astype(str), jl["33業種区分"]))
        self.sectors = np.array([sec.get(str(c), "NA") for c in self.D.codes], dtype=object)

    def b7(self, N, cost_mult=1.0, fee=None, capital=None):
        """B7: 毎日 U(t) の全銘柄を等金額で購入し N 日後に売却（トランシェ方式、参加率上限・単元なし）。"""
        D = self.D
        J = int(D.inuniv.sum(1).max()) + 1
        cand = np.full((len(D.dates), J), -1, np.int64)
        for t in range(self.sig_start, self.sig_end + 1):
            ss = np.flatnonzero(D.inuniv[t])
            cand[t, :len(ss)] = ss
        res = bt.run(D, cand, self.sig_start, self.sig_end, self.end_day, N, J, None, fee=fee, cost_mult=cost_mult,
                     b7=True, capital=capital)
        m = evaluate(res, D.dates, self.c["trading"]["initial_capital"] if capital is None else capital,
                     sectors=self.sectors, boot=False)
        return res, m

    def regime_ok(self, name):
        if name == "none":
            return np.ones(len(self.D.dates), bool)
        if name == "idx_above_sma200":
            return self.above
        if name == "idx_vol_below_p80":
            return self.vol_ok
        if name == "excl_down_lowvol":   # 日経平均 < SMA200 かつ 低ボラ の日は買わない
            return self.above | ~self.vol_ok
        raise ValueError(name)

    def build_cand(self, sc: pd.DataFrame, theta=None, regime="none", pred_rule=None, J=10 + J_EXTRA, ascending=False):
        """sc: Date, Code, score[, pred]（ユニバース全行）。θ は同日のユニバース内のスコア上位比率。"""
        D = self.D
        s = sc.copy()
        n_day = s.groupby("Date")["Code"].transform("size")
        r = s.groupby("Date")["score"].rank(ascending=ascending, method="first")
        ok = s["score"].notna()
        if theta is not None:
            ok &= (r <= np.floor(theta * n_day).clip(lower=1))
        if pred_rule == "prob":
            ok &= s["pred"] > 0.5
        elif pred_rule == "ret":
            adv = D.adv[s["Date"].map(D.date_idx).to_numpy(), s["Code"].map(D.code_idx).to_numpy()]
            rt = 2 * (self.c["trading"]["fee_oneway"] + slippage_rate(adv, self.c["trading"]["slippage"]))
            ok &= s["pred"] > rt
        reg = self.regime_ok(regime)
        ti = s["Date"].map(D.date_idx).to_numpy()
        ok &= reg[ti]
        s = s[ok.to_numpy()].assign(_r=r[ok.to_numpy()])
        s = s[s["_r"] <= 10_000].sort_values(["Date", "_r"])
        s["_k"] = s.groupby("Date").cumcount()
        s = s[s["_k"] < J]
        cand = np.full((len(D.dates), J), -1, np.int64)
        cand[s["Date"].map(D.date_idx).to_numpy(), s["_k"].to_numpy()] = s["Code"].map(D.code_idx).to_numpy()
        return cand

    def run(self, cand, N, K, gmax=None, cost_mult=1.0, fee=None, boot=True, capital=None):
        res = bt.run(self.D, cand, self.sig_start, self.sig_end, self.end_day, N, K, gmax, fee=fee, cost_mult=cost_mult,
                     capital=capital)
        m = evaluate(res, self.D.dates, self.c["trading"]["initial_capital"] if capital is None else capital,
                     sectors=self.sectors, boot=boot)
        return res, m


def daily_ret(nav: pd.Series, capital: float) -> pd.Series:
    return nav / nav.shift(1).fillna(capital) - 1


def baselines(ctx: Ctx):
    """Step 6。B1 は 1000 回、B2〜B6 は N×K（g_max なし・レジームなし）。"""
    c = ctx.c
    D = ctx.D
    feat = pd.read_parquet(PROC / "features.parquet", columns=["Date", "Code", "ret20", "ret1", "sma_dev_25", "sma_slope_25", "adv20_log"])
    lo, hi = D.dates[ctx.sig_start], D.dates[ctx.sig_end]
    feat = feat[(feat["Date"] >= lo) & (feat["Date"] <= hi)]
    b4 = feat.assign(score=np.where((feat["sma_dev_25"] > 0) & (feat["sma_slope_25"] > 0), feat["adv20_log"], np.nan))
    scs = {"B3": feat.assign(score=feat["ret20"]), "B4": b4, "B5": feat.assign(score=feat["ret1"]),
           "B6": feat.assign(score=-feat["ret1"])}
    rows, table, logs = [], {}, []
    Jr = 10 + J_EXTRA
    b7r = {}
    for N in c["split"]["n_candidates"]:
        res7, m7 = ctx.b7(N)
        _, m7c2 = ctx.b7(N, cost_mult=2.0)
        b7r[f"N{N}"] = daily_ret(res7["nav"], c["trading"]["initial_capital"])
        table[f"B7_N{N}"] = {**{k: m7[k] for k in ("sharpe", "cagr", "mdd", "vol", "expectancy", "n_trades", "invested")},
                             "sharpe_cost2": m7c2["sharpe"], "year_returns": m7["year_returns"]}
        logs.append({"stage": "baseline", "model": "B7", "N": N, "regime": "none", "metrics": m7})
        print("B7", N, round(m7["sharpe"], 3), flush=True)
        for K in c["trading"]["K_candidates"]:
            key = f"N{N}_K{K}"
            table[key] = {}
            for b, sc in scs.items():
                cand = ctx.build_cand(sc[["Date", "Code", "score"]], J=K + J_EXTRA)
                _, m = ctx.run(cand, N, K)
                table[key][b] = {k: m[k] for k in ("sharpe", "cagr", "mdd", "expectancy", "n_trades", "exp_ci_lo")}
                logs.append({"stage": "baseline", "model": b, "N": N, "K": K, "theta": None, "gmax": None, "regime": "none", "metrics": m})
            t0 = time.time()
            srs, exps, cagrs = [], [], []
            for i in range(c.get("b1_trials", 1000)):
                cand = bt.random_cands(D.inuniv, ctx.sig_start, ctx.sig_end, K + J_EXTRA, 42 + i)
                _, m = ctx.run(cand, N, K, boot=False)
                srs.append(m["sharpe"]); exps.append(m["expectancy"]); cagrs.append(m["cagr"])
            srs, exps, cagrs = map(np.array, (srs, exps, cagrs))
            table[key]["B1"] = {f"{nm}_p{p}": float(np.nanpercentile(arr, p)) for nm, arr in
                                (("sharpe", srs), ("expectancy", exps), ("cagr", cagrs)) for p in (5, 50, 95)}
            np.save(OUT / f"b1_sharpes_{key}.npy", srs)
            np.save(OUT / f"b1_expectancy_{key}.npy", exps)
            logs.append({"stage": "baseline", "model": "B1", "N": N, "K": K, "regime": "none",
                         "params": {"n_trials": c.get("b1_trials", 1000)}, "metrics": {"sharpe": table[key]["B1"]["sharpe_p50"], **table[key]["B1"]}})
            table[key]["B7"] = table[f"B7_N{N}"]
            print("baseline", key, f"{time.time() - t0:.0f}s", json.dumps({k: round(v['sharpe'], 3) if 'sharpe' in v else round(v['sharpe_p95'], 3) for k, v in table[key].items()}), flush=True)
    pd.DataFrame(b7r).to_parquet(PROC / ("b7_returns_holdout.parquet" if ctx.holdout else "b7_returns.parquet"))
    # B2: 日経225連動ETF 1321 の買い持ち（分配金は再投資）。株を一切買わない戦略として、待機資金と同じ計算で求める
    empty = np.full((len(D.dates), 1), -1, np.int64)
    res = bt.run(D, empty, ctx.sig_start, ctx.sig_end, ctx.end_day, 1, 1, None)
    nav = res["nav"]
    m = evaluate(res, D.dates, c["trading"]["initial_capital"], boot=False)
    table["B2"] = {k: m[k] for k in ("sharpe", "cagr", "mdd", "vol", "total_return")}
    table["B2"]["year_returns"] = m["year_returns"]
    b2 = daily_ret(nav, c["trading"]["initial_capital"])
    b2.rename("B2").to_frame().to_parquet(PROC / ("b2_returns_holdout.parquet" if ctx.holdout else "b2_returns.parquet"))
    logs.append({"stage": "baseline", "model": "B2", "metrics": m})
    log_many(logs)
    save_json(table, OUT / "step6_baselines.json")
    return table


def model_grid(ctx: Ctx, families, base: dict, tag="wf"):
    c = ctx.c
    tc = c["trading"]
    g = c["gates"]
    Ns = c["split"]["n_candidates"]
    results = []
    ret_cols = {}
    logs = []
    b7r = pd.read_parquet(PROC / "b7_returns.parquet")
    b2r = pd.read_parquet(PROC / "b2_returns.parquet")["B2"]
    for fam in families:
        for N in Ns:
            f = PROC / "scores" / f"{fam}_N{N}.parquet"
            if not f.exists():
                continue
            sc = pd.read_parquet(f)
            if sc["score"].notna().sum() == 0:
                continue
            meta = load_json(PROC / "scores" / f"{fam}_N{N}_meta.json")
            t0 = time.time()
            for theta, regime in itertools.product(tc["theta_candidates"], tc["regime_candidates"]):
                cand = ctx.build_cand(sc, theta=theta, regime=regime, pred_rule=PRED_RULE.get(fam[:3] if fam.startswith("M4") else fam))
                for K, gmax in itertools.product(tc["K_candidates"], tc["gmax_candidates"]):
                    res, m = ctx.run(cand, N, K, gmax)
                    _, m2 = ctx.run(cand, N, K, gmax, cost_mult=2.0, boot=False)
                    key = f"N{N}_K{K}"
                    b1p95 = base[key]["B1"]["sharpe_p95"]
                    bbest = max(base[key][b]["sharpe"] for b in ("B3", "B4", "B5", "B6", "B7") if np.isfinite(base[key][b]["sharpe"]))
                    r = daily_ret(res["nav"], tc["initial_capital"])
                    lo7, hi7 = diff_ci(r, b7r[f"N{N}"])
                    m["b7_diff_ci"] = [lo7, hi7]
                    m["b7_sharpe"] = base[key]["B7"]["sharpe"]
                    lo2, hi2 = diff_ci(r, b2r)
                    m["b2_diff_ci"] = [lo2, hi2]
                    gt = gates(m, m2, b1p95, bbest, g, b7_diff_ci_lo=lo7, b2_diff_ci_lo=lo2)
                    b1 = np.load(OUT / f"b1_sharpes_{key}.npy")
                    m["b1_percentile"] = float((b1 < m["sharpe"]).mean() * 100) if np.isfinite(m["sharpe"]) else np.nan
                    m["expectancy_cost2"] = m2.get("expectancy")
                    m["sharpe_cost2"] = m2.get("sharpe")
                    tid = f"{fam}|N{N}|K{K}|th{theta}|g{gmax}|{regime}"
                    ret_cols[tid] = r.astype(np.float32)
                    row = {"id": tid, "family": fam, "N": N, "K": K, "theta": theta, "gmax": gmax, "regime": regime,
                           **{k: m[k] for k in ("sharpe", "cagr", "mdd", "expectancy", "exp_ci_lo", "n_trades", "trades_per_year",
                                                "pos_year_ratio", "max_year_share", "top5_stock_share", "expectancy_ex_top1pct",
                                                "nobuy_ratio", "invested", "win_rate", "profit_factor", "b1_percentile",
                                                "expectancy_cost2", "sharpe_cost2", "max_sector_share", "b7_sharpe")},
                           "b7_diff_ci_lo": m["b7_diff_ci"][0], "b7_diff_ci_hi": m["b7_diff_ci"][1],
                           "b2_diff_ci_lo": lo2, "b2_diff_ci_hi": hi2,
                           **gt}
                    results.append(row)
                    logs.append({"stage": tag, "model": fam, "N": N, "K": K, "theta": theta, "gmax": gmax, "regime": regime,
                                 "label": {"M2r": "none", "M2b": "LM", "M4r": "LM", "M5": "LMrank"}.get(fam),
                                 "features": load_json(OUT / "step5_adopted.json")[str(N)]["features"],
                                 "params": {"hp": load_json(OUT / "hp.json").get({"M4r": "reg", "M5": "rank"}.get(fam, ""), None)},
                                 "metrics": m, "gates_pass": gt["pass"]})
            print(fam, N, f"{time.time() - t0:.0f}s", "pass", sum(r["pass"] for r in results if r["family"] == fam and r["N"] == N), flush=True)
            log_many(logs)
            logs = []
    df = pd.DataFrame(results)
    R = pd.DataFrame(ret_cols)
    return df, R


def build_m6(ctx: Ctx, grid: pd.DataFrame):
    """M6: M2〜M5 のうち合格ゲートを通過したモデル（N ごと）のスコア順位の平均。"""
    made = []
    for N in ctx.c["split"]["n_candidates"]:
        fams = sorted(set(grid[(grid["N"] == N) & grid["pass"] & grid["family"].isin(["M2a", "M2b", "M3", "M4r", "M4c", "M5"])]["family"]))
        if len(fams) < 2:
            continue
        parts = []
        for fam in fams:
            s = pd.read_parquet(PROC / "scores" / f"{fam}_N{N}.parquet")
            s["rk"] = s.groupby("Date")["score"].rank(pct=True)
            parts.append(s.set_index(["Date", "Code"])["rk"].rename(fam))
        e = pd.concat(parts, axis=1)
        out = pd.DataFrame({"score": e.mean(axis=1), "pred": np.nan}).reset_index()
        out["fold"] = -1
        out.to_parquet(PROC / "scores" / f"M6_N{N}.parquet")
        save_json([{"members": fams}], PROC / "scores" / f"M6_N{N}_meta.json")
        made.append((N, fams))
    return made



def diff_ci(a: pd.Series, b: pd.Series, block=20, n=1000, seed=42):
    d = (a - b).dropna().to_numpy()
    nb = len(d) // block
    if nb < 2:
        return np.nan, np.nan
    blocks = d[:nb * block].reshape(nb, block).mean(1)
    rng = np.random.default_rng(seed)
    means = blocks[rng.integers(0, nb, size=(n, nb))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def plateau(grid: pd.DataFrame) -> pd.Series:
    """同じ (モデル, K, θ, g_max, レジーム) の N=5,10,20 の Sharpe の平均。"""
    key = grid["id"].str.replace(r"\|N\d+\|", "|N*|", regex=True)
    return grid.groupby(key)["sharpe"].transform("mean")


def select(grid: pd.DataFrame, R: pd.DataFrame, margin: float):
    """v2: 合格候補のうち台地 Sharpe が最大のもの。単純なファミリーの最良合格候補が台地 Sharpe で margin 以内、
    または日次リターン差の CI が 0 を含むなら、単純な方を選ぶ。"""
    grid = grid.assign(plateau=plateau(grid))
    p = grid[grid["pass"]].sort_values(["plateau", "sharpe"], ascending=False)
    if p.empty:
        return None, []
    top = p.iloc[0]
    choice = top
    simpler = p[p["family"].map(SIMPLICITY) < SIMPLICITY[top["family"]]]
    notes = []
    for lvl in sorted(simpler["family"].map(SIMPLICITY).unique()):
        cand = simpler[simpler["family"].map(SIMPLICITY) == lvl].iloc[0]
        lo, hi = diff_ci(R[top["id"]], R[cand["id"]])
        notes.append({"simpler": cand["id"], "plateau_diff": float(top["plateau"] - cand["plateau"]), "diff_ci": [lo, hi]})
        if top["plateau"] - cand["plateau"] < margin or (np.isfinite(lo) and lo <= 0 <= hi):
            choice = cand
            break
    order = [choice["id"]] + [i for i in p["id"] if i != choice["id"]]
    return choice, {"order": order, "notes": notes}


def main(stage):
    ctx = Ctx()
    if stage == "baselines":
        baselines(ctx)
        return
    base = load_json(OUT / "step6_baselines.json")
    if stage == "grid":
        grid, R = model_grid(ctx, FAMILIES, base)
        grid["plateau"] = plateau(grid)
        grid.to_csv(OUT / "step9_grid.csv", index=False)
        R.to_parquet(PROC / "grid_returns.parquet")
    if stage in ("grid", "select"):
        grid = pd.read_csv(OUT / "step9_grid.csv")
        R = pd.read_parquet(PROC / "grid_returns.parquet")
        choice, info = select(grid, R, ctx.c["selection"]["sharpe_margin"])
        out = {"n_candidates": int(len(grid)), "n_pass": int(grid["pass"].sum()),
               "gate_pass_counts": {g: int(grid[g].sum()) for g in ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]},
               "choice": None if choice is None else choice.to_dict(), "selection": info}
        save_json(out, OUT / "step10_selection.json")
        print(json.dumps({k: v for k, v in out.items() if k != "selection"}, ensure_ascii=False, default=str, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])

"""Step 6（ベースライン）, Step 9（グリッドバックテスト）, Step 10（合格ゲートと選定）。WF検証期間のみ。"""
from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np
import pandas as pd

from . import bt
from .common import OUT, PROC, cfg, load_json, save_json, slippage_rate
from .cv import load_splits
from .metrics import evaluate, gates, sharpe
from .triallog import log_many

J_EXTRA = 40
PRED_RULE = {"M3": "prob", "M4c": "prob", "M4r": "ret", "M7": "ret"}


class Ctx:
    def __init__(self, holdout=False):
        self.c = cfg()
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

    def regime_ok(self, name):
        if name == "none":
            return np.ones(len(self.D.dates), bool)
        if name == "idx_above_sma200":
            return self.above
        if name == "idx_vol_below_p80":
            return self.vol_ok
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

    def run(self, cand, N, K, gmax=None, cost_mult=1.0, fee=None, boot=True):
        res = bt.run(self.D, cand, self.sig_start, self.sig_end, self.end_day, N, K, gmax, fee=fee, cost_mult=cost_mult)
        m = evaluate(res, self.D.dates, self.c["trading"]["initial_capital"], boot=boot)
        return res, m


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
    for N in c["split"]["n_candidates"]:
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
            print("baseline", key, f"{time.time() - t0:.0f}s", json.dumps({k: round(v['sharpe'], 3) if 'sharpe' in v else round(v['sharpe_p95'], 3) for k, v in table[key].items()}), flush=True)
    # B2: 1306 を検証期間の最初の寄り付きで買って保有
    ix = D.code_idx[c["data"]["index_ticker"]]
    d0 = ctx.sig_start + 1
    fee = c["trading"]["fee_oneway"]
    sl = slippage_rate(np.array([D.adv[ctx.sig_start, ix]]), c["trading"]["slippage"])[0]
    sh = c["trading"]["initial_capital"] / (D.O[d0, ix] * (1 + sl) * (1 + fee))
    nav = pd.Series(sh * D.Cf[d0:ctx.end_day + 1, ix], index=D.dates[d0:ctx.end_day + 1])
    divs = pd.Series(sh * D.div[d0:ctx.end_day + 1, ix], index=nav.index).cumsum()
    nav = nav + divs
    res = {"nav": nav, "invested": pd.Series(1.0, index=nav.index), "trades": pd.DataFrame(columns=["stock", "buy_d", "exit_d", "cost", "proceeds", "ret", "pnl"]),
           "fails": {"stop_high": 0, "gap": 0, "halt": 0, "lot": 0}, "nobuy_days": 0, "signal_days": 1}
    m = evaluate(res, D.dates, c["trading"]["initial_capital"], boot=False)
    table["B2"] = {k: m[k] for k in ("sharpe", "cagr", "mdd", "vol", "total_return")}
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
                    bbest = max(base[key][b]["sharpe"] for b in ("B3", "B4", "B5", "B6") if np.isfinite(base[key][b]["sharpe"]))
                    gt = gates(m, m2, b1p95, bbest, g)
                    b1 = np.load(OUT / f"b1_sharpes_{key}.npy")
                    m["b1_percentile"] = float((b1 < m["sharpe"]).mean() * 100) if np.isfinite(m["sharpe"]) else np.nan
                    m["expectancy_cost2"] = m2.get("expectancy")
                    m["sharpe_cost2"] = m2.get("sharpe")
                    tid = f"{fam}|N{N}|K{K}|th{theta}|g{gmax}|{regime}"
                    r = daily_ret = res["nav"] / res["nav"].shift(1).fillna(tc["initial_capital"]) - 1
                    ret_cols[tid] = r.astype(np.float32)
                    row = {"id": tid, "family": fam, "N": N, "K": K, "theta": theta, "gmax": gmax, "regime": regime,
                           **{k: m[k] for k in ("sharpe", "cagr", "mdd", "expectancy", "exp_ci_lo", "n_trades", "trades_per_year",
                                                "pos_year_ratio", "max_year_share", "top5_stock_share", "expectancy_ex_top1pct",
                                                "nobuy_ratio", "invested", "win_rate", "profit_factor", "b1_percentile",
                                                "expectancy_cost2", "sharpe_cost2")},
                           **gt}
                    results.append(row)
                    logs.append({"stage": tag, "model": fam, "N": N, "K": K, "theta": theta, "gmax": gmax, "regime": regime,
                                 "label": {"M1": "L3", "M2a": "L3", "M2b": "L3", "M3": "L4", "M4r": "L3", "M4c": "L4", "M5": "L3rank", "M6": "ens", "M7": "L3"}.get(fam),
                                 "features": load_json(OUT / "step5_adopted.json")[str(N)]["features"],
                                 "params": {"hp": load_json(OUT / "hp.json").get({"M4r": f"reg_{N}", "M4c": f"clf_{N}", "M5": f"rank_{N}", "M7": f"reg_{N}", "M3": f"m3_{N}"}.get(fam, ""), None)},
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


SIMPLICITY = {"M1": 1, "M2a": 2, "M2b": 2, "M3": 3, "M4r": 4, "M4c": 4, "M5": 4, "M6": 5, "M7": 5}


def diff_ci(a: pd.Series, b: pd.Series, block=20, n=1000, seed=42):
    d = (a - b).dropna().to_numpy()
    nb = len(d) // block
    if nb < 2:
        return np.nan, np.nan
    blocks = d[:nb * block].reshape(nb, block).mean(1)
    rng = np.random.default_rng(seed)
    means = blocks[rng.integers(0, nb, size=(n, nb))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def select(grid: pd.DataFrame, R: pd.DataFrame, margin: float):
    p = grid[grid["pass"]].sort_values("sharpe", ascending=False)
    if p.empty:
        return None, []
    ranked = []
    top = p.iloc[0]
    choice = top
    # 単純な候補を優先: 単純なファミリーの最良合格候補から順に確認
    simpler = p[p["family"].map(SIMPLICITY) < SIMPLICITY[top["family"]]]
    notes = []
    for lvl in sorted(simpler["family"].map(SIMPLICITY).unique()):
        cand = simpler[simpler["family"].map(SIMPLICITY) == lvl].iloc[0]
        lo, hi = diff_ci(R[top["id"]], R[cand["id"]])
        notes.append({"simpler": cand["id"], "sharpe_diff": float(top["sharpe"] - cand["sharpe"]), "diff_ci": [lo, hi]})
        if top["sharpe"] - cand["sharpe"] < margin or (np.isfinite(lo) and lo <= 0 <= hi):
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
        fams = ["M1", "M2a", "M2b", "M3", "M4r", "M4c", "M5", "M7"]
        grid, R = model_grid(ctx, fams, base)
        grid.to_csv(OUT / "step9_grid.csv", index=False)
        R.to_parquet(PROC / "grid_returns.parquet")
        made = build_m6(ctx, grid)
        save_json(made, OUT / "step9_m6_members.json")
        if made:
            g6, R6 = model_grid(ctx, ["M6"], base)
            grid = pd.concat([grid, g6], ignore_index=True)
            R = pd.concat([R, R6], axis=1)
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

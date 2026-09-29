"""Step 11（v2）: 過学習チェック（選定順の上位3候補）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import bt
from .common import OUT, PROC, cfg, load_json, save_json
from .cv import train_end_for
from .grid import PRED_RULE, Ctx
from .metrics import daily_returns, deflated_sharpe, pbo_cscv, sharpe
from .models import KIND, Data, run_family, score_with
from .triallog import log_trial


def parse_id(tid: str) -> dict:
    fam, n, k, th, g, reg = tid.split("|")
    gv = g[1:]
    return {"family": fam, "N": int(n[1:]), "K": int(k[1:]), "theta": float(th[2:]),
            "gmax": None if gv in ("None", "nan") else float(gv), "regime": reg}


def run_spec(ctx: Ctx, sc: pd.DataFrame, spec: dict, theta=None, K=None, gmax="same", N=None, cost_mult=1.0, fee=None,
             capital=None):
    th = spec["theta"] if theta is None else theta
    g = spec["gmax"] if gmax == "same" else gmax
    cand = ctx.build_cand(sc, theta=th, regime=spec["regime"], pred_rule=PRED_RULE.get(spec["family"]))
    return ctx.run(cand, N or spec["N"], K or spec["K"], g, cost_mult=cost_mult, fee=fee, capital=capital)


def check(ctx: Ctx, tid: str, grid: pd.DataFrame, R: pd.DataFrame) -> dict:
    c = ctx.c
    spec = parse_id(tid)
    fam, N = spec["family"], spec["N"]
    oc = c["overfit_checks"]
    out = {"id": tid, "spec": spec}
    sc = pd.read_parquet(PROC / "scores" / f"{fam}_N{N}.parquet")
    res, m = run_spec(ctx, sc, spec)
    out["wf_metrics"] = {k: m[k] for k in ("sharpe", "cagr", "mdd", "expectancy", "n_trades")}
    wf_sr = m["sharpe"]
    r = daily_returns(res["nav"], c["trading"]["initial_capital"])

    # 1) 学習期間内 Sharpe と WF の乖離（フォールドごとのモデルで学習期間を採点してバックテスト）
    D = Data()
    _, _, models = run_family(fam, tag="_tmp", return_models=True)
    ins, wfs = [], []
    di = ctx.D.date_idx
    start = pd.Timestamp(ctx.sp["analysis_start"])
    for f in ctx.sp["folds"]:
        te = train_end_for(pd.Timestamp(f["val_start"]), D.H)
        d = D.df[(D.df["Date"] >= start) & (D.df["Date"] <= te)]
        tsc = pd.DataFrame({"Date": d["Date"].to_numpy(), "Code": d["Code"].to_numpy(),
                            "score": score_with(fam, models[f["fold"]], d, D), "pred": np.nan})
        cand = ctx.build_cand(tsc, theta=spec["theta"], regime=spec["regime"])
        rr = bt.run(ctx.D, cand, di[start], di[te], min(di[te] + N + 1, len(ctx.D.dates) - 1), N, spec["K"], spec["gmax"])
        ins.append(sharpe(daily_returns(rr["nav"], c["trading"]["initial_capital"])))
        a, b = pd.Timestamp(f["val_start"]), pd.Timestamp(f["val_end"])
        wfs.append(sharpe(r[(r.index > a) & (r.index <= b + pd.Timedelta(days=1))]))
    ins = np.array(ins, float)
    med = float(np.nanmedian(ins))
    out["train_vs_wf"] = {"in_sample_sharpe_by_fold": ins.tolist(), "wf_sharpe_by_fold": wfs, "median_in_sample": med,
                          "wf_sharpe": wf_sr, "ratio": float(wf_sr / med) if med > 0 else np.nan,
                          "pass": bool(med <= 0 or wf_sr / med >= oc["train_wf_ratio_min"])}

    # 2) 年度・レジーム別
    reg = pd.DataFrame({"r": r, "above": ctx.above[[di[d] for d in r.index]], "vol_ok": ctx.vol_ok[[di[d] for d in r.index]]})
    rtab = reg.groupby(["above", "vol_ok"])["r"].agg(days="size", ann_ret=lambda x: x.mean() * 252, sharpe=lambda x: sharpe(x))
    out["year_returns"] = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1).to_dict()
    out["regime_table"] = rtab.reset_index().to_dict(orient="records")

    # 3) パラメータ感度（隣接候補値と θ±20%、N は WF グリッドの値）
    tc = c["trading"]

    def adj(lst, v):
        lst = list(lst)
        i = lst.index(v)
        return [lst[j] for j in (i - 1, i + 1) if 0 <= j < len(lst)]
    neigh = []
    for Kv in adj(tc["K_candidates"], spec["K"]):
        neigh.append(("K", Kv, run_spec(ctx, sc, spec, K=Kv)[1]["sharpe"]))
    for tv in adj(tc["theta_candidates"], spec["theta"]) + [spec["theta"] * 0.8, spec["theta"] * 1.2]:
        neigh.append(("theta", tv, run_spec(ctx, sc, spec, theta=tv)[1]["sharpe"]))
    for Nv in adj(c["split"]["n_candidates"], N):
        rid = f"{fam}|N{Nv}|K{spec['K']}|th{spec['theta']}|g{spec['gmax']}|{spec['regime']}"
        row = grid[grid["id"] == rid]
        neigh.append(("N", Nv, float(row["sharpe"].iloc[0]) if len(row) else np.nan))
    ratios = [s / wf_sr if wf_sr > 0 else np.nan for _, _, s in neigh]
    mr = float(np.nanmin(ratios)) if ratios else np.nan
    out["param_sensitivity"] = {"neighbors": [{"param": a, "value": b, "sharpe": s, "ratio": q} for (a, b, s), q in zip(neigh, ratios)],
                                "min_ratio": mr, "pass": bool(np.isfinite(mr) and mr >= oc["param_sensitivity_min"])}

    # 4) シード感度（LightGBM 系のみ。別のシード集合のアンサンブル間の変動係数）
    if fam in KIND:
        srs = []
        for base in c["seed_sensitivity"]:
            s2 = run_family(fam, seeds=list(range(base, base + len(c["model_seeds"]))), tag="_tmp")[0]
            srs.append(run_spec(ctx, s2, spec)[1]["sharpe"])
        srs = np.array(srs)
        cv = float(np.std(srs, ddof=1) / abs(np.mean(srs))) if np.mean(srs) != 0 else np.inf
        out["seed_sensitivity"] = {"sharpes": srs.tolist(), "cv": cv, "pass": bool(cv <= oc["seed_cv_max"])}
    else:
        out["seed_sensitivity"] = {"note": "乱数を使わない", "pass": True}

    # 5) ラベル・シャッフル
    if fam == "M2r":
        out["label_shuffle"] = {"note": "ラベルで学習しないため対象外", "pass": True}
    else:
        s3 = run_family(fam, shuffle_labels=True, tag="_tmp")[0]
        _, m3 = run_spec(ctx, s3, spec)
        b1e = np.load(OUT / f"b1_expectancy_N{N}_K{spec['K']}.npy")
        lo, hi = np.nanpercentile(b1e, 5), np.nanpercentile(b1e, 95)
        e = m3.get("expectancy", np.nan)
        out["label_shuffle"] = {"expectancy": e, "sharpe": m3.get("sharpe"), "b1_expectancy_p5_p95": [lo, hi],
                                "pass": bool(lo <= e <= hi) if np.isfinite(e) else True}

    # 6) PBO / DSR
    out["pbo"] = pbo_cscv(R.to_numpy(dtype=np.float64), oc["pbo_S"])
    out["pbo_pass"] = bool(out["pbo"] < oc["pbo_max"])
    trial_sr = np.array([sharpe(R[col]) for col in R.columns]) / np.sqrt(252)
    rr = r.dropna().to_numpy()
    out["n_trials_total"] = int(pd.read_csv(OUT / "trial_log.csv").shape[0])
    out["n_strategy_trials_for_dsr"] = int(len(trial_sr))
    out["dsr"] = deflated_sharpe(wf_sr / np.sqrt(252), len(rr), float(stats.skew(rr)), float(stats.kurtosis(rr, fisher=False)), trial_sr)

    # 7) 特徴量の寄与の安定性
    meta = load_json(PROC / "scores" / f"{fam}_N{N}_meta.json")
    if fam in KIND:
        imps = [pd.Series(x["importance"]) for x in meta]
        cors = []
        for a, b in zip(imps[:-1], imps[1:]):
            top = sorted(set(a.nlargest(10).index) | set(b.nlargest(10).index))
            cors.append(stats.spearmanr(a[top], b[top]).statistic)
        out["importance_stability"] = {"mean_top10_rank_corr": float(np.nanmean(cors)), "top10_last_fold": imps[-1].nlargest(10).index.tolist()}
    elif fam == "M2b":
        out["importance_stability"] = {"sign_flip_share": float((pd.DataFrame([x["sign"] for x in meta]).nunique() > 1).mean())}
    else:
        out["importance_stability"] = {"note": "固定の等ウェイト"}

    # 8) 切り詰め一致テストの再確認（別シード）
    from .features import load_inputs, truncation_test
    _, p, idx, idx_close, sector = load_inputs()
    feat = pd.read_parquet(PROC / "features.parquet")
    n, mism = truncation_test(p, idx, idx_close, sector, feat, seed=7)
    ut = load_json(OUT / "step4_universe_truncation_test.json")
    out["truncation_recheck"] = {"n_checked": n, "n_mismatch": len(mism), "universe_test_mismatch_dates": ut["n_mismatch_dates"],
                                 "pass": len(mism) == 0 and ut["n_mismatch_dates"] == 0}

    keys = ["train_vs_wf", "param_sensitivity", "seed_sensitivity", "label_shuffle", "truncation_recheck"]
    out["pass"] = bool(all(out[k]["pass"] for k in keys) and out["pbo_pass"])
    out["dsr_below_0.95"] = bool(not np.isfinite(out["dsr"]) or out["dsr"] < oc["dsr_warn"])
    log_trial(stage="overfit_check", model=fam, N=N, K=spec["K"], theta=spec["theta"], gmax=spec["gmax"], regime=spec["regime"],
              params={"seed_sets": c["seed_sensitivity"]}, metrics={"sharpe": wf_sr, "pbo": out["pbo"], "dsr": out["dsr"]},
              gates_pass=out["pass"])
    return out


def main():
    ctx = Ctx()
    sel = load_json(OUT / "step10_selection.json")
    if sel["choice"] is None:
        save_json({"result": "運用可能モデルなし（合格ゲート通過なし）", "final": None}, OUT / "step11_overfit.json")
        print("no passing candidate")
        return
    grid = pd.read_csv(OUT / "step9_grid.csv")
    R = pd.read_parquet(PROC / "grid_returns.parquet")
    results, final = [], None
    for tid in sel["selection"]["order"][:3]:
        res = check(ctx, tid, grid, R)
        results.append(res)
        save_json({"checked": results, "final": final}, OUT / "step11_overfit.json")
        print(tid, "pass" if res["pass"] else "FAIL", flush=True)
        if res["pass"] and final is None:
            final = tid
    save_json({"checked": results, "final": final,
               "result": final if final else "運用可能モデルなし（過学習チェック不合格）"}, OUT / "step11_overfit.json")


if __name__ == "__main__":
    main()

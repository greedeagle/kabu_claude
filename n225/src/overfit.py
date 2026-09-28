"""Step 11: 過学習チェック（最終候補と次点2つ）。"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from .common import OUT, PROC, cfg, load_json, save_json
from .cv import load_splits, train_end_for
from .grid import PRED_RULE, Ctx
from .metrics import daily_returns, deflated_sharpe, evaluate, pbo_cscv, sharpe
from .models import Data, pred_lgb, pred_m3, pred_m7, run_family, score_m1, score_m2
from .triallog import log_trial


def parse_id(tid: str) -> dict:
    fam, n, k, th, g, reg = tid.split("|")
    gv = g[1:]
    return {"family": fam, "N": int(n[1:]), "K": int(k[1:]), "theta": float(th[2:]),
            "gmax": None if gv in ("None", "nan") else float(gv), "regime": reg}


def score_with(fam, mdl, d, D):
    if fam == "M1":
        return score_m1(mdl, d), np.full(len(d), np.nan)
    if fam == "M2a":
        return score_m2(d, D), np.full(len(d), np.nan)
    if fam == "M2b":
        return score_m2(d, D, mdl), np.full(len(d), np.nan)
    if fam == "M3":
        p = pred_m3(mdl, d, D)
        return p, p
    if fam in ("M4r", "M4c"):
        p = pred_lgb(mdl, d, D)
        return p, p
    if fam == "M5":
        return pred_lgb(mdl, d, D), np.full(len(d), np.nan)
    if fam == "M7":
        p = pred_m7(mdl, d, D)
        return p, p
    raise ValueError(fam)


def run_spec(ctx: Ctx, sc: pd.DataFrame, spec: dict, theta=None, K=None, gmax="same", N=None, cost_mult=1.0, fee=None):
    th = spec["theta"] if theta is None else theta
    g = spec["gmax"] if gmax == "same" else gmax
    cand = ctx.build_cand(sc, theta=th, regime=spec["regime"], pred_rule=PRED_RULE.get(spec["family"]))
    return ctx.run(cand, N or spec["N"], K or spec["K"], g, cost_mult=cost_mult, fee=fee)


def scores_for(fam, N, **kw):
    if fam == "M6":
        raise ValueError("M6 は構成モデルから作る")
    return run_family(fam, N, **kw)


def m6_scores(N, members, **kw):
    parts = []
    for fam in members:
        s, _ = run_family(fam, N, tag="_tmp", **kw)
        s["rk"] = s.groupby("Date")["score"].rank(pct=True)
        parts.append(s.set_index(["Date", "Code"])["rk"].rename(fam))
    e = pd.concat(parts, axis=1)
    return pd.DataFrame({"score": e.mean(axis=1), "pred": np.nan}).reset_index()


def check(ctx: Ctx, tid: str, grid: pd.DataFrame, R: pd.DataFrame, base: dict) -> dict:
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

    # 1) 学習期間内 Sharpe と WF の乖離（フォールドごとに学習データで採点して学習期間をバックテスト）
    members = load_json(PROC / "scores" / f"M6_N{N}_meta.json")[0]["members"] if fam == "M6" else None
    if fam != "M6":
        D = Data(N)
        _, meta, models = run_family(fam, N, tag="_tmp", return_models=True)
        ins, wfs = [], []
        di = ctx.D.date_idx
        start = pd.Timestamp(ctx.sp["analysis_start"])
        for f in ctx.sp["folds"]:
            mdl = models.get(f["fold"])
            if mdl is None and fam not in ("M2a",):
                continue
            te = train_end_for(pd.Timestamp(f["val_start"]), N)
            d = D.df[(D.df["Date"] >= start) & (D.df["Date"] <= te)]
            s, p = score_with(fam, mdl, d, D)
            tsc = pd.DataFrame({"Date": d["Date"].to_numpy(), "Code": d["Code"].to_numpy(), "score": s, "pred": p})
            cand = ctx.build_cand(tsc, theta=spec["theta"], regime=spec["regime"], pred_rule=PRED_RULE.get(fam))
            from . import bt
            rr = bt.run(ctx.D, cand, di[start], di[te], min(di[te] + N + 1, len(ctx.D.dates) - 1), N, spec["K"], spec["gmax"])
            ins.append(sharpe(daily_returns(rr["nav"], c["trading"]["initial_capital"])))
            a, b = pd.Timestamp(f["val_start"]), pd.Timestamp(f["val_end"])
            wfs.append(sharpe(r[(r.index > a) & (r.index <= b + pd.Timedelta(days=1))]))
        ins = np.array(ins, float)
        out["train_vs_wf"] = {"in_sample_sharpe_by_fold": ins.tolist(), "wf_sharpe_by_fold": wfs,
                              "median_in_sample": float(np.nanmedian(ins)), "wf_sharpe": wf_sr,
                              "ratio": float(wf_sr / np.nanmedian(ins)) if np.nanmedian(ins) > 0 else np.nan}
        out["train_vs_wf"]["pass"] = bool(np.nanmedian(ins) <= 0 or out["train_vs_wf"]["ratio"] >= oc["train_wf_ratio_min"])
    else:
        out["train_vs_wf"] = {"note": "M6 は構成モデルごとに確認", "pass": True}

    # 2) 年度・レジーム別
    yr = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    reg = pd.DataFrame({"r": r, "above": ctx.above[[ctx.D.date_idx[d] for d in r.index]],
                        "vol_ok": ctx.vol_ok[[ctx.D.date_idx[d] for d in r.index]]})
    rtab = reg.groupby(["above", "vol_ok"])["r"].agg(days="size", ann_ret=lambda x: x.mean() * 252, sharpe=lambda x: sharpe(x))
    out["year_returns"] = yr.to_dict()
    out["regime_table"] = rtab.reset_index().to_dict(orient="records")

    # 3) パラメータ感度（隣接候補値と θ±20%）
    tc = c["trading"]
    neigh = []

    def adj(lst, v):
        lst = list(lst)
        i = lst.index(v)
        return [lst[j] for j in (i - 1, i + 1) if 0 <= j < len(lst)]
    for Kv in adj(tc["K_candidates"], spec["K"]):
        neigh.append(("K", Kv, run_spec(ctx, sc, spec, K=Kv)[1]["sharpe"]))
    for gv in adj(tc["gmax_candidates"], spec["gmax"]):
        neigh.append(("gmax", gv, run_spec(ctx, sc, spec, gmax=gv)[1]["sharpe"]))
    for tv in adj(tc["theta_candidates"], spec["theta"]) + [spec["theta"] * 0.8, spec["theta"] * 1.2]:
        neigh.append(("theta", tv, run_spec(ctx, sc, spec, theta=tv)[1]["sharpe"]))
    for Nv in adj(c["split"]["n_candidates"], N):
        rid = f"{fam}|N{Nv}|K{spec['K']}|th{spec['theta']}|g{spec['gmax']}|{spec['regime']}"
        row = grid[grid["id"] == rid]
        neigh.append(("N", Nv, float(row["sharpe"].iloc[0]) if len(row) else np.nan))
    ratios = [s / wf_sr if wf_sr > 0 else np.nan for _, _, s in neigh]
    out["param_sensitivity"] = {"neighbors": [{"param": a, "value": b, "sharpe": s, "ratio": q} for (a, b, s), q in zip(neigh, ratios)],
                                "min_ratio": float(np.nanmin(ratios)) if ratios else np.nan}
    out["param_sensitivity"]["pass"] = bool(np.isfinite(out["param_sensitivity"]["min_ratio"]) and out["param_sensitivity"]["min_ratio"] >= oc["param_sensitivity_min"])

    # 4) シード感度
    srs = []
    for sd in c["seed_sensitivity"]:
        s2 = m6_scores(N, members, seed=sd) if fam == "M6" else run_family(fam, N, seed=sd, tag="_tmp")[0]
        srs.append(run_spec(ctx, s2, spec)[1]["sharpe"])
    srs = np.array(srs)
    cv = float(np.std(srs, ddof=1) / abs(np.mean(srs))) if np.mean(srs) != 0 else np.inf
    out["seed_sensitivity"] = {"sharpes": srs.tolist(), "cv": cv, "pass": bool(cv <= oc["seed_cv_max"])}

    # 5) ラベル・シャッフル
    if fam == "M2a":
        out["label_shuffle"] = {"note": "M2a はラベルで学習しない（等ウェイト）ため対象外", "pass": True}
    else:
        s3 = m6_scores(N, members, shuffle_labels=True) if fam == "M6" else run_family(fam, N, shuffle_labels=True, tag="_tmp")[0]
        _, m3 = run_spec(ctx, s3, spec)
        key = f"N{N}_K{spec['K']}"
        b1e = np.load(OUT / f"b1_expectancy_{key}.npy")
        lo, hi = np.nanpercentile(b1e, 5), np.nanpercentile(b1e, 95)
        e = m3.get("expectancy", np.nan)
        out["label_shuffle"] = {"expectancy": e, "sharpe": m3.get("sharpe"), "b1_expectancy_p5_p95": [lo, hi],
                                "pass": bool(np.isfinite(e) and lo <= e <= hi) if np.isfinite(e) else True,
                                "note": "取引なし" if not np.isfinite(e) else ""}

    # 6) PBO / DSR
    Rm = R.to_numpy(dtype=np.float64)
    out["pbo"] = pbo_cscv(Rm, c["overfit_checks"]["pbo_S"])
    out["pbo_pass"] = bool(out["pbo"] < oc["pbo_max"])
    trial_sr = np.array([sharpe(R[col]) for col in R.columns]) / np.sqrt(252)
    rr = r.dropna().to_numpy()
    out["n_trials_total"] = int(pd.read_csv(OUT / "trial_log.csv").shape[0])
    out["n_strategy_trials_for_dsr"] = int(len(trial_sr))
    out["dsr"] = deflated_sharpe(wf_sr / np.sqrt(252), len(rr), float(stats.skew(rr)), float(stats.kurtosis(rr, fisher=False)), trial_sr)

    # 7) 特徴量の寄与の安定性
    meta = load_json(PROC / "scores" / f"{fam}_N{N}_meta.json")
    if fam in ("M4r", "M4c", "M5", "M7"):
        imps = [pd.Series(x["importance"]) for x in meta if "importance" in x]
        cors = []
        for a, b in zip(imps[:-1], imps[1:]):
            top = sorted(set(a.nlargest(10).index) | set(b.nlargest(10).index))
            cors.append(stats.spearmanr(a[top], b[top]).statistic)
        out["importance_stability"] = {"mean_top10_rank_corr": float(np.nanmean(cors)), "by_fold_pair": cors,
                                       "top10_last_fold": imps[-1].nlargest(10).index.tolist()}
    elif fam == "M1":
        feats = [tuple(x["features"]) for x in meta]
        out["importance_stability"] = {"rule_by_fold": [list(f) for f in feats],
                                       "share_most_common_rule": float(pd.Series(feats).value_counts(normalize=True).iloc[0])}
    elif fam == "M2b":
        sg = pd.DataFrame([x["sign"] for x in meta])
        out["importance_stability"] = {"sign_flip_share": float((sg.nunique() > 1).mean())}
    else:
        out["importance_stability"] = {"note": "等ウェイト（固定）"}

    # 8) 切り詰め一致テストの再確認（別シード）
    from .features import load_inputs, truncation_test
    cc, p, idx, idx_close, sector = load_inputs()
    feat = pd.read_parquet(PROC / "features.parquet")
    n, mism = truncation_test(p, idx, idx_close, sector, feat, seed=7)
    ut = load_json(OUT / "step4_universe_truncation_test.json")
    out["truncation_recheck"] = {"n_checked": n, "n_mismatch": len(mism), "universe_test_mismatch_dates": ut["n_mismatch_dates"],
                                 "pass": len(mism) == 0 and ut["n_mismatch_dates"] == 0}

    # 9) 構成銘柄リークテスト（参考）: U(t) を現在の構成銘柄に置き換えた場合
    from .leak_eval import run_in_parent
    lk = run_in_parent(tid, members)
    lsr = lk["metrics"]["sharpe"]
    out["constituent_leak_test"] = {**lk, "correct_universe_sharpe": wf_sr, "diff_sharpe_correct_minus_current": (wf_sr - lsr) if lsr is not None else None,
                                    "note": ("正しい U(t) の成績が現在構成版と同等以上 → 構成銘柄の処理を再確認"
                                             if lsr is not None and np.isfinite(lsr) and wf_sr >= lsr else "現在構成版の方が良い（生存・採用バイアスの向きと整合）")}
    if lsr is not None and np.isfinite(lsr) and wf_sr >= lsr:
        # 指示書: 同等以上に良い場合は構成銘柄の処理を再確認（自動では不合格にしない。レポートで確認結果を記載）
        out["constituent_leak_test"]["recheck_required"] = True

    keys = ["train_vs_wf", "param_sensitivity", "seed_sensitivity", "label_shuffle", "truncation_recheck"]
    out["pass"] = bool(all(out[k]["pass"] for k in keys) and out["pbo_pass"])
    out["dsr_below_0.95"] = bool(not np.isfinite(out["dsr"]) or out["dsr"] < c["overfit_checks"]["dsr_warn"])
    log_trial(stage="overfit_check", model=fam, N=N, K=spec["K"], theta=spec["theta"], gmax=spec["gmax"], regime=spec["regime"],
              params={"seeds": c["seed_sensitivity"]}, metrics={"sharpe": wf_sr, "pbo": out["pbo"], "dsr": out["dsr"]}, gates_pass=out["pass"])
    return out


def main():
    ctx = Ctx()
    sel = load_json(OUT / "step10_selection.json")
    if sel["choice"] is None:
        save_json({"result": "運用可能モデルなし（合格ゲート通過なし）"}, OUT / "step11_overfit.json")
        print("no passing candidate")
        return
    grid = pd.read_csv(OUT / "step9_grid.csv")
    R = pd.read_parquet(PROC / "grid_returns.parquet")
    base = load_json(OUT / "step6_baselines.json")
    order = sel["selection"]["order"][:3]
    results = []
    final = None
    for tid in order:
        res = check(ctx, tid, grid, R, base)
        results.append(res)
        save_json({"checked": results, "final": final}, OUT / "step11_overfit.json")
        print(tid, "pass" if res["pass"] else "FAIL", flush=True)
        if res["pass"] and final is None:
            final = tid  # 選定順で最初に合格した候補（3候補とも検査は実施する）
    save_json({"checked": results, "final": final,
               "result": final if final else "運用可能モデルなし（過学習チェック不合格）"}, OUT / "step11_overfit.json")


if __name__ == "__main__":
    main()

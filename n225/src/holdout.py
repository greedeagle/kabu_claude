"""Step 12: ホールドアウト評価（1回限り）。

out/step12_holdout.json が存在する場合は再実行しない（P3）。
仕様（モデル種別・特徴量・ハイパーパラメータ・N・K・θ・g_max・レジーム）は Step 11 の結果で固定。
学習データはホールドアウト開始日より前の全データ（パージ適用）。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import bt
from .common import OUT, PROC, cfg, load_json, save_json
from .grid import J_EXTRA, PRED_RULE, Ctx
from .metrics import evaluate
from .models import Data, fit_model
from .overfit import parse_id, score_with
from .triallog import log_trial

LOCK = OUT / "step12_holdout.json"


def holdout_scores(fam, N, D: Data, val_start, members=None):
    if fam == "M6":
        parts = []
        for mf in members:
            s = holdout_scores(mf, N, D, val_start)
            s["rk"] = s.groupby("Date")["score"].rank(pct=True)
            parts.append(s.set_index(["Date", "Code"])["rk"].rename(mf))
        e = pd.concat(parts, axis=1)
        return pd.DataFrame({"score": e.mean(axis=1), "pred": np.nan}).reset_index()
    tr = D.train(val_start)
    mdl = fit_model(fam, tr, D)
    va = D.df[D.df["Date"] >= pd.Timestamp(val_start)]
    s, p = score_with(fam, mdl, va, D)
    return pd.DataFrame({"Date": va["Date"].to_numpy(), "Code": va["Code"].to_numpy(), "score": s, "pred": p})


def main():
    if LOCK.exists():
        sys.exit("ホールドアウトは評価済み（再評価は禁止）: " + str(LOCK))
    c = cfg()
    ov = load_json(OUT / "step11_overfit.json")
    final = ov.get("final")
    if not final:
        save_json({"evaluated": False, "reason": ov.get("result")}, LOCK)
        print("holdout not evaluated:", ov.get("result"))
        return
    spec = parse_id(final)
    fam, N = spec["family"], spec["N"]
    members = load_json(PROC / "scores" / f"M6_N{N}_meta.json")[0]["members"] if fam == "M6" else None
    ctx = Ctx(holdout=True)
    D = Data(N, include_holdout=True)
    hs = ctx.sp["holdout_start"]
    sc = holdout_scores(fam, N, D, hs, members)
    sc.to_parquet(PROC / "scores" / f"HOLDOUT_{fam}_N{N}.parquet")
    cand = ctx.build_cand(sc, theta=spec["theta"], regime=spec["regime"], pred_rule=PRED_RULE.get(fam))
    out = {"evaluated": True, "spec": spec, "id": final, "period": [hs, ctx.sp["last_date"]]}
    for cm in c["trading"]["cost_multipliers"]:
        res, m = ctx.run(cand, N, spec["K"], spec["gmax"], cost_mult=cm)
        out[f"metrics_cost{cm:g}"] = m
        if cm == 1.0:
            nav1 = res["nav"]
            trades1 = res["trades"]
    nav1.to_csv(OUT / "step12_holdout_nav.csv")
    trades1.assign(code=ctx.D.codes[trades1["stock"].to_numpy()], buy_date=ctx.D.dates[trades1["buy_d"].to_numpy()],
                   exit_date=ctx.D.dates[trades1["exit_d"].to_numpy()]).to_csv(OUT / "step12_holdout_trades.csv", index=False)
    wf = load_json(OUT / "step10_selection.json")
    wf_sr = next(x for x in ov["checked"] if x["id"] == final)["wf_metrics"]["sharpe"]
    m1 = out["metrics_cost1"]
    hp = c["holdout_pass"]
    out["wf_sharpe"] = wf_sr
    out["pass_cagr"] = bool(m1["cagr"] > hp["cagr_gt"])
    out["pass_sharpe_ratio"] = bool(np.isfinite(m1["sharpe"]) and m1["sharpe"] >= hp["sharpe_ratio_to_wf_min"] * wf_sr)
    # B7（同期間・同じ N の構成銘柄等ウェイト保有）の Sharpe を上回ること
    _, m7 = ctx.b7(N)
    out["B7_holdout"] = {k: m7[k] for k in ("sharpe", "cagr", "mdd", "vol")}
    out["pass_beat_b7"] = bool(np.isfinite(m1["sharpe"]) and m1["sharpe"] > m7["sharpe"])
    out["pass"] = out["pass_cagr"] and out["pass_sharpe_ratio"] and out["pass_beat_b7"]
    # 参考: 同期間のベースライン（判定には使わない）
    srs = []
    for i in range(c.get("b1_trials", 1000)):
        rc = bt.random_cands(ctx.D.inuniv, ctx.sig_start, ctx.sig_end, spec["K"] + J_EXTRA, 42 + i)
        srs.append(ctx.run(rc, N, spec["K"], boot=False)[1]["sharpe"])
    out["reference_B1_sharpe_p5_p50_p95"] = [float(np.nanpercentile(srs, q)) for q in (5, 50, 95)]
    out["reference_B1_percentile_of_model"] = float((np.array(srs) < m1["sharpe"]).mean() * 100)
    ix = ctx.D.code_idx[c["data"]["etf_code"]]
    d0 = ctx.sig_start + 1
    px = ctx.D.Cf[d0:, ix]
    out["reference_B2_1321_return"] = float(px[-1] / ctx.D.O[d0, ix] - 1)
    save_json(out, LOCK)
    log_trial(stage="holdout", model=fam, N=N, K=spec["K"], theta=spec["theta"], gmax=spec["gmax"], regime=spec["regime"],
              metrics=m1, gates_pass=out["pass"])
    print("holdout", "PASS" if out["pass"] else "FAIL", {k: m1[k] for k in ("sharpe", "cagr", "mdd", "expectancy", "n_trades")})


if __name__ == "__main__":
    main()

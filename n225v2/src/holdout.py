"""Step 12（v2）: テスト期間（2025-09-29〜2026-09-28）の評価（1回限り）。

out/step12_holdout.json が存在する場合は再実行しない。
仕様は Step 11 で固定。学習データはテスト開始日より前の全データ（パージ適用、2016-09-29 以降）。
合格基準（config.yaml holdout_pass、事前登録）:
  CAGR > 0 / Sharpe >= 0.4 × WF Sharpe / 総リターンが 1321 買い持ち（B2）を上回る（いずれもコスト1倍）
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import bt
from .common import OUT, PROC, cfg, load_json, save_json
from .grid import J_EXTRA, PRED_RULE, Ctx
from .models import Data, fit_model, score_with
from .overfit import parse_id
from .triallog import log_trial

LOCK = OUT / "step12_holdout.json"


def test_scores(fam, D: Data, val_start):
    tr = D.train(val_start)
    mdl = fit_model(fam, tr, D)
    va = D.df[D.df["Date"] >= pd.Timestamp(val_start)]
    return pd.DataFrame({"Date": va["Date"].to_numpy(), "Code": va["Code"].to_numpy(),
                         "score": score_with(fam, mdl, va, D), "pred": np.nan})


def b2_total(ctx: Ctx, c) -> dict:
    """1321 の買い持ち（分配金は再投資）。株を一切買わない戦略として計算する（grid.baselines の B2 と同じ）。"""
    D = ctx.D
    empty = np.full((len(D.dates), 1), -1, np.int64)
    nav = bt.run(D, empty, ctx.sig_start, ctx.sig_end, ctx.end_day, 1, 1, None)["nav"]
    r = nav / nav.shift(1).fillna(c["trading"]["initial_capital"]) - 1
    return {"total_return": float(nav.iloc[-1] / c["trading"]["initial_capital"] - 1),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)), "mdd": float((nav / nav.cummax() - 1).min())}


def main():
    if LOCK.exists():
        sys.exit("テストは評価済み（再評価は禁止）: " + str(LOCK))
    c = cfg()
    ov = load_json(OUT / "step11_overfit.json")
    final = ov.get("final")
    if not final:
        save_json({"evaluated": False, "reason": ov.get("result")}, LOCK)
        print("test not evaluated:", ov.get("result"))
        return
    spec = parse_id(final)
    fam, N = spec["family"], spec["N"]
    ctx = Ctx(holdout=True)
    D = Data(include_holdout=True)
    hs = ctx.sp["holdout_start"]
    sc = test_scores(fam, D, hs)
    sc.to_parquet(PROC / "scores" / f"HOLDOUT_{fam}.parquet")
    cand = ctx.build_cand(sc, theta=spec["theta"], regime=spec["regime"], pred_rule=PRED_RULE.get(fam))
    out = {"evaluated": True, "spec": spec, "id": final, "period": [hs, ctx.sp["last_date"]]}
    for cm in c["trading"]["cost_multipliers"]:
        res, m = ctx.run(cand, N, spec["K"], spec["gmax"], cost_mult=cm)
        out[f"metrics_cost{cm:g}"] = m
        if cm == 1.0:
            nav1, trades1 = res["nav"], res["trades"]
    nav1.to_csv(OUT / "step12_holdout_nav.csv")
    trades1.assign(code=ctx.D.codes[trades1["stock"].to_numpy()], buy_date=ctx.D.dates[trades1["buy_d"].to_numpy()],
                   exit_date=ctx.D.dates[trades1["exit_d"].to_numpy()]).to_csv(OUT / "step12_holdout_trades.csv", index=False)
    wf_sr = next(x for x in ov["checked"] if x["id"] == final)["wf_metrics"]["sharpe"]
    m1 = out["metrics_cost1"]
    hp = c["holdout_pass"]
    b2 = b2_total(ctx, c)
    out["wf_sharpe"] = wf_sr
    out["B2_test"] = b2
    out["pass_cagr"] = bool(m1["cagr"] > hp["cagr_gt"])
    out["pass_sharpe_ratio"] = bool(np.isfinite(m1["sharpe"]) and m1["sharpe"] >= hp["sharpe_ratio_to_wf_min"] * wf_sr)
    out["pass_beat_b2"] = bool(m1["total_return"] > b2["total_return"])
    out["pass"] = out["pass_cagr"] and out["pass_sharpe_ratio"] and out["pass_beat_b2"]
    # 参考（判定には使わない）: B7 と B1
    _, m7 = ctx.b7(N)
    out["reference_B7_test"] = {k: m7[k] for k in ("sharpe", "cagr", "mdd", "total_return")}
    srs = []
    for i in range(c.get("b1_trials", 1000)):
        rc = bt.random_cands(ctx.D.inuniv, ctx.sig_start, ctx.sig_end, spec["K"] + J_EXTRA, 42 + i)
        srs.append(ctx.run(rc, N, spec["K"], boot=False)[1]["sharpe"])
    out["reference_B1_sharpe_p5_p50_p95"] = [float(np.nanpercentile(srs, q)) for q in (5, 50, 95)]
    out["reference_B1_percentile_of_model"] = float((np.array(srs) < m1["sharpe"]).mean() * 100)
    save_json(out, LOCK)
    log_trial(stage="holdout", model=fam, N=N, K=spec["K"], theta=spec["theta"], gmax=spec["gmax"], regime=spec["regime"],
              metrics=m1, gates_pass=out["pass"])
    print("test", "PASS" if out["pass"] else "FAIL", {k: m1[k] for k in ("sharpe", "cagr", "total_return", "mdd", "n_trades")},
          "B2", b2)


if __name__ == "__main__":
    main()

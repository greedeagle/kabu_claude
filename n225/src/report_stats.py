"""第15章レポート用の追加集計（選定判定には使わない。WF検証期間のみ）。

- 最終候補（過学習チェックで不合格だが WF 選定1位）の感度分析: コスト 0/1/2 倍、手数料 0/0.05/0.10%、初期資金
- 年別・月別・レジーム別・セクター別成績、資産曲線
- モデル別の最良 WF 成績の比較表、B1 パーセンタイル
出力: out/report_stats.json, out/wf_nav_top.csv, out/wf_monthly_top.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import OUT, PROC, cfg, load_json, save_json
from .grid import PRED_RULE, Ctx, daily_ret, diff_ci
from .overfit import parse_id
from .triallog import log_many

KEYS = ("sharpe", "cagr", "vol", "sortino", "mdd", "mdd_peak", "mdd_trough", "mdd_recovery", "calmar", "invested",
        "n_trades", "trades_per_year", "nobuy_ratio", "win_rate", "avg_win", "avg_loss", "payoff", "expectancy",
        "profit_factor", "max_consec_losses", "top5_stock_share", "expectancy_ex_top1pct", "max_year_share",
        "pos_year_ratio", "max_sector_share", "exp_ci_lo", "exp_ci_hi", "fail_stop_high", "fail_gap", "fail_halt", "fail_lot")


def pick(m):
    return {k: m.get(k) for k in KEYS}


def main():
    c = cfg()
    ctx = Ctx()
    sel = load_json(OUT / "step10_selection.json")
    tid = sel["choice"]["id"]
    spec = parse_id(tid)
    fam, N = spec["family"], spec["N"]
    sc = pd.read_parquet(PROC / "scores" / f"{fam}_N{N}.parquet")
    cand = ctx.build_cand(sc, theta=spec["theta"], regime=spec["regime"], pred_rule=PRED_RULE.get(fam))
    out = {"id": tid}
    logs = []
    res, m = ctx.run(cand, N, spec["K"], spec["gmax"])
    out["base"] = pick(m)
    out["year_returns"] = m["year_returns"]
    out["sector_pnl"] = m.get("sector_pnl")
    for cm in (0.0, 2.0):
        _, mm = ctx.run(cand, N, spec["K"], spec["gmax"], cost_mult=cm)
        out[f"cost{cm:g}x"] = pick(mm)
        logs.append({"stage": "sensitivity_cost", "model": fam, "N": N, "K": spec["K"], "theta": spec["theta"], "gmax": spec["gmax"],
                     "regime": spec["regime"], "cost_mult": cm, "metrics": mm})
    for fee in c["trading"]["fee_sensitivity"]:
        _, mm = ctx.run(cand, N, spec["K"], spec["gmax"], fee=fee)
        out[f"fee{fee}"] = pick(mm)
        logs.append({"stage": "sensitivity_fee", "model": fam, "N": N, "K": spec["K"], "theta": spec["theta"], "gmax": spec["gmax"],
                     "regime": spec["regime"], "params": {"fee": fee}, "metrics": mm})
    for cap in c["trading"]["capital_sensitivity"]:
        _, mm = ctx.run(cand, N, spec["K"], spec["gmax"], capital=cap)
        out[f"capital{cap:.0e}"] = pick(mm)
        logs.append({"stage": "sensitivity_capital", "model": fam, "N": N, "K": spec["K"], "theta": spec["theta"], "gmax": spec["gmax"],
                     "regime": spec["regime"], "params": {"capital": cap}, "metrics": mm})
    log_many(logs)
    nav = res["nav"]
    nav.to_csv(OUT / "wf_nav_top.csv")
    r = daily_ret(nav, c["trading"]["initial_capital"])
    mon = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    mon.index.names = ["year", "month"]
    mon.unstack().to_csv(OUT / "wf_monthly_top.csv")
    di = ctx.D.date_idx
    reg = pd.DataFrame({"r": r, "above": ctx.above[[di[d] for d in r.index]], "vol_ok": ctx.vol_ok[[di[d] for d in r.index]]})
    out["regime_table"] = reg.groupby(["above", "vol_ok"])["r"].agg(
        days="size", ann_ret=lambda x: x.mean() * 252,
        sharpe=lambda x: x.mean() / x.std() * np.sqrt(252)).reset_index().to_dict(orient="records")
    # B7・B2 との比較
    b7 = pd.read_parquet(PROC / "b7_returns.parquet")[f"N{N}"]
    out["b7_diff_ci"] = diff_ci(r, b7)
    base = load_json(OUT / "step6_baselines.json")
    out["baselines_same_NK"] = base[f"N{N}_K{spec['K']}"]
    out["B2"] = base["B2"]
    b1 = np.load(OUT / f"b1_sharpes_N{N}_K{spec['K']}.npy")
    out["b1_percentile"] = float((b1 < m["sharpe"]).mean() * 100)
    b7nav = (1 + b7).cumprod()
    b7nav.to_csv(OUT / "wf_nav_b7.csv")
    # モデル別の最良（WF Sharpe）とゲート通過数
    g = pd.read_csv(OUT / "step9_grid.csv")
    best = g.sort_values("sharpe", ascending=False).groupby("family").head(1)
    out["model_best"] = best[["id", "sharpe", "cagr", "mdd", "expectancy", "exp_ci_lo", "n_trades", "b1_percentile", "b7_diff_ci_lo",
                              "G1", "G2", "G3", "G4", "G5", "G6", "G7", "pass"]].to_dict(orient="records")
    out["model_pass_counts"] = g.groupby("family")["pass"].sum().to_dict()
    out["grid_rows"] = int(len(g))
    tl = pd.read_csv(OUT / "trial_log.csv")
    out["trial_log_rows"] = int(len(tl))
    out["trial_log_by_stage"] = tl["stage"].value_counts().to_dict()
    save_json(out, OUT / "report_stats.json")
    print({k: out[k] for k in ("id", "b1_percentile", "b7_diff_ci", "trial_log_rows")})


if __name__ == "__main__":
    main()

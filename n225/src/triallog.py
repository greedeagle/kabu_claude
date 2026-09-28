"""第11章: 試行ログ。すべての評価を out/trial_log.csv に1行ずつ追記する。"""
from __future__ import annotations

import csv
import datetime as dt
import json

from .common import OUT

PATH = OUT / "trial_log.csv"
COLS = ["trial_id", "timestamp", "stage", "model", "features", "params", "N", "K", "theta", "gmax", "regime", "label",
        "cost_mult", "sharpe", "cagr", "mdd", "expectancy", "exp_ci_lo", "n_trades", "gates_pass", "metrics_json"]


def _next_id() -> int:
    if not PATH.exists():
        return 1
    with PATH.open() as fh:
        return sum(1 for _ in fh)  # ヘッダ行を含むので行数 = 次のID


def log_trial(stage, model, N=None, K=None, theta=None, gmax=None, regime=None, label=None, features=None,
              params=None, metrics=None, cost_mult=1.0, gates_pass=None):
    new = not PATH.exists()
    m = metrics or {}
    row = {"trial_id": _next_id(), "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "stage": stage,
           "model": model, "features": json.dumps(features, ensure_ascii=False) if features is not None else "",
           "params": json.dumps(params, ensure_ascii=False, default=str) if params is not None else "",
           "N": N, "K": K, "theta": theta, "gmax": gmax, "regime": regime, "label": label, "cost_mult": cost_mult,
           "sharpe": m.get("sharpe"), "cagr": m.get("cagr"), "mdd": m.get("mdd"), "expectancy": m.get("expectancy"),
           "exp_ci_lo": m.get("exp_ci_lo"), "n_trades": m.get("n_trades"), "gates_pass": gates_pass,
           "metrics_json": json.dumps(m, ensure_ascii=False, default=str)}
    with PATH.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row["trial_id"]


def log_many(rows: list[dict]):
    """大量のバックテスト結果を一括で追記する（各 dict は log_trial と同じキー）。"""
    new = not PATH.exists()
    start = _next_id()
    with PATH.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        for i, r in enumerate(rows):
            m = r.get("metrics", {})
            w.writerow({"trial_id": start + i, "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                        "stage": r["stage"], "model": r["model"],
                        "features": json.dumps(r.get("features"), ensure_ascii=False) if r.get("features") is not None else "",
                        "params": json.dumps(r.get("params"), ensure_ascii=False, default=str) if r.get("params") is not None else "",
                        "N": r.get("N"), "K": r.get("K"), "theta": r.get("theta"), "gmax": r.get("gmax"),
                        "regime": r.get("regime"), "label": r.get("label"), "cost_mult": r.get("cost_mult", 1.0),
                        "sharpe": m.get("sharpe"), "cagr": m.get("cagr"), "mdd": m.get("mdd"),
                        "expectancy": m.get("expectancy"), "exp_ci_lo": m.get("exp_ci_lo"),
                        "n_trades": m.get("n_trades"), "gates_pass": r.get("gates_pass"),
                        "metrics_json": json.dumps(m, ensure_ascii=False, default=str)})

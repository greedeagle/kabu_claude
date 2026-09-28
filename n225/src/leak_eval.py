"""第11章: 構成銘柄リークテスト（参考値）。

U(t) を「現在の日経225構成銘柄」に置き換えた別の作業ディレクトリ（KABU_WORK, KABU_UNIV=current）で
Step 1-4 を作り直し、同じ仕様（特徴量セット・ハイパーパラメータ・N・K・θ・g_max・レジーム）で WF スコアを
再計算してバックテストする。結果は out/leak_result.json に保存。

使い方（親プロセスから）: python -m src.leak_eval run <spec_id> [members(M6のみ, カンマ区切り)]
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from .common import OUT, ROOT, save_json, load_json

ALT = ROOT / "work_current_universe"
COPY = ["splits.json", "step5_adopted.json", "hp.json", "constituents.csv"]


def prepare():
    (ALT / "out").mkdir(parents=True, exist_ok=True)
    for f in COPY:
        shutil.copy(OUT / f, ALT / "out" / f)
    env = dict(os.environ, KABU_WORK=str(ALT), KABU_UNIV="current")
    if not (ALT / "data" / "proc" / "labels.parquet").exists():
        for mod, args in (("src.panel", []), ("src.features", ["--no-test"]), ("src.labels", [])):
            subprocess.run([sys.executable, "-m", mod, *args], cwd=ROOT, env=env, check=True)
    return env


def run_in_parent(spec_id: str, members=None) -> dict:
    env = prepare()
    args = [sys.executable, "-m", "src.leak_eval", "eval", spec_id] + ([",".join(members)] if members else [])
    subprocess.run(args, cwd=ROOT, env=env, check=True)
    return load_json(ALT / "out" / f"leak_result_{spec_id.replace('|', '_')}.json")


def evaluate_here(spec_id: str, members=None):
    # この関数は KABU_WORK=ALT の子プロセスで実行される
    from .grid import PRED_RULE, Ctx
    from .models import run_family
    from .overfit import parse_id
    spec = parse_id(spec_id)
    fam, N = spec["family"], spec["N"]
    if fam == "M6":
        parts = []
        for mf in members:
            s, _ = run_family(mf, N, tag="_leak")
            s["rk"] = s.groupby("Date")["score"].rank(pct=True)
            parts.append(s.set_index(["Date", "Code"])["rk"].rename(mf))
        e = pd.concat(parts, axis=1)
        sc = pd.DataFrame({"score": e.mean(axis=1), "pred": np.nan}).reset_index()
    else:
        sc, _ = run_family(fam, N, tag="_leak")
    ctx = Ctx()
    cand = ctx.build_cand(sc, theta=spec["theta"], regime=spec["regime"], pred_rule=PRED_RULE.get(fam))
    _, m = ctx.run(cand, N, spec["K"], spec["gmax"])
    _, m7 = ctx.b7(N)
    out = {"spec": spec_id, "universe": "現在の日経225構成銘柄で全期間を遡及",
           "metrics": {k: m.get(k) for k in ("sharpe", "cagr", "mdd", "expectancy", "n_trades", "win_rate")},
           "b7_current_universe": {k: m7.get(k) for k in ("sharpe", "cagr", "mdd")}}
    save_json(out, OUT / f"leak_result_{spec_id.replace('|', '_')}.json")
    print(out)


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        evaluate_here(sys.argv[2], sys.argv[3].split(",") if len(sys.argv) > 3 else None)

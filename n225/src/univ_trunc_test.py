"""Step 4 の切り詰め一致テスト（ユニバース U(t) 部分）。
無作為の20日付について、生データを t 日で切り詰めて Step 1-2 を再実行し、U(t) が全期間データからの結果と完全一致するか確認する。"""
import os, shutil, subprocess, sys, tempfile
import numpy as np, pandas as pd
from .common import OUT, PROC, ROOT, save_json

def main(n=20, seed=42):
    full = pd.read_parquet(PROC / "panel.parquet", columns=["Date", "Code", "in_univ", "ADV20", "C_u"])
    dates = np.sort(full.loc[full["in_univ"], "Date"].unique())
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(dates[len(dates) // 10:], size=n, replace=False))
    res = []
    for t in pick:
        t = pd.Timestamp(t)
        w = Path_tmp = tempfile.mkdtemp(prefix="utr_", dir=os.environ.get("TMPDIR"))
        os.makedirs(f"{w}/out", exist_ok=True)
        shutil.copy(OUT / "constituents.csv", f"{w}/out/constituents.csv")
        env = dict(os.environ, KABU_WORK=w, KABU_MAX_DATE=t.date().isoformat())
        subprocess.run([sys.executable, "-m", "src.panel"], cwd=ROOT, env=env, check=True, capture_output=True)
        tr = pd.read_parquet(f"{w}/data/proc/panel.parquet", columns=["Date", "Code", "in_univ"])
        a = set(tr.loc[(tr["Date"] == t) & tr["in_univ"], "Code"])
        b = set(full.loc[(full["Date"] == t) & full["in_univ"], "Code"])
        res.append({"date": t.date().isoformat(), "n_full": len(b), "n_trunc": len(a),
                    "only_full": sorted(b - a), "only_trunc": sorted(a - b)})
        print(res[-1]["date"], len(a), len(b), flush=True)
        shutil.rmtree(w)
    bad = [r for r in res if r["only_full"] or r["only_trunc"]]
    save_json({"n_dates": n, "n_mismatch_dates": len(bad), "results": res}, OUT / "step4_universe_truncation_test.json")
    print("universe truncation test mismatched dates:", len(bad))
    if bad:
        sys.exit("UNIVERSE TRUNCATION TEST FAILED")

if __name__ == "__main__":
    main()

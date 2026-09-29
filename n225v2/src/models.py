"""Step 7（v2）: モデル候補（フォールドごとに再学習し、検証期間のスコアを出力）。

目的変数は LM（5・10・20 日の相対リターン順位の平均）で、保有期間 N に依存しない。
同じスコアを N=5,10,20 のバックテストに共通で使う（scores/<fam>_N<N>.parquet に同じ内容を保存）。
LightGBM は model_seeds の予測平均（シード・アンサンブル）。
score: 大きいほど良い。pred: 使わない（NaN）。
"""
from __future__ import annotations

import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from .common import OUT, PROC, cfg, load_json, save_json
from .cv import load_splits, purge_check, train_end_for
from .ic import MARKET_COLS
from .triallog import log_trial

warnings.filterwarnings("ignore")
SCORES = PROC / "scores"
SCORES.mkdir(exist_ok=True)
PER_DAY_TRAIN = 300   # 学習時の1日あたりサンプル上限（計算時間のため。無作為抽出、seed固定）
FAMILIES = ["M2r", "M2b", "M4r", "M5"]
KIND = {"M4r": "reg", "M5": "rank"}


def horizon() -> int:
    return max(cfg()["split"]["label_horizons"])


class Data:
    def __init__(self, N: int | None = None, include_holdout: bool = False):
        c = cfg()
        self.N = N or horizon()
        self.H = horizon()          # パージは目的変数の最長ホライズンで行う
        self.sp = load_splits()
        self.adopt = load_json(OUT / "step5_adopted.json")[str(c["split"]["n_candidates"][0])]
        self.fcols = self.adopt["features"]
        self.mcols = MARKET_COLS
        self.m2r = c["m2r_features"]
        cols = ["Date", "Code"] + sorted(set(self.fcols) | set(self.mcols) | set(self.m2r))
        feat = pd.read_parquet(PROC / "features.parquet", columns=cols)
        lab = pd.read_parquet(PROC / "labels.parquet", columns=["Date", "Code", "ADV20", "LM"])
        df = feat.merge(lab, on=["Date", "Code"], how="left")
        start = pd.Timestamp(self.sp["analysis_start"])
        lim = pd.Timestamp(self.sp["last_date"]) if include_holdout else pd.Timestamp(self.sp["dev_end_signal"])
        self.df = df[(df["Date"] >= start) & (df["Date"] <= lim)].sort_values(["Date", "Code"]).reset_index(drop=True)

    def train(self, val_start, seed=42):
        te = train_end_for(pd.Timestamp(val_start), self.H)
        d = self.df[(self.df["Date"] <= te) & self.df["LM"].notna()]
        assert purge_check(d["Date"], val_start, self.H), "purge violated"
        return self._subsample(d, seed)

    def train_until(self, te, seed=42):
        """最終学習用: シグナル日 <= te かつラベルが確定している行。"""
        d = self.df[(self.df["Date"] <= pd.Timestamp(te)) & self.df["LM"].notna()]
        return self._subsample(d, seed)

    @staticmethod
    def _subsample(d, seed):
        rng = np.random.default_rng(seed)
        rk = pd.Series(rng.random(len(d)), index=d.index).groupby(d["Date"]).rank(method="first")
        return d[rk <= PER_DAY_TRAIN]

    def val(self, a, b):
        return self.df[(self.df["Date"] >= pd.Timestamp(a)) & (self.df["Date"] <= pd.Timestamp(b))]

    def Xm(self, d):
        return d[self.fcols + self.mcols].astype(np.float32).to_numpy()


def _daily_ic(d: pd.DataFrame, pred: np.ndarray, lab: str = "LM") -> float:
    t = pd.DataFrame({"Date": d["Date"].to_numpy(), "p": pred, "y": d[lab].to_numpy()}).dropna()
    ic = t.groupby("Date").apply(lambda g: g["p"].rank().corr(g["y"].rank()) if len(g) > 20 else np.nan)
    return float(ic.mean())


def _inner_split(d: pd.DataFrame, H: int, frac=0.8):
    dates = np.sort(d["Date"].unique())
    cut = int(len(dates) * frac)
    tr_end = dates[max(0, cut - H - 2)]
    return d[d["Date"] <= tr_end], d[d["Date"] >= dates[cut]]


# ---------------- M2 スコアリング ----------------
def score_m2r(d, D):
    """短期リバーサル4指標（向きを揃えた日次順位）の平均。学習しない。"""
    parts = [(d[f] if s > 0 else 1 - d[f]).to_numpy(dtype=float) for f, s in D.m2r.items()]
    return np.nanmean(np.vstack(parts), axis=0)


def fold_ic_sign(tr, D):
    return {f: (1.0 if _daily_ic(tr, tr[f].to_numpy()) >= 0 else -1.0) for f in D.fcols}


def score_m2(d, D, sign):
    X = d[D.fcols].to_numpy(dtype=float)
    s = np.array([sign[f] for f in D.fcols])
    return np.nanmean(np.where(s > 0, X, 1 - X), axis=1)


# ---------------- LightGBM 系 ----------------
def lgb_params(kind, hp, seed):
    base = {"verbosity": -1, "num_threads": 3, "max_bin": 63, "seed": seed, "bagging_seed": seed,
            "feature_fraction_seed": seed, "deterministic": True, "force_row_wise": True,
            "bagging_fraction": 0.8, "bagging_freq": 1}
    base.update(hp)
    if kind == "reg":
        base.update({"objective": "regression", "metric": "l2"})
    else:
        base.update({"objective": "lambdarank", "metric": "ndcg", "eval_at": [20]})
    return base


def _target(d, kind):
    if kind == "reg":
        return d["LM"].to_numpy()
    r = d.groupby("Date")["LM"].rank(pct=True)
    return np.minimum((r * 10).astype(int), 9).to_numpy()


def fit_lgb(tr, D, kind, hp, seed=42, max_rounds=1000):
    """学習期間の内側で時系列分割（パージつき）し、early stopping。"""
    a, b = _inner_split(tr, D.H)
    p = lgb_params(kind, hp, seed)
    grp = (lambda d: d.groupby("Date", sort=True).size().to_numpy()) if kind == "rank" else (lambda d: None)
    da = lgb.Dataset(D.Xm(a), _target(a, kind), group=grp(a), free_raw_data=True)
    db = lgb.Dataset(D.Xm(b), _target(b, kind), group=grp(b), reference=da)
    return lgb.train(p, da, num_boost_round=max_rounds, valid_sets=[db], callbacks=[lgb.early_stopping(50, verbose=False)])


def fit_ens(tr, D, kind, hp, seeds):
    return [fit_lgb(tr, D, kind, hp, s) for s in seeds]


def pred_ens(ms, d, D):
    X = D.Xm(d)
    return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in ms], axis=0)


def sample_hps(n=5, seed=42):
    sp = cfg()["lgb_space"]
    rng = np.random.default_rng(seed)
    return [{"num_leaves": int(rng.integers(sp["num_leaves"][0], sp["num_leaves"][1] + 1)),
             "learning_rate": float(np.exp(rng.uniform(np.log(sp["learning_rate"][0]), np.log(sp["learning_rate"][1])))),
             "min_data_in_leaf": int(rng.integers(sp["min_data_in_leaf"][0], sp["min_data_in_leaf"][1] + 1)),
             "feature_fraction": float(rng.uniform(*sp["feature_fraction"]))} for _ in range(n)]


def tune_all():
    """最初のフォールドの学習期間内で探索（内側の検証は学習期間の最後20%、パージつき）。単一シード。"""
    c = cfg()
    hp = load_json(OUT / "hp.json") if (OUT / "hp.json").exists() else {}
    D = Data()
    tr = D.train(D.sp["folds"][0]["val_start"])
    a, b = _inner_split(tr, D.H)
    for kind in ("reg", "rank"):
        if kind in hp:
            continue
        best, best_v = None, -np.inf
        for h in sample_hps(c.get("hp_trials_per_family", 5)):
            t0 = time.time()
            m = fit_lgb(a, D, kind, h, seed=c["seed"])
            v = _daily_ic(b, m.predict(D.Xm(b), num_iteration=m.best_iteration))
            log_trial(stage="hp_tune", model="M4r" if kind == "reg" else "M5", label="LM", features=D.fcols + D.mcols,
                      params={**h, "best_iter": m.best_iteration}, metrics={"inner_ic": v, "sec": time.time() - t0})
            if v > best_v:
                best, best_v = h, v
        hp[kind] = {**best, "inner_ic": best_v}
        save_json(hp, OUT / "hp.json")
        print("tuned", kind, hp[kind], flush=True)


def _hp(kind):
    return {k: v for k, v in load_json(OUT / "hp.json")[kind].items() if k != "inner_ic"}


def fit_model(fam: str, tr: pd.DataFrame, D: Data, seeds=None):
    """固定済みの仕様で1つのモデルを学習する。"""
    seeds = seeds or cfg()["model_seeds"]
    if fam == "M2r":
        return None
    if fam == "M2b":
        return fold_ic_sign(tr, D)
    return fit_ens(tr, D, KIND[fam], _hp(KIND[fam]), seeds)


def score_with(fam, mdl, d, D):
    if fam == "M2r":
        return score_m2r(d, D)
    if fam == "M2b":
        return score_m2(d, D, mdl)
    return pred_ens(mdl, d, D)


def run_family(fam: str, seeds=None, shuffle_labels=False, tag="", folds=None, return_models=False):
    D = Data()
    parts, meta, models = [], [], {}
    for f in (folds or D.sp["folds"]):
        tr = D.train(f["val_start"])
        if shuffle_labels:
            rng = np.random.default_rng(42)
            tr = tr.copy()
            tr["LM"] = tr.groupby("Date")["LM"].transform(
                lambda s: s.sample(frac=1.0, random_state=int(rng.integers(1e9))).to_numpy())
        va = D.val(f["val_start"], f["val_end"])
        mdl = fit_model(fam, tr, D, seeds)
        score = score_with(fam, mdl, va, D)
        info = {"fold": f["fold"], "train_rows": len(tr)}
        if fam == "M2b":
            info["sign"] = mdl
        if fam in KIND:
            info["importance"] = sum(pd.Series(m.feature_importance("gain"), index=D.fcols + D.mcols) for m in mdl).to_dict()
            info["best_iter"] = [m.best_iteration for m in mdl]
        if return_models:
            models[f["fold"]] = mdl
        parts.append(pd.DataFrame({"Date": va["Date"].to_numpy(), "Code": va["Code"].to_numpy(),
                                   "score": score, "pred": np.nan, "fold": f["fold"]}))
        meta.append(info)
        print(fam, tag, "fold", f["fold"], "train_rows", len(tr), flush=True)
    out = pd.concat(parts, ignore_index=True)
    if not tag:
        for N in cfg()["split"]["n_candidates"]:
            out.to_parquet(SCORES / f"{fam}_N{N}.parquet")
            save_json(meta, SCORES / f"{fam}_N{N}_meta.json")
    return (out, meta, models) if return_models else (out, meta)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "tune":
        tune_all()
    else:
        for fam in what.split(","):
            if (SCORES / f"{fam}_N{cfg()['split']['n_candidates'][0]}.parquet").exists():
                continue
            t0 = time.time()
            run_family(fam)
            print("done", fam, f"{time.time() - t0:.0f}s", flush=True)

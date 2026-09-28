"""Step 7: モデル候補（フォールドごとに再学習し、検証期間のスコアを出力）。

score: 大きいほど良い。pred: (b) 条件の判定に使う予測値（回帰=予測L3、分類=確率）。
"""
from __future__ import annotations

import itertools
import json
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .common import OUT, PROC, cfg, load_json, save_json
from .cv import calendar, load_splits, purge_check, train_end_for
from .ic import MARKET_COLS
from .triallog import log_trial

warnings.filterwarnings("ignore")
SCORES = PROC / "scores"
SCORES.mkdir(exist_ok=True)
PER_DAY_TRAIN = 300   # 学習時の1日あたりサンプル上限（計算時間のため。無作為抽出、seed固定）


class Data:
    def __init__(self, N: int, include_holdout: bool = False):
        c = cfg()
        self.N = N
        self.sp = load_splits()
        self.adopt = load_json(OUT / "step5_adopted.json")[str(N)]
        self.fcols = self.adopt["features"]
        self.mcols = MARKET_COLS
        cols = ["Date", "Code"] + sorted(set(self.fcols) | set(self.mcols))
        feat = pd.read_parquet(PROC / "features.parquet", columns=cols)
        lab = pd.read_parquet(PROC / "labels.parquet", columns=["Date", "Code", "ADV20", f"L1_{N}", f"L3_{N}", f"L4_{N}"])
        df = feat.merge(lab, on=["Date", "Code"], how="left")
        start = pd.Timestamp(self.sp["analysis_start"])
        lim = pd.Timestamp(self.sp["last_date"]) if include_holdout else pd.Timestamp(self.sp["dev_end_signal"])
        self.df = df[(df["Date"] >= start) & (df["Date"] <= lim)].sort_values(["Date", "Code"]).reset_index(drop=True)

    def train(self, val_start, seed=42, full=False):
        te = train_end_for(pd.Timestamp(val_start), self.N)
        d = self.df[(self.df["Date"] <= te) & self.df[f"L3_{self.N}"].notna()]
        assert purge_check(d["Date"], val_start, self.N), "purge violated"
        if full:
            return d
        rng = np.random.default_rng(seed)
        key = rng.random(len(d))
        rk = pd.Series(key, index=d.index).groupby(d["Date"]).rank(method="first")
        return d[rk <= PER_DAY_TRAIN]

    def train_until(self, te, seed=42):
        """最終学習用: シグナル日 <= te かつラベルが確定している行（t+1+N <= 最新日）。"""
        d = self.df[(self.df["Date"] <= pd.Timestamp(te)) & self.df[f"L3_{self.N}"].notna()]
        rng = np.random.default_rng(seed)
        rk = pd.Series(rng.random(len(d)), index=d.index).groupby(d["Date"]).rank(method="first")
        return d[rk <= PER_DAY_TRAIN]

    def val(self, a, b):
        return self.df[(self.df["Date"] >= pd.Timestamp(a)) & (self.df["Date"] <= pd.Timestamp(b))]

    def X(self, d):
        x = d[self.fcols].astype(np.float32).to_numpy() - 0.5
        return np.nan_to_num(x, nan=0.0)

    def Xm(self, d):
        return d[self.fcols + self.mcols].astype(np.float32).to_numpy()


def _daily_ic(d: pd.DataFrame, pred: np.ndarray, lab: str) -> float:
    t = pd.DataFrame({"Date": d["Date"].to_numpy(), "p": pred, "y": d[lab].to_numpy()}).dropna()
    ic = t.groupby("Date").apply(lambda g: g["p"].rank().corr(g["y"].rank()) if len(g) > 20 else np.nan)
    return float(ic.mean())


def _inner_split(d: pd.DataFrame, N: int, frac=0.8):
    dates = np.sort(d["Date"].unique())
    cut = int(len(dates) * frac)
    tr_end = dates[max(0, cut - N - 2)]
    return d[d["Date"] <= tr_end], d[d["Date"] >= dates[cut]]


# ---------------- M1 ルール ----------------
def fit_m1(tr: pd.DataFrame, D: Data, top=6, qs=(0.1, 0.2, 0.3)):
    ir = D.adopt["ic_ir"]
    feats = sorted(D.fcols, key=lambda f: -abs(ir[f]))[:top]
    sign = D.adopt["sign"]
    O = {f: (tr[f] if sign[f] > 0 else 1 - tr[f]) for f in feats}
    lab = tr[f"L3_{D.N}"]
    best, best_v = None, -np.inf
    for k in (1, 2, 3):
        for comb in itertools.combinations(feats, k):
            for q in qs:
                m = np.ones(len(tr), bool)
                for f in comb:
                    m &= (O[f] >= 1 - q).to_numpy()
                if m.sum() < max(1000, 0.002 * len(tr)):
                    continue
                v = lab[m].groupby(tr["Date"][m]).mean().mean()
                if v > best_v:
                    best, best_v = (comb, q), v
    return {"features": list(best[0]), "q": best[1], "train_mean_L3": float(best_v), "sign": {f: sign[f] for f in best[0]}}


def score_m1(rule, d):
    m = np.ones(len(d), bool)
    parts = []
    for f in rule["features"]:
        o = d[f] if rule["sign"][f] > 0 else 1 - d[f]
        m &= (o >= 1 - rule["q"]).to_numpy()
        parts.append(o.to_numpy())
    s = np.nanmean(np.vstack(parts), axis=0)
    return np.where(m, s, np.nan)


# ---------------- M2 スコアリング ----------------
def fold_ic_sign(tr, D):
    out = {}
    for f in D.fcols:
        ic = _daily_ic(tr, tr[f].to_numpy(), f"L3_{D.N}")
        out[f] = 1.0 if ic >= 0 else -1.0
    return out


def score_m2(d, D, sign=None):
    X = d[D.fcols].to_numpy(dtype=float)
    if sign is not None:
        s = np.array([sign[f] for f in D.fcols])
        X = np.where(s > 0, X, 1 - X)
    return np.nanmean(X, axis=1)


# ---------------- M3 ロジスティック ----------------
def fit_m3(tr, D, C=0.1, seed=42):
    y = tr[f"L4_{D.N}"].to_numpy()
    ok = np.isfinite(y)
    sc = StandardScaler().fit(tr.loc[ok, D.mcols].fillna(0).to_numpy())
    X = np.hstack([D.X(tr[ok]), sc.transform(tr.loc[ok, D.mcols].fillna(0).to_numpy())])
    m = LogisticRegression(C=C, penalty="l2", max_iter=300, random_state=seed).fit(X, y[ok].astype(int))
    return m, sc


def pred_m3(model, d, D):
    m, sc = model
    X = np.hstack([D.X(d), sc.transform(d[D.mcols].fillna(0).to_numpy())])
    return m.predict_proba(X)[:, 1]


# ---------------- LightGBM 系 ----------------
def lgb_params(kind, hp, seed):
    base = {"verbosity": -1, "num_threads": 4, "max_bin": 63, "seed": seed, "bagging_seed": seed,
            "feature_fraction_seed": seed, "deterministic": True, "force_row_wise": True,
            "bagging_fraction": 0.8, "bagging_freq": 1}
    base.update(hp)
    if kind == "reg":
        base.update({"objective": "regression", "metric": "l2"})
    elif kind == "clf":
        base.update({"objective": "binary", "metric": "binary_logloss"})
    elif kind == "rank":
        base.update({"objective": "lambdarank", "metric": "ndcg", "eval_at": [20]})
    return base


def _target(d, kind, N):
    if kind == "reg":
        return d[f"L3_{N}"].to_numpy()
    if kind == "clf":
        return d[f"L4_{N}"].to_numpy()
    r = d.groupby("Date")[f"L3_{N}"].rank(pct=True)
    return np.minimum((r * 10).astype(int), 9).to_numpy()


def _groups(d):
    return d.groupby("Date", sort=True).size().to_numpy()


def fit_lgb(tr, D, kind, hp, seed=42, max_rounds=1000):
    """学習期間の内側で時系列分割（パージつき）し、early stopping。"""
    a, b = _inner_split(tr, D.N)
    a = a[np.isfinite(_target(a, kind, D.N))]
    b = b[np.isfinite(_target(b, kind, D.N))]
    p = lgb_params(kind, hp, seed)
    da = lgb.Dataset(D.Xm(a), _target(a, kind, D.N), group=_groups(a) if kind == "rank" else None, free_raw_data=True)
    db = lgb.Dataset(D.Xm(b), _target(b, kind, D.N), group=_groups(b) if kind == "rank" else None, reference=da)
    m = lgb.train(p, da, num_boost_round=max_rounds, valid_sets=[db], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def pred_lgb(m, d, D):
    return m.predict(D.Xm(d), num_iteration=m.best_iteration)


def sample_hps(n=6, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append({"num_leaves": int(rng.integers(7, 64)),
                    "learning_rate": float(np.exp(rng.uniform(np.log(0.03), np.log(0.1)))),
                    "min_data_in_leaf": int(rng.integers(200, 2001)),
                    "feature_fraction": float(rng.uniform(0.5, 0.9))})
    return out


def tune(D: Data, kind: str, n=6):
    """最初のフォールドの学習期間内で探索（内側の検証は学習期間の最後20%、パージつき）。"""
    v0 = D.sp["folds"][0]["val_start"]
    tr = D.train(v0)
    a, b = _inner_split(tr, D.N)
    best, best_v = None, -np.inf
    for i, hp in enumerate(sample_hps(n)):
        t0 = time.time()
        m = fit_lgb(a, D, kind, hp)
        v = _daily_ic(b, pred_lgb(m, b, D), f"L3_{D.N}")
        log_trial(stage="hp_tune", model=f"M4{kind}" if kind != "rank" else "M5", N=D.N, label="L3" if kind != "clf" else "L4",
                  features=D.fcols + D.mcols, params={**hp, "best_iter": m.best_iteration}, metrics={"inner_ic": v, "sec": time.time() - t0})
        if v > best_v:
            best, best_v = hp, v
    return best


# ---------------- M7 レジーム別 ----------------
def regimes(d, vol_med):
    a = d["mk_above200"].fillna(0).to_numpy() > 0.5
    h = d["mk_vol20"].to_numpy() > vol_med
    return a.astype(int) * 2 + h.astype(int)


def fit_m7(tr, D, hp, seed=42):
    vol_med = float(tr.drop_duplicates("Date")["mk_vol20"].median())
    for scheme in ("4", "2"):
        rg = regimes(tr, vol_med) if scheme == "4" else (tr["mk_above200"].fillna(0).to_numpy() > 0.5).astype(int)
        share = pd.Series(rg).value_counts(normalize=True)
        if len(share) == (4 if scheme == "4" else 2) and share.min() >= 0.15:
            models = {}
            for r in sorted(share.index):
                models[r] = fit_lgb(tr[rg == r], D, "reg", hp, seed)
            return {"scheme": scheme, "vol_med": vol_med, "models": models}
    return None


def pred_m7(M, d, D):
    rg = regimes(d, M["vol_med"]) if M["scheme"] == "4" else (d["mk_above200"].fillna(0).to_numpy() > 0.5).astype(int)
    out = np.full(len(d), np.nan)
    for r, m in M["models"].items():
        k = rg == r
        if k.any():
            out[k] = pred_lgb(m, d[k], D)
    return out


# ---------------- 実行 ----------------
FAMILIES = ["M1", "M2a", "M2b", "M3", "M4r", "M4c", "M5", "M7"]


def run_family(fam: str, N: int, seed=42, hp=None, shuffle_labels=False, tag="", folds=None, return_models=False):
    D = Data(N)
    if fam in ("M4r", "M7") and hp is None:
        hp = load_json(OUT / "hp.json")[f"reg_{N}"]
    if fam == "M4c" and hp is None:
        hp = load_json(OUT / "hp.json")[f"clf_{N}"]
    if fam == "M5" and hp is None:
        hp = load_json(OUT / "hp.json")[f"rank_{N}"]
    parts, meta = [], []
    models = {}
    for f in (folds or D.sp["folds"]):
        tr = D.train(f["val_start"], seed=seed)
        if shuffle_labels:
            rng = np.random.default_rng(seed)
            tr = tr.copy()
            for col in (f"L1_{N}", f"L3_{N}", f"L4_{N}"):
                tr[col] = tr.groupby("Date")[col].transform(lambda s: s.sample(frac=1.0, random_state=int(rng.integers(1e9))).to_numpy())
        va = D.val(f["val_start"], f["val_end"])
        info = {"fold": f["fold"]}
        pred = np.full(len(va), np.nan)
        if fam == "M1":
            rule = fit_m1(tr, D)
            score = score_m1(rule, va)
            info.update(rule)
            mdl = rule
        elif fam == "M2a":
            score = score_m2(va, D)
            mdl = None
        elif fam == "M2b":
            sg = fold_ic_sign(tr, D)
            score = score_m2(va, D, sg)
            info["sign"] = sg
            mdl = sg
        elif fam == "M3":
            C = load_json(OUT / "hp.json")[f"m3_{N}"]["C"]
            mdl = fit_m3(tr, D, C=C, seed=seed)
            score = pred = pred_m3(mdl, va, D)
        elif fam in ("M4r", "M4c", "M5"):
            kind = {"M4r": "reg", "M4c": "clf", "M5": "rank"}[fam]
            mdl = fit_lgb(tr, D, kind, hp, seed)
            score = pred_lgb(mdl, va, D)
            pred = score if kind != "rank" else pred
            imp = pd.Series(mdl.feature_importance("gain"), index=D.fcols + D.mcols)
            info["importance"] = imp.to_dict()
            info["best_iter"] = mdl.best_iteration
        elif fam == "M7":
            mdl = fit_m7(tr, D, hp, seed)
            if mdl is None:
                info["rejected"] = "regime share < 15%"
                meta.append(info)
                continue
            score = pred = pred_m7(mdl, va, D)
            info["scheme"] = mdl["scheme"]
            imp = sum(pd.Series(m.feature_importance("gain"), index=D.fcols + D.mcols) for m in mdl["models"].values())
            info["importance"] = imp.to_dict()
        if return_models:
            models[f["fold"]] = mdl
        parts.append(pd.DataFrame({"Date": va["Date"].to_numpy(), "Code": va["Code"].to_numpy(),
                                   "score": score, "pred": pred, "fold": f["fold"]}))
        meta.append(info)
        print(fam, N, "fold", f["fold"], "train_rows", len(tr), flush=True)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["Date", "Code", "score", "pred", "fold"])
    name = f"{fam}_N{N}{tag}"
    if not tag:
        out.to_parquet(SCORES / f"{name}.parquet")
        save_json(meta, SCORES / f"{name}_meta.json")
    return (out, meta, models) if return_models else (out, meta)


def tune_all():
    c = cfg()
    hp = load_json(OUT / "hp.json") if (OUT / "hp.json").exists() else {}
    for N in c["split"]["n_candidates"]:
        D = Data(N)
        for kind in ("reg", "clf", "rank"):
            k = f"{kind}_{N}"
            if k not in hp:
                hp[k] = tune(D, kind, n=c.get("hp_trials_per_family", 6))
                save_json(hp, OUT / "hp.json")
        k = f"m3_{N}"
        if k not in hp:
            tr = D.train(D.sp["folds"][0]["val_start"])
            a, b = _inner_split(tr, N)
            best, bv = None, -np.inf
            for C in (0.01, 0.1, 1.0):
                m = fit_m3(a, D, C=C)
                v = _daily_ic(b, pred_m3(m, b, D), f"L3_{N}")
                log_trial(stage="hp_tune", model="M3", N=N, label="L4", features=D.fcols + D.mcols, params={"C": C}, metrics={"inner_ic": v})
                if v > bv:
                    best, bv = C, v
            hp[k] = {"C": best}
            save_json(hp, OUT / "hp.json")
        print("tuned N", N, {k: v for k, v in hp.items() if k.endswith(f"_{N}")}, flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "tune":
        tune_all()
    else:
        fams = what.split(",")
        Ns = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else cfg()["split"]["n_candidates"]
        for N in Ns:
            for fam in fams:
                if (SCORES / f"{fam}_N{N}.parquet").exists():
                    continue
                t0 = time.time()
                run_family(fam, N)
                print("done", fam, N, f"{time.time() - t0:.0f}s", flush=True)


def fit_model(fam: str, tr: pd.DataFrame, D: Data, seed=42):
    """固定済みの仕様で1つのモデルを学習する（ホールドアウト・最終学習用）。"""
    hp = load_json(OUT / "hp.json")
    N = D.N
    if fam == "M1":
        return fit_m1(tr, D)
    if fam == "M2a":
        return None
    if fam == "M2b":
        return fold_ic_sign(tr, D)
    if fam == "M3":
        return fit_m3(tr, D, C=hp[f"m3_{N}"]["C"], seed=seed)
    if fam == "M4r":
        return fit_lgb(tr, D, "reg", hp[f"reg_{N}"], seed)
    if fam == "M4c":
        return fit_lgb(tr, D, "clf", hp[f"clf_{N}"], seed)
    if fam == "M5":
        return fit_lgb(tr, D, "rank", hp[f"rank_{N}"], seed)
    if fam == "M7":
        return fit_m7(tr, D, hp[f"reg_{N}"], seed)
    raise ValueError(fam)

"""第9章: 評価指標、第10章: 合格ゲート、DSR、PBO。無リスク金利 = 0。"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

ANN = 252


def daily_returns(nav: pd.Series, capital: float) -> pd.Series:
    prev = nav.shift(1)
    prev.iloc[0] = capital
    return nav / prev - 1


def sharpe(r) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1) if len(r) > 1 else np.nan
    return float(r.mean() / sd * np.sqrt(ANN)) if sd and sd > 0 else np.nan


def max_drawdown(nav: pd.Series):
    peak = nav.cummax()
    dd = nav / peak - 1
    mdd = float(dd.min())
    trough = dd.idxmin()
    pk = nav.loc[:trough].idxmax()
    rec = nav.loc[trough:][nav.loc[trough:] >= nav.loc[pk]]
    end = rec.index[0] if len(rec) else nav.index[-1]
    return mdd, pk, trough, end


def block_bootstrap_mean(trades: pd.DataFrame, dates: pd.DatetimeIndex, block=20, n=1000, seed=42):
    """取引の期待値（平均リターン）の95% CI。エントリー日の20営業日ブロック単位で復元抽出。"""
    if len(trades) < 2:
        return np.nan, np.nan
    b = (trades["buy_d"].to_numpy() // block)
    ub, inv = np.unique(b, return_inverse=True)
    sums = np.bincount(inv, weights=trades["ret"].to_numpy())
    cnts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ub), size=(n, len(ub)))
    means = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(res: dict, dates: pd.DatetimeIndex, capital: float, codes=None, boot=True) -> dict:
    nav = res["nav"].dropna()
    tr = res["trades"]
    r = daily_returns(nav, capital)
    n = len(r)
    years = n / ANN
    total = float(nav.iloc[-1] / capital - 1) if n else np.nan
    cagr = float((nav.iloc[-1] / capital) ** (1 / years) - 1) if n and nav.iloc[-1] > 0 else -1.0
    vol = float(r.std() * np.sqrt(ANN))
    dn = r[r < 0]
    sortino = float(r.mean() / np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(ANN)) if len(dn) else np.nan
    mdd, pk, trough, rec = max_drawdown(nav) if n else (np.nan, None, None, None)
    m = {"days": n, "total_return": total, "cagr": cagr, "vol": vol, "sharpe": sharpe(r), "sortino": sortino,
         "mdd": mdd, "mdd_peak": str(pk.date()) if pk is not None else None,
         "mdd_trough": str(trough.date()) if trough is not None else None,
         "mdd_recovery": str(rec.date()) if rec is not None else None,
         "calmar": cagr / abs(mdd) if mdd and mdd < 0 else np.nan,
         "invested": float(res["invested"].mean()),
         "n_trades": len(tr), "trades_per_year": len(tr) / years if years else np.nan,
         "nobuy_ratio": res["nobuy_days"] / max(res["signal_days"], 1)}
    m.update({f"fail_{k}": v for k, v in res["fails"].items()})
    if len(tr):
        ret = tr["ret"]
        win, loss = ret[ret > 0], ret[ret <= 0]
        m.update({"win_rate": float((ret > 0).mean()), "avg_win": float(win.mean()) if len(win) else np.nan,
                  "avg_loss": float(loss.mean()) if len(loss) else np.nan,
                  "expectancy": float(ret.mean()),
                  "profit_factor": float(tr.loc[tr.pnl > 0, "pnl"].sum() / abs(tr.loc[tr.pnl <= 0, "pnl"].sum())) if (tr.pnl <= 0).any() else np.inf})
        m["payoff"] = m["avg_win"] / abs(m["avg_loss"]) if m["avg_loss"] and m["avg_loss"] < 0 else np.nan
        seq = (tr.sort_values("exit_d")["ret"] <= 0).to_numpy()
        best = cur = 0
        for x in seq:
            cur = cur + 1 if x else 0
            best = max(best, cur)
        m["max_consec_losses"] = best
        total_pnl = tr["pnl"].sum()
        by_stock = tr.groupby("stock")["pnl"].sum().sort_values(ascending=False)
        m["top5_stock_share"] = float(by_stock.head(5).sum() / total_pnl) if total_pnl > 0 else np.nan
        cut = max(1, int(np.ceil(len(tr) * 0.01)))
        rest = tr.sort_values("pnl", ascending=False).iloc[cut:]
        m["expectancy_ex_top1pct"] = float(rest["ret"].mean()) if len(rest) else np.nan
        if boot:
            lo, hi = block_bootstrap_mean(tr, dates)
            m["exp_ci_lo"], m["exp_ci_hi"] = lo, hi
        # 年別の取引数（エントリー年）。部分年は営業日数で按分した基準を使う
        ey = pd.Series(dates[tr["buy_d"].to_numpy()].year)
        m["trades_by_year"] = ey.value_counts().sort_index().to_dict()
    else:
        m.update({"expectancy": np.nan, "win_rate": np.nan, "top5_stock_share": np.nan, "expectancy_ex_top1pct": np.nan,
                  "exp_ci_lo": np.nan, "exp_ci_hi": np.nan, "trades_by_year": {}, "profit_factor": np.nan,
                  "max_consec_losses": 0, "avg_win": np.nan, "avg_loss": np.nan, "payoff": np.nan})
    yv = nav.groupby(nav.index.year).last()
    y0 = pd.concat([pd.Series([capital], index=[yv.index[0] - 1]), yv]) if n else yv
    ypnl = y0.diff().dropna()
    m["year_returns"] = (yv / y0.shift(1).dropna().to_numpy() - 1).to_dict() if n else {}
    m["year_days"] = nav.groupby(nav.index.year).size().to_dict()
    m["pos_year_ratio"] = float((ypnl > 0).mean()) if len(ypnl) else np.nan
    tot = ypnl.sum()
    m["max_year_share"] = float(ypnl.max() / tot) if tot > 0 else np.nan
    return m


def gates(m: dict, m_cost2: dict, b1_p95: float, b_best: float, g: dict) -> dict:
    tby = m.get("trades_by_year", {})
    ydays = m.get("year_days", {})
    per_year_ok = all(cnt >= g["G1_min_trades_per_year"] * min(1.0, ydays.get(y, ANN) / ANN) for y, cnt in tby.items())
    out = {
        "G1": bool(m["n_trades"] >= g["G1_min_trades"] and per_year_ok and len(tby) > 0),
        "G2": bool(np.isfinite(m.get("exp_ci_lo", np.nan)) and m["exp_ci_lo"] > g["G2_ci_lower_gt"]),
        "G3": bool(np.isfinite(m["sharpe"]) and m["sharpe"] > b1_p95 and m["sharpe"] > b_best),
        "G4": bool(np.isfinite(m["pos_year_ratio"]) and m["pos_year_ratio"] >= g["G4_pos_year_ratio"]
                   and np.isfinite(m["max_year_share"]) and m["max_year_share"] <= g["G4_max_year_share"]),
        "G5": bool(np.isfinite(m["top5_stock_share"]) and m["top5_stock_share"] <= g["G5_top5_share_max"]
                   and np.isfinite(m["expectancy_ex_top1pct"]) and m["expectancy_ex_top1pct"] > g["G5_ex_top1pct_expectancy_gt"]),
        "G6": bool(np.isfinite(m_cost2.get("expectancy", np.nan)) and m_cost2["expectancy"] > g["G6_cost2x_expectancy_gt"]),
        "G7": bool(np.isfinite(m["mdd"]) and m["mdd"] >= g["G7_mdd_floor"]),
    }
    out["pass"] = all(out.values())
    return out


# ---- 多重検定 ----
def deflated_sharpe(sr_daily: float, n_obs: int, skew: float, kurt: float, trial_srs_daily: np.ndarray) -> float:
    """Bailey & López de Prado (2014)。SR は日次（非年率）。kurt は通常の尖度（正規=3）。"""
    trial_srs_daily = trial_srs_daily[np.isfinite(trial_srs_daily)]
    Ntr = len(trial_srs_daily)
    if Ntr < 2 or not np.isfinite(sr_daily):
        return np.nan
    v = trial_srs_daily.var(ddof=1)
    em = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - em) * stats.norm.ppf(1 - 1 / Ntr) + em * stats.norm.ppf(1 - 1 / (Ntr * np.e)))
    den = np.sqrt(1 - skew * sr_daily + (kurt - 1) / 4 * sr_daily ** 2)
    return float(stats.norm.cdf((sr_daily - sr0) * np.sqrt(n_obs - 1) / den))


def pbo_cscv(R: np.ndarray, S: int = 16) -> float:
    """R: T×M の日次リターン行列（列 = 試行）。CSCV による過学習確率。"""
    T, M = R.shape
    R = np.nan_to_num(R)
    bounds = np.linspace(0, T, S + 1).astype(int)
    s1 = np.array([R[bounds[i]:bounds[i + 1]].sum(0) for i in range(S)])
    s2 = np.array([(R[bounds[i]:bounds[i + 1]] ** 2).sum(0) for i in range(S)])
    cnt = np.diff(bounds).astype(float)
    logits = []
    for comb in itertools.combinations(range(S), S // 2):
        ins = np.zeros(S, bool)
        ins[list(comb)] = True

        def sr(mask):
            n = cnt[mask].sum()
            mu = s1[mask].sum(0) / n
            var = s2[mask].sum(0) / n - mu ** 2
            return mu / np.sqrt(np.maximum(var, 1e-18))
        sr_in, sr_out = sr(ins), sr(~ins)
        best = int(np.argmax(sr_in))
        rank = stats.rankdata(sr_out)[best] / (M + 1)
        logits.append(math.log(rank / (1 - rank)))
    logits = np.array(logits)
    return float((logits <= 0).mean())

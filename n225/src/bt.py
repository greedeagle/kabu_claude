"""第8章: バックテスト（トランシェ方式、寄り付き約定）。

時間軸: シグナル日 t（dense のインデックス）→ 買い d=t+1 の寄り付き → 売り d+N の寄り付き。
cand[t, :] は t 日の情報だけで確定した購入候補（スコア順、-1 で打ち切り）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from .common import PROC, cfg

FAIL_STOPHIGH, FAIL_GAP, FAIL_HALT, FAIL_LOT = 0, 1, 2, 3


class Dense:
    def __init__(self):
        z = np.load(PROC / "dense.npz", allow_pickle=True)
        self.dates = pd.DatetimeIndex(z["dates"].astype("datetime64[ns]"))
        self.codes = z["codes"]
        self.O, self.C, self.O_u = z["O"], z["C"], z["O_u"]
        self.tradable, self.shopen, self.slopen = z["tradable"], z["shopen"], z["slopen"]
        self.div, self.adv, self.inuniv = z["div"], z["adv"], z["inuniv"]
        self.Cf = pd.DataFrame(self.C).ffill().to_numpy()
        T, S = self.C.shape
        fin = np.isfinite(self.C)
        self.last_valid = np.where(fin.any(0), T - 1 - np.argmax(fin[::-1], axis=0), -1).astype(np.int64)
        self.code_idx = {c: i for i, c in enumerate(self.codes)}
        self.date_idx = {d: i for i, d in enumerate(self.dates)}
        c = cfg()["trading"]["slippage"]
        tab = sorted(c, key=lambda r: -r["min_adv"])
        self.slip_thr = np.array([r["min_adv"] for r in tab], dtype=np.float64)
        self.slip_rate = np.array([r["rate"] for r in tab], dtype=np.float64)


@njit(cache=True)
def _slip(adv, thr, rate):
    for i in range(len(thr)):
        if adv >= thr[i]:
            return rate[i]
    return rate[len(rate) - 1]


@njit(cache=True)
def simulate(O, Cf, O_u, tradable, shopen, slopen, div, adv, last_valid, cand,
             sig_start, sig_end, end_day, N, K, gmax, fee, cost_mult, slip_thr, slip_rate,
             capital, part_cap, lot, b7_mode):
    """b7_mode=1: B7（構成銘柄の等ウェイト保有）。その日の候補全銘柄に等金額、参加率上限・単元・保有中スキップなし。"""
    T, S = O.shape
    J = cand.shape[1]
    maxpos = (sig_end - sig_start + 2) * K + 10
    # ポジション
    p_stock = np.full(maxpos, -1, np.int64)
    p_tr = np.zeros(maxpos, np.int64)
    p_sh = np.zeros(maxpos, np.float64)      # 調整後株数
    p_cost = np.zeros(maxpos, np.float64)
    p_div = np.zeros(maxpos, np.float64)
    p_buy = np.zeros(maxpos, np.int64)
    p_selld = np.zeros(maxpos, np.int64)
    p_exit = np.full(maxpos, -1, np.int64)
    p_proc = np.zeros(maxpos, np.float64)
    held = np.zeros(S, np.int64)
    npos = 0
    tr_cash = np.full(N, capital / N)
    nav = np.full(T, np.nan)
    inv = np.full(T, np.nan)
    fails = np.zeros(4, np.int64)
    nobuy_days = 0
    sig_days = 0
    lo = 0   # これより前のポジションはすべて決済済み（走査範囲の短縮）
    for d in range(sig_start + 1, end_day + 1):
        while lo < npos and p_exit[lo] >= 0:
            lo += 1
        # 1) 配当: 権利落ち日 d に、d-1 の終値時点で保有していたポジション
        for k in range(lo, npos):
            if p_exit[k] < 0 and p_buy[k] < d:
                dv = div[d, p_stock[k]]
                if dv > 0:
                    p_div[k] += p_sh[k] * dv
        # 2) 売り（寄り付き）
        for k in range(lo, npos):
            if p_exit[k] >= 0 or p_selld[k] > d:
                continue
            s = p_stock[k]
            if d > last_valid[s]:
                # データ終了（上場廃止等）: 最終取引日の終値で売却
                px = Cf[last_valid[s], s]
                sl = _slip(adv[last_valid[s], s], slip_thr, slip_rate) * cost_mult
                p_proc[k] = p_sh[k] * px * (1 - sl - fee * cost_mult)
            elif tradable[d, s] and (not slopen[d, s]) and np.isfinite(O[d, s]):
                sl = _slip(adv[d - 1, s], slip_thr, slip_rate) * cost_mult
                p_proc[k] = p_sh[k] * O[d, s] * (1 - sl - fee * cost_mult)
            else:
                continue  # 翌営業日に持ち越し
            p_exit[k] = d
            tr_cash[p_tr[k]] += p_proc[k] + p_div[k]
            held[s] -= 1
        # 3) 買い（t = d-1 のシグナル）
        t = d - 1
        if t <= sig_end:
            sig_days += 1
            j = (d - sig_start - 1) % N
            alloc = tr_cash[j] / K
            if b7_mode == 1:
                nc = 0
                for c in range(J):
                    if cand[t, c] >= 0:
                        nc += 1
                alloc = tr_cash[j] / max(nc, 1)
            slots = 0
            bought = 0
            for c in range(J):
                if slots >= K:
                    break
                s = cand[t, c]
                if s < 0:
                    break
                if held[s] > 0 and b7_mode == 0:
                    continue  # 保有中の銘柄は飛ばして次の順位へ（枠は消費しない）
                slots += 1
                if (not tradable[d, s]) or (not np.isfinite(O[d, s])):
                    fails[FAIL_HALT] += 1
                    continue
                if shopen[d, s]:
                    fails[FAIL_STOPHIGH] += 1
                    continue
                if gmax >= 0 and O[d, s] > Cf[t, s] * (1 + gmax):
                    fails[FAIL_GAP] += 1
                    continue
                sl = _slip(adv[t, s], slip_thr, slip_rate) * cost_mult
                pu = O_u[d, s] * (1 + sl)
                if b7_mode == 1:
                    lots = alloc / (pu * lot * (1 + fee * cost_mult))   # 端株を許容
                else:
                    amt = min(alloc, part_cap * adv[t, s])
                    lots = np.floor(amt / (pu * lot * (1 + fee * cost_mult)))
                if lots < 1 and b7_mode == 0:
                    fails[FAIL_LOT] += 1
                    continue
                cost = lots * lot * pu * (1 + fee * cost_mult)
                tr_cash[j] -= cost
                p_stock[npos] = s
                p_tr[npos] = j
                p_sh[npos] = lots * lot * (O_u[d, s] / O[d, s])  # 実株数 → 調整後株数
                p_cost[npos] = cost
                p_buy[npos] = d
                p_selld[npos] = d + N
                held[s] += 1
                npos += 1
                bought += 1
            if bought == 0:
                nobuy_days += 1
        # 4) 終値で評価
        mv = 0.0
        for k in range(lo, npos):
            if p_exit[k] < 0:
                s = p_stock[k]
                px = Cf[min(d, last_valid[s]), s]
                mv += p_sh[k] * px + p_div[k]
        cash = 0.0
        for j2 in range(N):
            cash += tr_cash[j2]
        nav[d] = cash + mv
        inv[d] = mv / (cash + mv) if cash + mv > 0 else 0.0
    # 期末で未決済のポジションは終値評価のまま（取引としては未確定）
    return (nav, inv, p_stock[:npos], p_buy[:npos], p_exit[:npos], p_cost[:npos],
            p_proc[:npos] + p_div[:npos], fails, nobuy_days, sig_days)


@njit(cache=True)
def random_cands(inuniv, sig_start, sig_end, J, seed):
    np.random.seed(seed)
    T, S = inuniv.shape
    out = np.full((T, J), -1, np.int64)
    buf = np.empty(S, np.int64)
    for t in range(sig_start, sig_end + 1):
        n = 0
        for s in range(S):
            if inuniv[t, s]:
                buf[n] = s
                n += 1
        m = min(J, n)
        for i in range(m):
            r = i + np.random.randint(n - i)
            tmp = buf[i]
            buf[i] = buf[r]
            buf[r] = tmp
            out[t, i] = buf[i]
    return out


def run(D: Dense, cand, sig_start, sig_end, end_day, N, K, gmax=None, fee=None, cost_mult=1.0, b7=False, capital=None):
    c = cfg()["trading"]
    capital = c["initial_capital"] if capital is None else capital
    fee = c["fee_oneway"] if fee is None else fee
    res = simulate(D.O, D.Cf, D.O_u, D.tradable, D.shopen, D.slopen, D.div, D.adv, D.last_valid, cand,
                   int(sig_start), int(sig_end), int(end_day), int(N), int(K), -1.0 if gmax is None else float(gmax),
                   float(fee), float(cost_mult), D.slip_thr, D.slip_rate, float(capital),
                   float(c["participation_cap"]), float(c["lot"]), 1 if b7 else 0)
    nav, inv, st, bd, ed, cost, proc, fails, nobuy, sigd = res
    sl = slice(sig_start + 1, end_day + 1)
    navs = pd.Series(nav[sl], index=D.dates[sl])
    done = ed >= 0
    trades = pd.DataFrame({"stock": st[done], "buy_d": bd[done], "exit_d": ed[done], "cost": cost[done], "proceeds": proc[done]})
    trades["ret"] = trades["proceeds"] / trades["cost"] - 1
    trades["pnl"] = trades["proceeds"] - trades["cost"]
    return {"nav": navs, "invested": pd.Series(inv[sl], index=D.dates[sl]), "trades": trades,
            "fails": {"stop_high": int(fails[0]), "gap": int(fails[1]), "halt": int(fails[2]), "lot": int(fails[3])},
            "nobuy_days": int(nobuy), "signal_days": int(sigd)}

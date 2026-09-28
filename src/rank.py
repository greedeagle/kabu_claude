"""Step 13: 最終学習と翌営業日の購入候補ランキング（第14章の形式）。

ホールドアウトに合格した仕様のみ、変更せずに最新日までの全データで再学習する。
合格していない場合は「購入候補なし（検証不合格）」と、参考として WF 検証で最良だった候補のスコア上位3銘柄を出力する。
"""
from __future__ import annotations

import datetime as dt

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

from .common import OUT, PROC, RAW, cfg, load_json, save_json, slippage_rate
from .cv import calendar
from .grid import PRED_RULE, Ctx
from .models import Data, fit_model
from .overfit import parse_id, run_spec, score_with

FEAT_LABEL = {
    "r_rs60": "60日RS", "r_rs20": "20日RS", "r_vr5": "5日出来高比", "r_vr20": "20日出来高比", "r_rsi14": "RSI14",
    "r_ret1": "1日リターン", "r_ret3": "3日リターン", "r_ret5": "5日リターン", "r_ret20": "20日リターン",
}


def next_business_day(d: pd.Timestamp) -> pd.Timestamp:
    jp = holidays.Japan(years=[d.year, d.year + 1])
    x = d + pd.Timedelta(days=1)
    while x.weekday() >= 5 or x.date() in jp or (x.month == 12 and x.day == 31) or (x.month == 1 and x.day in (1, 2, 3)):
        x += pd.Timedelta(days=1)
    return x


def add_bdays(d: pd.Timestamp, n: int) -> pd.Timestamp:
    for _ in range(n):
        d = next_business_day(d)
    return d


def band_stats(fam, N, spec):
    """WF検証で同じスコア帯（同日クロスセクションの上位 θ）に入った銘柄の、コスト控除後の実現リターン。"""
    c = cfg()
    sc = pd.read_parquet(PROC / "scores" / f"{fam}_N{N}.parquet")
    lab = pd.read_parquet(PROC / "labels.parquet", columns=["Date", "Code", "ADV20", f"L1_{N}"])
    s = sc.merge(lab, on=["Date", "Code"])
    s["pct"] = s.groupby("Date")["score"].rank(ascending=False, pct=True)
    rt = 2 * (c["trading"]["fee_oneway"] + slippage_rate(s["ADV20"].to_numpy(), c["trading"]["slippage"]))
    s["net"] = s[f"L1_{N}"] - rt
    b = s[(s["pct"] <= spec["theta"]) & s["net"].notna()]
    return {"mean_net": float(b["net"].mean()), "up_rate": float((b["net"] > 0).mean()), "n": int(len(b))}


def calibration_ok(fam, N):
    """分類モデル: WF検証で予測確率の十分位ごとの実現頻度との差が最大5ポイント以内なら確認済みとする。"""
    if fam not in ("M3", "M4c"):
        return False, None
    sc = pd.read_parquet(PROC / "scores" / f"{fam}_N{N}.parquet")
    lab = pd.read_parquet(PROC / "labels.parquet", columns=["Date", "Code", f"L4_{N}"])
    s = sc.merge(lab, on=["Date", "Code"]).dropna(subset=["pred", f"L4_{N}"])
    s["bin"] = pd.qcut(s["pred"], 10, duplicates="drop")
    t = s.groupby("bin", observed=True).agg(p=("pred", "mean"), y=(f"L4_{N}", "mean"))
    dev = float((t["p"] - t["y"]).abs().max())
    return dev <= 0.05, t.reset_index(drop=True).to_dict(orient="records")


def contributions(fam, mdl, row: pd.DataFrame, D: Data, adopted):
    """上位3要因。ルール・スコア: 各項目（向きを揃えた順位 − 0.5）、木: SHAP 値。Step 5 で採用（符号安定）した特徴量のみ。"""
    feats = D.fcols
    if fam in ("M4r", "M4c", "M5"):
        sh = mdl.predict(D.Xm(row), num_iteration=mdl.best_iteration, pred_contrib=True)[0][:-1]
        contrib = pd.Series(sh, index=feats + D.mcols)[feats]
    elif fam == "M7":
        contrib = pd.Series(0.0, index=feats)
    else:
        sign = adopted["sign"] if fam != "M1" else mdl["sign"]
        use = feats if fam != "M1" else mdl["features"]
        contrib = pd.Series({f: ((row[f].iloc[0] if sign.get(f, 1) > 0 else 1 - row[f].iloc[0]) - 0.5) for f in use})
    top = contrib.sort_values(ascending=False).head(3)
    out = []
    for f, v in top.items():
        pct = row[f].iloc[0]
        out.append(f"{FEAT_LABEL.get(f, f[2:])} が上位{(1 - pct) * 100:.0f}%（寄与 {v:+.3f}）")
    return out


def main():
    c = cfg()
    sp = load_json(OUT / "splits.json")
    cal = calendar()
    t = cal[-1]
    t1 = next_business_day(t)
    ho = load_json(OUT / "step12_holdout.json") if (OUT / "step12_holdout.json").exists() else {"evaluated": False}
    passed = bool(ho.get("evaluated") and ho.get("pass"))
    header = {"signal_date_t": t.date().isoformat(), "buy_date_t1": t1.date().isoformat(),
              "holdout_result": ("合格" if passed else ("不合格" if ho.get("evaluated") else "未評価（" + str(ho.get("reason")) + "）"))}
    ctx = Ctx(holdout=True)
    ti = ctx.D.date_idx[t]
    header["regime"] = {"idx_above_sma200": bool(ctx.above[ti]), "idx_vol20_below_p80": bool(ctx.vol_ok[ti]) if np.isfinite(ctx.vol_ok[ti]) else None}

    if passed:
        spec = ho["spec"]
        tid = ho["id"]
    else:
        # 参考表示用: 過学習チェック対象のうち WF Sharpe 最良の候補、無ければ WF グリッドの最良
        ov = load_json(OUT / "step11_overfit.json") if (OUT / "step11_overfit.json").exists() else {}
        grid = pd.read_csv(OUT / "step9_grid.csv")
        tid = (ov.get("checked") or [{}])[0].get("id") or grid.sort_values("sharpe", ascending=False)["id"].iloc[0]
        spec = parse_id(tid)
    fam, N = spec["family"], spec["N"]
    header["model_id"] = tid
    header["threshold"] = {"theta_top_pct": spec["theta"], "gmax": spec["gmax"], "regime_filter": spec["regime"],
                           "pred_condition": PRED_RULE.get(fam)}
    if fam == "M6":
        members = load_json(PROC / "scores" / f"M6_N{N}_meta.json")[0]["members"]
    D = Data(N, include_holdout=True)
    te = cal[-1 - N - 1]
    fams = members if fam == "M6" else [fam]
    today = D.df[D.df["Date"] == t].reset_index(drop=True)
    scores, mdls = [], {}
    for f in fams:
        mdl = fit_model(f, D.train_until(te), D)
        mdls[f] = mdl
        s, p = score_with(f, mdl, today, D)
        scores.append((f, s, p))
    if fam == "M6":
        score = np.mean([pd.Series(s).rank(pct=True).to_numpy() for _, s, _ in scores], axis=0)
        pred = np.full(len(today), np.nan)
    else:
        _, score, pred = scores[0]
    sc = pd.DataFrame({"Date": t, "Code": today["Code"], "score": score, "pred": pred})
    sc["pct"] = sc["score"].rank(ascending=False, pct=True)
    cand = ctx.build_cand(sc, theta=spec["theta"], regime=spec["regime"], pred_rule=PRED_RULE.get(fam))
    chosen = [ctx.D.codes[s] for s in cand[ti] if s >= 0][: spec["K"]]

    jl = pd.read_parquet(RAW / "jpx_list.parquet")
    name = dict(zip(jl["コード"].astype(str), jl["銘柄名"]))
    sector = dict(zip(jl["コード"].astype(str), jl["33業種区分"]))
    feat_t = pd.read_parquet(PROC / "features.parquet", filters=[("Date", "==", t)])
    atr_q90 = feat_t["atr14p"].quantile(0.9)
    bs = band_stats(fam if fam != "M6" else fams[0], N, spec) if fam != "M6" else band_stats("M6", N, spec)
    cal_ok, _ = calibration_ok(fam, N)
    sell_date = add_bdays(t1, N)
    alloc = c["trading"]["initial_capital"] / N / spec["K"]
    ov = load_json(OUT / "step11_overfit.json") if (OUT / "step11_overfit.json").exists() else {}
    rows = []
    reason = None
    if not passed:
        reason = "検証不合格" if ho.get("evaluated") else f"検証不合格（{ho.get('reason')}）"
    elif not chosen:
        reason = "レジームフィルタ" if not ctx.regime_ok(spec["regime"])[ti] else "閾値未達"
    if passed and chosen:
        for rank_i, code in enumerate(chosen, 1):
            s = ctx.D.code_idx[code]
            r = today[today["Code"] == code]
            ft = feat_t[feat_t["Code"] == code].iloc[0]
            C_u = ctx.D.C_u[ti, s]
            lim = C_u * (1 + spec["gmax"]) if spec["gmax"] is not None else None
            px = lim if lim else C_u
            cap = min(alloc, c["trading"]["participation_cap"] * ctx.D.adv[ti, s])
            shares = int(np.floor(cap / (px * 100)) * 100)
            risks = []
            if ft["atr14p"] >= atr_q90:
                risks.append("高ボラティリティ（ATR%がユニバース上位10%）")
            if spec["gmax"] is not None:
                risks.append(f"寄り付きが指値 {lim:,.1f} 円を上回ると約定しない")
            if ctx.D.adv[ti, s] < 3e8:
                risks.append(f"流動性（ADV20 {ctx.D.adv[ti, s] / 1e8:.1f}億円）")
            risks.append("決算予定: データなし（無料データ源のため未確認）")
            same = sum(sector.get(x) == sector.get(code) for x in chosen)
            if same >= 2:
                risks.append(f"セクター集中（{sector.get(code)} が候補内に{same}銘柄）")
            chk = next((x for x in ov.get("checked", []) if x["id"] == tid), {})
            rows.append({
                "Rank": rank_i, "銘柄コード": code, "銘柄名": name.get(code, ""),
                "Score": f"{float(sc.loc[sc['Code'] == code, 'score'].iloc[0]):.4f}（上位{float(sc.loc[sc['Code'] == code, 'pct'].iloc[0]) * 100:.1f}%）",
                "予測リターン": f"{bs['mean_net'] * 100:+.2f}%（WF検証の同スコア帯の平均実現値, n={bs['n']}）",
                "予測上昇確率": (f"{float(r['pred'].iloc[0]) if fam in ('M3', 'M4c') else bs['up_rate']:.2f}" if cal_ok else "—"),
                "推奨保有期間": f"{N}営業日（売却予定日 {sell_date.date()} の寄り付き）",
                "購入判断": "買い",
                "発注方法": f"寄り付き指値 上限 {lim:,.1f} 円（= 終値 {C_u:,.1f} × {1 + spec['gmax']:.2f}）" if lim else "寄り付き成行（指値なし）",
                "数量上限": f"{shares:,} 株（約 {shares * px / 1e4:,.1f} 万円、配分額と ADV20 の1%の小さい方）",
                "売却条件": f"{sell_date.date()} の寄り付きで売却（損切り・利確ルールなし）",
                "根拠": contributions(fams[0], mdls[fams[0]], today[today["Code"] == code], D, D.adopt),
                "リスク要因": risks,
            })
    ref = sc.sort_values("score", ascending=False).head(3)
    out = {"header": header, "rows": rows, "no_candidate_reason": reason if not rows else None,
           "reference_top3": [{"code": r.Code, "name": name.get(r.Code, ""), "score": float(r.score), "pct": float(r.pct)} for r in ref.itertuples()],
           "band_stats": bs, "calibration_confirmed": cal_ok}
    save_json(out, OUT / "step13_ranking.json")
    print(out["header"], "rows", len(rows), "reason", out["no_candidate_reason"])


if __name__ == "__main__":
    main()

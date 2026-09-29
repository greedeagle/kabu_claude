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
from .models import KIND, Data, fit_model, score_with
from .overfit import parse_id

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


def contributions(fam, mdl, row: pd.DataFrame, D: Data):
    """上位3要因。LightGBM: SHAP 値（シード・アンサンブルの平均）、順位平均: 向きを揃えた順位 − 0.5。"""
    if fam in KIND:
        sh = np.mean([m.predict(D.Xm(row), num_iteration=m.best_iteration, pred_contrib=True)[0][:-1] for m in mdl], axis=0)
        contrib = pd.Series(sh, index=D.fcols + D.mcols)[D.fcols]
    else:
        sign = D.m2r if fam == "M2r" else mdl
        contrib = pd.Series({f: ((row[f].iloc[0] if sg > 0 else 1 - row[f].iloc[0]) - 0.5) for f, sg in sign.items()})
    out = []
    for f, v in contrib.sort_values(ascending=False).head(3).items():
        out.append(f"{FEAT_LABEL.get(f, f[2:])} が上位{(1 - row[f].iloc[0]) * 100:.0f}%（寄与 {v:+.3f}）")
    return out


def splits_between(code: str, a: pd.Timestamp, b: pd.Timestamp) -> float:
    """a < 日付 <= b に権利落ちする分割比率の積（生データ。data_end より後の情報だが、公表済みの分割）。"""
    f = RAW / "prices" / f"{code}.parquet"
    if not f.exists():
        return 1.0
    d = pd.read_parquet(f, columns=["Date", "Splits"])
    r = d.loc[(d["Date"] > a) & (d["Date"] <= b) & (d["Splits"] > 0), "Splits"]
    return float(r.prod()) if len(r) else 1.0


def main():
    c = cfg()
    cal = calendar()
    t = cal[-1]
    # 購入日: シグナル日の翌営業日。実行日がそれより後なら、実行日の翌営業日（間に空く営業日を注記）
    today = pd.Timestamp(dt.date.today())
    t1 = max(next_business_day(t), next_business_day(today))
    gap_days = [d for d in pd.bdate_range(t + pd.Timedelta(days=1), t1 - pd.Timedelta(days=1)) if next_business_day(d - pd.Timedelta(days=1)) == d]
    ho = load_json(OUT / "step12_holdout.json") if (OUT / "step12_holdout.json").exists() else {"evaluated": False}
    passed = bool(ho.get("evaluated") and ho.get("pass"))
    header = {"signal_date_t": t.date().isoformat(), "buy_date": t1.date().isoformat(),
              "test_result": ("合格" if passed else ("不合格" if ho.get("evaluated") else "未評価（" + str(ho.get("reason")) + "）"))}
    if gap_days:
        header["note_gap"] = (f"シグナル日と購入日の間に {', '.join(d.date().isoformat() for d in gap_days)} が挟まる。"
                              "モデルは t+1 の寄り付きで買う前提で検証しており、この間の値動きは考慮していない")
    ctx = Ctx(holdout=True)
    ti = ctx.D.date_idx[t]
    header["n_universe_Ut"] = int(ctx.D.inuniv[ti].sum())
    header["regime"] = {"idx_above_sma200": bool(ctx.above[ti]), "idx_vol20_below_p80": bool(ctx.vol_ok[ti])}
    if passed:
        spec, tid = ho["spec"], ho["id"]
    else:
        ov = load_json(OUT / "step11_overfit.json") if (OUT / "step11_overfit.json").exists() else {}
        grid = pd.read_csv(OUT / "step9_grid.csv")
        tid = (ov.get("checked") or [{}])[0].get("id") or grid.sort_values("plateau", ascending=False)["id"].iloc[0]
        spec = parse_id(tid)
    fam, N = spec["family"], spec["N"]
    header["model_id"] = tid
    header["threshold"] = {"theta_top_pct": spec["theta"], "gmax": spec["gmax"], "regime_filter": spec["regime"]}
    D = Data(include_holdout=True)
    te = cal[-1 - D.H - 1]
    today_rows = D.df[D.df["Date"] == t].reset_index(drop=True)
    mdl = fit_model(fam, D.train_until(te), D)
    score = score_with(fam, mdl, today_rows, D)
    sc = pd.DataFrame({"Date": t, "Code": today_rows["Code"], "score": score, "pred": np.nan})
    sc["pct"] = sc["score"].rank(ascending=False, pct=True)
    cand = ctx.build_cand(sc, theta=spec["theta"], regime=spec["regime"])
    chosen = [ctx.D.codes[s] for s in cand[ti] if s >= 0][: spec["K"]]

    jl = pd.read_parquet(RAW / "jpx_list.parquet")
    name = dict(zip(jl["コード"].astype(str), jl["銘柄名"]))
    sector = dict(zip(jl["コード"].astype(str), jl["33業種区分"]))
    # t 日までに公表済みの日経の発表（2026-09-04 / 09-08 のインデックス・ニュース。n225/REPORT.md から転記）
    announced = {"4902": "日経225から除外（2026-09-04発表、10/1実施）", "543A": "日経225から除外（2026-09-04発表、10/1実施）",
                 "7004": "日経225から除外（2026-09-04発表、10/1実施）", "4004": "9/29 にスピンオフ（クラサスケミカル）で権利落ち"}
    feat_t = pd.read_parquet(PROC / "features.parquet", filters=[("Date", "==", t)])
    atr_q90 = feat_t["atr14p"].quantile(0.9)
    bs = band_stats(fam, N, spec)
    sell_date = add_bdays(t1, N)
    alloc = c["trading"]["initial_capital"] / N / spec["K"]
    rows = []
    reason = None
    if not passed:
        reason = "検証不合格" if ho.get("evaluated") else f"検証不合格（{ho.get('reason')}）"
    elif not chosen:
        reason = "レジームフィルタ" if not ctx.regime_ok(spec["regime"])[ti] else "閾値未達"
    if passed and chosen:
        for rank_i, code in enumerate(chosen, 1):
            s = ctx.D.code_idx[code]
            r = today_rows[today_rows["Code"] == code]
            ft = feat_t[feat_t["Code"] == code].iloc[0]
            split = splits_between(code, t, t1)     # t の翌日から購入日までの分割（1株あたり価格を割り戻す）
            C_u = ctx.D.C_u[ti, s] / split
            lim = C_u * (1 + spec["gmax"])
            cap = min(alloc, c["trading"]["participation_cap"] * ctx.D.adv[ti, s])
            shares = int(np.floor(cap / (lim * 100)) * 100)
            risks = []
            if split != 1.0:
                risks.append(f"{t.date()} の後に株式分割（1:{split:g}）の権利落ち。終値・指値は分割後の株数ベースに換算済み")
            if ft["atr14p"] >= atr_q90:
                risks.append("高ボラティリティ（ATR%がユニバース上位10%）")
            risks.append(f"寄り付きが指値 {lim:,.1f} 円を上回ると約定しない")
            if gap_days:
                risks.append(f"{gap_days[0].date()} の値動きを見ていない（シグナルは {t.date()} の終値時点）")
            risks.append("決算予定: データなし（無料データ源のため未確認）")
            if code in announced:
                risks.append(announced[code])
            same = sum(sector.get(x) == sector.get(code) for x in chosen)
            if same >= 2:
                risks.append(f"セクター集中（{sector.get(code)} が候補内に{same}銘柄）")
            rows.append({
                "Rank": rank_i, "銘柄コード": code, "銘柄名": name.get(code, ""),
                "Score": f"{float(sc.loc[sc['Code'] == code, 'score'].iloc[0]):.4f}（上位{float(sc.loc[sc['Code'] == code, 'pct'].iloc[0]) * 100:.1f}%）",
                "予測リターン": f"{bs['mean_net'] * 100:+.2f}%（WF検証の同スコア帯の平均実現値, n={bs['n']}）",
                "上昇確率（参考）": f"{bs['up_rate']:.2f}（同上）",
                "保有期間": f"{N}営業日（売却予定日 {sell_date.date()} の寄り付き）",
                "発注方法": f"寄り付き指値 上限 {lim:,.1f} 円（= 基準終値 {C_u:,.1f} × {1 + spec['gmax']:.2f}）",
                "数量上限": f"{shares:,} 株（約 {shares * lim / 1e4:,.1f} 万円、配分額と ADV20 の1%の小さい方）",
                "売却条件": f"{sell_date.date()} の寄り付きで売却（損切り・利確ルールなし）",
                "根拠": contributions(fam, mdl, r, D),
                "リスク要因": risks,
            })
    ref = sc.sort_values("score", ascending=False).head(5)
    header["announced_events_used"] = announced
    out = {"header": header, "rows": rows, "no_candidate_reason": reason if not rows else None,
           "reference_top5": [{"code": r.Code, "name": name.get(r.Code, ""), "score": float(r.score), "pct": float(r.pct)} for r in ref.itertuples()],
           "band_stats": bs}
    save_json(out, OUT / "step13_ranking.json")
    print(out["header"], "rows", len(rows), "reason", out["no_candidate_reason"])
    for r in rows:
        print(r["Rank"], r["銘柄コード"], r["銘柄名"], r["Score"], r["発注方法"], r["数量上限"])


if __name__ == "__main__":
    main()

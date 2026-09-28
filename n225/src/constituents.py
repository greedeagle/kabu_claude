"""日経225構成銘柄の履歴（ポイントインタイム、実施日基準）。

入力: 日経公式『日経平均株価銘柄変更履歴』PDF と現在の構成銘柄一覧ページ（data/raw/n225/）。
方法: 現在の構成から入替イベントを新しい順に逆算して、各銘柄の在籍区間 [in_date, out_date) を作る。
      t 日の構成 = in_date <= t < out_date（採用の実施日から対象、除外の実施日以降は対象外）。
発表日: PDF に掲載がないため欠損（入替関連の特徴量は作らない）。
出力: out/constituents.csv, out/step1_constituents_check.json
環境変数 KABU_UNIV=current のときは「現在の構成銘柄で全期間を遡る」版（構成銘柄リークテスト用）を返す。
"""
from __future__ import annotations

import os
import re

import pandas as pd
import pdfplumber

from .common import OUT, RAW, cfg, save_json

N225 = RAW / "n225"
PDF = N225 / "history_of_nikkei_stock_average_component_changes_jp.pdf"
COMPONENT_HTML = N225 / "component.html"


def parse_changes() -> pd.DataFrame:
    rows = []
    with pdfplumber.open(PDF) as p:
        for pg in p.pages:
            for t in pg.extract_tables():
                for r in t:
                    if r[0] == "年月日" or r[1] == "銘柄名":
                        continue
                    rows.append([(x or "").replace("\n", "") if x is not None else None for x in r])
    df = pd.DataFrame(rows, columns=["date", "out_name", "out_code", "in_name", "in_code"])
    df["date"] = df["date"].replace("", None).ffill()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def current_members() -> pd.DataFrame:
    t = COMPONENT_HTML.read_text(encoding="utf-8")
    # 表の行: <td>コード</td><td>銘柄名</td>...
    rows = re.findall(r"<td[^>]*>\s*([0-9][0-9A-Z]{3})\s*</td>\s*<td[^>]*>(.*?)</td>", t, flags=re.S)
    df = pd.DataFrame(rows, columns=["code", "name"]).drop_duplicates("code")
    df["name"] = df["name"].str.replace(r"<[^>]+>", "", regex=True).str.strip()
    return df


def build_intervals() -> tuple[pd.DataFrame, dict]:
    c = cfg()
    since = pd.Timestamp(c["data"]["constituents_from"])
    ch = parse_changes()
    # 対象期間のイベントに日付欠損がないこと（PDF 解析の検証）
    idx_since = ch.index[ch["date"] >= since]
    assert ch.loc[: idx_since.max(), "date"].notna().all(), "date parse failure in analysis period"
    ev = ch[ch["date"] >= since].copy()
    cur = current_members()
    assert len(cur) == 225, f"current members {len(cur)}"
    names = {}
    for r in ev.itertuples():
        if r.out_code not in ("－", None, ""):
            names.setdefault(r.out_code, r.out_name)
        if r.in_code not in ("－", None, ""):
            names.setdefault(r.in_code, r.in_name)
    for r in cur.itertuples():
        names[r.code] = r.name
    members = set(cur["code"])
    open_end = {code: pd.NaT for code in members}   # 在籍中（除外日なし）
    iv, checks = [], []
    for d, g in ev.sort_values("date", ascending=False).groupby("date", sort=False):
        ins = {x for x in g["in_code"] if x not in ("－", None, "")}
        outs = {x for x in g["out_code"] if x not in ("－", None, "")}
        miss = sorted(ins - members)
        clash = sorted((outs & members) - ins)
        assert not miss and not clash, (d, miss, clash)
        for code in ins:
            iv.append({"code": code, "in_date": d, "out_date": open_end.pop(code)})
        members = (members - ins) | outs
        for code in outs:
            open_end[code] = d
        checks.append({"date": d.date().isoformat(), "in": sorted(ins), "out": sorted(outs), "n_before": len(members)})
    for code, e in open_end.items():
        iv.append({"code": code, "in_date": pd.NaT, "out_date": e})   # since より前から在籍
    df = pd.DataFrame(iv)
    df["name"] = df["code"].map(names)
    df["announce_date"] = pd.NaT   # 公式履歴に掲載なし
    df = df.sort_values(["in_date", "code"], na_position="first").reset_index(drop=True)
    return df, {"n_events_since": int(ev["date"].nunique()), "events": checks}


def membership(dates: pd.DatetimeIndex, codes: list[str]) -> pd.DataFrame:
    """日付×銘柄の bool 行列（t 日終了時点の構成）。"""
    if os.environ.get("KABU_UNIV") == "current":
        cur = set(current_members()["code"])
        return pd.DataFrame({c: c in cur for c in codes}, index=dates)
    iv = pd.read_csv(OUT / "constituents.csv", dtype={"code": str}, parse_dates=["in_date", "out_date"])
    m = pd.DataFrame(False, index=dates, columns=codes)
    for r in iv.itertuples():
        if r.code not in m.columns:
            continue
        lo = r.in_date if pd.notna(r.in_date) else dates[0]
        hi = r.out_date if pd.notna(r.out_date) else dates[-1] + pd.Timedelta(days=1)
        m.loc[(dates >= lo) & (dates < hi), r.code] = True
    return m


def main():
    df, info = build_intervals()
    df.to_csv(OUT / "constituents.csv", index=False)
    save_json(info, OUT / "step1_constituents_check.json")
    print("intervals", len(df), "codes", df["code"].nunique(), "events", info["n_events_since"])


if __name__ == "__main__":
    main()

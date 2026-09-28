from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
# デバッグ用: KABU_WORK で中間生成物・出力先を切り替え、KABU_MAX_DATE で生データを切り詰め、KABU_CONFIG で設定を差し替える
WORK = Path(os.environ.get("KABU_WORK", ROOT))
PROC = WORK / "data" / "proc"
OUT = WORK / "out"
MAX_DATE = os.environ.get("KABU_MAX_DATE")
for _p in (PROC, OUT):
    _p.mkdir(parents=True, exist_ok=True)


def cfg() -> dict:
    return yaml.safe_load(Path(os.environ.get("KABU_CONFIG", ROOT / "config.yaml")).read_text())


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def load_json(path: Path):
    return json.loads(path.read_text())


# 東証の値幅制限（基準値 < 上限 → 制限値幅）
_LIMIT_TABLE = [
    (100, 30), (200, 50), (500, 80), (700, 100), (1000, 150), (1500, 300), (2000, 400),
    (3000, 500), (5000, 700), (7000, 1000), (10000, 1500), (15000, 3000), (20000, 4000),
    (30000, 5000), (50000, 7000), (70000, 10000), (100000, 15000), (150000, 30000),
    (200000, 40000), (300000, 50000), (500000, 70000), (700000, 100000), (1000000, 150000),
    (1500000, 300000), (2000000, 400000), (3000000, 500000), (5000000, 700000),
    (7000000, 1000000), (10000000, 1500000), (15000000, 3000000), (20000000, 4000000),
    (30000000, 5000000), (50000000, 7000000), (np.inf, 10000000),
]
_BOUNDS = np.array([b for b, _ in _LIMIT_TABLE], dtype=float)
_WIDTHS = np.array([w for _, w in _LIMIT_TABLE], dtype=float)


def price_limit_width(base: np.ndarray) -> np.ndarray:
    base = np.asarray(base, dtype=float)
    idx = np.searchsorted(_BOUNDS, base, side="right")
    idx = np.clip(idx, 0, len(_WIDTHS) - 1)
    out = _WIDTHS[idx]
    return np.where(np.isfinite(base), out, np.nan)


def slippage_rate(adv: np.ndarray, table: list[dict]) -> np.ndarray:
    adv = np.asarray(adv, dtype=float)
    out = np.full(adv.shape, table[-1]["rate"], dtype=float)
    for row in sorted(table, key=lambda r: r["min_adv"]):
        out = np.where(adv >= row["min_adv"], row["rate"], out)
    return out

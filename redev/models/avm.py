"""avm.py — 심장2 Stage3 입지가치 (대지지분 평당단가 AVM). 설계: docs/design/avm.md.

★R6: 개별 호수 매핑 금지 → 필지 반경 집계 대지지분 평당단가가 타깃. 계층 폴백(50m→100m→동)
으로 거짓0(결측을 0으로)도 결측천지도 둘 다 피하고, 어느 단계서 채웠는지 agg_level로 기록.
★R14·R17 상승여력은 범위·시나리오로만(단정 금지). ★R16 거래편향·R15 규제역설 출력 명시.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from redev.config import load_avm_config

_PYUNG_M2 = 3.305785          # 1평(㎡). 대지지분 평당단가 = 거래금액 / (대지권면적/이 값)


def _villa_pyung(trades: pd.DataFrame) -> pd.DataFrame:
    """연립다세대 거래 → 대지지분 평당단가(만원/평). 대지권 유효한 것만(R6)."""
    v = trades[(trades["trade_type"] == "villa")
               & trades["land_share_m2"].notna() & (trades["land_share_m2"] > 0)].copy()
    v["pyung"] = v["deal_amount"] / (v["land_share_m2"] / _PYUNG_M2)
    return v


def _window_filter(trades: pd.DataFrame, current_ym: str, years: int) -> pd.DataFrame:
    """최근 N년 거래만(시세 최신성). current_ym='YYYYMM'."""
    cy, cm = int(current_ym[:4]), int(current_ym[4:6])
    cutoff = (cy - years) * 100 + cm
    ym = pd.to_numeric(trades["deal_ym"], errors="coerce")
    return trades[ym >= cutoff]


def build_target(parcels, trades, *, current_ym: str, cfg=None) -> pd.DataFrame:
    """★R6 타깃: 필지별 반경 집계 대지지분 평당단가(median) + agg_level. 계층 폴백.

    입력: parcels(geom — 거래·타깃 필지 좌표), trades(transactions 정제), current_ym.
    출력: [pnu, target_pyung, agg_level('r50'/'r100'/'dong'/'missing'), n_trades].
    """
    cfg = (cfg or load_avm_config())["target"]
    v = _window_filter(_villa_pyung(trades), current_ym, cfg["window_years"])

    # 거래 좌표 = 거래 필지 centroid (집합건물·호수 무관, 땅 위치 — R6)
    cent = parcels.set_index("pnu").geometry.centroid
    v = v.assign(x=v["pnu"].map(cent.x), y=v["pnu"].map(cent.y)).dropna(subset=["x", "y"])
    v["dong"] = v["pnu"].str[:10]                       # 법정동코드 10자리(폴백 단위)
    dong_med = v.groupby("dong")["pyung"].median()

    # 타깃 필지 centroid
    pc = parcels[["pnu"]].copy()
    c = parcels.geometry.centroid
    pc["x"], pc["y"], pc["dong"] = c.x.values, c.y.values, parcels["pnu"].str[:10].values

    n = len(pc)
    target = np.full(n, np.nan)
    level = np.array(["missing"] * n, dtype=object)
    ntr = np.zeros(n, dtype=int)
    tree = cKDTree(v[["x", "y"]].to_numpy())
    prices = v["pyung"].to_numpy()
    min_tr = cfg["min_trades"]

    for radius in cfg["radius_levels_m"]:                # 50 → 100 (가까운 단계 우선)
        todo = np.where(level == "missing")[0]
        if todo.size == 0:
            break
        nbrs = tree.query_ball_point(pc.iloc[todo][["x", "y"]].to_numpy(), r=radius)
        for k, idx in zip(todo, nbrs):
            if len(idx) >= min_tr:
                target[k] = np.median(prices[idx])
                level[k] = f"r{radius}"
                ntr[k] = len(idx)

    # 동 단위 폴백 (반경서 못 채운 필지)
    todo = np.where(level == "missing")[0]
    dmap = pc["dong"].map(dong_med).to_numpy()
    for k in todo:
        if not np.isnan(dmap[k]):
            target[k] = dmap[k]
            level[k] = "dong"

    return pd.DataFrame({"pnu": pc["pnu"].values, "target_pyung": target,
                         "agg_level": level, "n_trades": ntr})

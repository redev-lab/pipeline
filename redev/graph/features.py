"""features.py — 노드 피처 (심장1 입력, ★각 (필지,시점) 행의 t 기준).

역할: 라벨 테이블의 (PNU, t) 행마다 그 시점 t의 필지 상태를 피처 벡터로 만든다.
그래프는 정적(필지 1개=노드 1개)이고, t는 *피처만* 고른다 — 같은 PNU 두 t면 두 피처
벡터(R5, 키스톤 §2-3). 건물 파생 피처(노후도·호수밀도)는 ★전부 사용승인일≤t 필터
(보강1, R1 누수 차단). 설계: docs/design/graph.md §4.

★R9: 베이스라인(XGBoost+이웃집계)도 이 동일 v1 피처만 쓴다(통제 비교 공정성).
"""
from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import STRtree

from redev.config import load_graph_config
from redev.data.aging import old_ratio_by_parcel

# v1 피처 컬럼(graph.md §3). 데이터 의존(용도지역·공시지가·역거리·배제)은 v1.1.
FEATURE_COLUMNS = ["aging", "area_m2", "compactness", "bldg_density", "road_abut"]


def _buildings_as_of(buildings: pd.DataFrame, t: int) -> pd.DataFrame:
    """사용승인일 ≤ t 건물만 (★보강1: 모든 건물 파생 피처의 공통 시점필터).

    노후도만 시점필터하면 호수밀도에 t 이후 신축 빌라가 새어 R1 위반. 건물에서 나오는
    피처는 전부 이걸 거친다.
    """
    b = buildings[buildings["approval_year"].notna()]
    return b[b["approval_year"] <= t]


def _compactness(geom) -> float:
    """Polsby-Popper 형상지수 = 4π·면적/둘레² (1=원, 낮을수록 길쭉·복잡).

    왜 후보 경계에 의미: 길쭉·복잡한 필지(낮은 값)는 비효율 이용·정비 필요 신호.
    """
    p = geom.length
    return float(4 * math.pi * geom.area / (p * p)) if p > 0 else 0.0


def _road_abutting(node_gdf: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> pd.Series:
    """접도: 각 필지가 도로(지목='도') 필지와 경계 공유하나 (1/0). STRtree 벡터화.

    왜 후보 경계에 의미: 접도율(도로 접한 필지 비율)이 낮으면 맹지·소방 취약 →
    주택정비형 재개발 요건(접도율 40%↓). 도로에 안 닿은 필지가 모이면 정비 필요.
    """
    if len(roads) == 0:
        return pd.Series(0, index=node_gdf.index)
    tree = STRtree(roads.geometry.values)               # 도로 공간인덱스
    qi, _tj = tree.query(node_gdf.geometry.values, predicate="touches")
    abut = np.zeros(len(node_gdf), dtype=int)
    abut[np.unique(qi)] = 1
    return pd.Series(abut, index=node_gdf.index)


def node_features(
    label_rows: pd.DataFrame,
    parcels: gpd.GeoDataFrame,
    buildings: pd.DataFrame,
    *,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """(필지,시점) 행마다 t 기준 노드 피처 → [pnu, t, + FEATURE_COLUMNS].

    입력: label_rows(pnu,t), parcels(geom·지목, 도로 포함 — 접도용), buildings(정제).
    출력: label_rows 순서 정렬 피처 테이블. Phase 3 GNN/베이스라인이 텐서로 변환.
    """
    cfg = cfg or load_graph_config()
    rows = label_rows[["pnu", "t"]].copy().reset_index(drop=True)
    psub = parcels[parcels["pnu"].isin(set(rows["pnu"]))].copy()

    # ── 정적 피처(t 무관): 면적·형상·접도 ──
    psub["area_m2"] = psub.geometry.area
    psub["compactness"] = psub.geometry.map(_compactness)
    psub["road_abut"] = _road_abutting(psub, parcels[parcels["jimok"] == "도"])
    static = psub.set_index("pnu")[["area_m2", "compactness", "road_abut"]]
    rows = rows.join(static, on="pnu")
    area_by = psub.set_index("pnu")["area_m2"]

    # ── 시점 피처(건물 파생, t별 as-of-t): 노후도·호수밀도(동/ha) ──
    # 왜: 노후도=낡은 동네(재개발 환경 핵심), 호수밀도=헥타르당 동수(과밀 노후, 요건 60↑).
    rows["aging"] = 0.0
    rows["bldg_density"] = 0.0
    for t, _grp in rows.groupby("t"):
        if pd.isna(t):
            continue
        ti = int(t)
        ratio = old_ratio_by_parcel(buildings, ti, weight="area")   # 노후도 as-of-t
        cnt = _buildings_as_of(buildings, ti).groupby("pnu").size()  # 동수 as-of-t (보강1)
        density = (cnt / area_by * 10000.0)                          # 동/ha
        mask = rows["t"] == t
        rows.loc[mask, "aging"] = rows.loc[mask, "pnu"].map(ratio).fillna(0.0).values
        rows.loc[mask, "bldg_density"] = rows.loc[mask, "pnu"].map(density).fillna(0.0).values
    return rows[["pnu", "t"] + FEATURE_COLUMNS]

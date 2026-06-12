"""zoning.py — 용도지역(AL_D124) 적재 + 세분류 ordinal 피처 (v1.1 ② 입지/규제 축).

설계: features_v1_1.md §1-2. 필지 대표점 ⨯ 용도지역 폴리곤 공간조인 → 세분류(A6)를 주거밀도
ordinal로(config). ★2026 스냅샷이라 과거 라벨에 붙이면 종상향 역류 가능(labels §13) — 근사
수용 + leakage_ablation으로 영향 측정. EPSG:5186 일치(변환 불요). 정적 피처(t 무관, 한 번 조인).
"""
from __future__ import annotations

import os
from functools import lru_cache

import geopandas as gpd
import pandas as pd
from pyogrio import read_dataframe

from redev.config import load_features_config

_ZIP = "도시구역 국토계획 전부.zip"
_SHP = "AL_D124_00_20260609.shp"
_RAW = "_data/raw/추가데이터"


def _vsizip() -> str:
    return f"/vsizip/{os.path.abspath(os.path.join(_RAW, _ZIP))}/{_SHP}"


@lru_cache(maxsize=None)
def load_zoning(sigungu_codes: tuple) -> gpd.GeoDataFrame:
    """용도지역 폴리곤(4구) → [a6(세분류명), geometry] EPSG:5186. A4=시군구로 필터."""
    g = read_dataframe(_vsizip(), encoding="cp949", columns=["A4", "A6"])
    g = g[g["A4"].astype(str).isin(set(sigungu_codes))].rename(columns={"A6": "a6"})
    return g[["a6", "geometry"]]


def zoning_features(parcels: gpd.GeoDataFrame, *, cfg=None) -> pd.DataFrame:
    """필지 → [pnu, zoning_ord, zoning_missing, zoning_name]. 대표점 within 용도지역 폴리곤.

    zoning_ord: 세분류 ordinal(주거밀도 순, config). 미매핑(녹지·공업 등)=default_other(0).
    zoning_missing: 공간조인 미스(어느 폴리곤에도 안 듦)=1. zoning_name: A6(stage1 FAR 룩업용).
    """
    zc = (cfg or load_features_config())["zoning"]
    codes = tuple(sorted(parcels["pnu"].str[:5].unique()))
    zones = load_zoning(codes)
    pts = gpd.GeoDataFrame(parcels[["pnu"]].copy(),
                           geometry=parcels.geometry.representative_point(), crs=parcels.crs)
    j = gpd.sjoin(pts, zones, predicate="within", how="left")
    # ★중첩(generic '도시지역' + 세분류) 시 세분류 우선 — ordinal 매핑된 행을 먼저 keep.
    j["_ord"] = j["a6"].map(zc["ordinal"])
    j = j.sort_values("_ord", na_position="last").drop_duplicates("pnu", keep="first")
    j["zoning_name"] = j["a6"]
    j["zoning_missing"] = j["a6"].isna().astype(int)
    j["zoning_ord"] = j["a6"].map(zc["ordinal"]).fillna(zc["default_other"]).astype(float)
    return j[["pnu", "zoning_ord", "zoning_missing", "zoning_name"]].reset_index(drop=True)

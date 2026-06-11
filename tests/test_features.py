"""graph/features 회귀 테스트 — 보강1(건물피처 as-of-t)·형상 고정.

실행: python -m pytest tests/test_features.py
"""
import math

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from redev.graph.features import _buildings_as_of, _compactness, node_features


def test_buildings_as_of_filters():
    """사용승인일 ≤ t 건물만 (보강1 공통 시점필터)."""
    b = pd.DataFrame([
        {"pnu": "A", "approval_year": 1985},
        {"pnu": "A", "approval_year": 2020},
    ])
    assert len(_buildings_as_of(b, 2010)) == 1     # 1985만
    assert len(_buildings_as_of(b, 2025)) == 2


def test_compactness_square_vs_thin():
    assert abs(_compactness(box(0, 0, 1, 1)) - math.pi / 4) < 0.01   # 정사각 ≈0.785
    assert _compactness(box(0, 0, 10, 0.1)) < 0.2                    # 길쭉 → 낮음


def test_node_features_aging_t_dependent():
    """★보강1/R1: 같은 필지의 노후도가 t에 따라 다르다(1985 건물이 30년 임계 넘음)."""
    parcels = gpd.GeoDataFrame(
        [{"pnu": "A", "jimok": "대", "geometry": box(0, 0, 10, 10)}],
        geometry="geometry", crs="EPSG:5186",
    )
    buildings = pd.DataFrame([
        {"pnu": "A", "approval_year": 1985, "structure": "rc", "gross_floor_area": 100},
    ])
    rows = pd.DataFrame({"pnu": ["A", "A"], "t": [2010, 2026]})
    feat = node_features(rows, parcels, buildings)
    a2010 = feat.loc[feat.t == 2010, "aging"].iloc[0]
    a2026 = feat.loc[feat.t == 2026, "aging"].iloc[0]
    assert a2010 == 0.0 and a2026 == 1.0          # 1985 rc: 2010=25yr<30, 2026=41yr>30
    assert feat["area_m2"].nunique() == 1          # 정적 피처는 t 무관 동일

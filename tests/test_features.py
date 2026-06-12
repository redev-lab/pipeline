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


def _stub_v11(monkeypatch):
    """v1.1 직교 피처 소스를 합성 기본값으로(실데이터 로드·sjoin 회피, 단위테스트 격리)."""
    import redev.data.ingest.land_price as LP
    import redev.data.ingest.rail as RAIL
    import redev.graph.features as F
    monkeypatch.setattr(F, "_v11_static", lambda p: {
        "zoning_ord": pd.Series(dtype=float), "zoning_missing": pd.Series(dtype=float),
        "centroid": p.set_index("pnu").geometry.centroid})
    monkeypatch.setattr(LP, "land_price_features",
                        lambda rows, **k: pd.DataFrame({"land_pct": [0.5] * len(rows), "land_missing": [1] * len(rows)}))
    monkeypatch.setattr(RAIL, "rail_features",
                        lambda rows, parcels, **k: pd.DataFrame({"rail_prox": [0.5] * len(rows)}))


def test_node_features_aging_t_dependent(monkeypatch):
    """★보강1/R1: 같은 필지의 노후도가 t에 따라 다르다(1985 건물이 30년 임계 넘음)."""
    _stub_v11(monkeypatch)
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

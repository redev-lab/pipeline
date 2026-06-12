"""stage1 회귀 테스트 — 루브릭·경로 분류(합성 클러스터, 결정론 검증).

실행: python -m pytest tests/test_stage1.py
"""
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from redev.rules.stage1 import classify_path, cluster_metrics, score_cluster


def _parcels(specs):
    # specs: list of (pnu, jimok, w, h)
    rows = [{"pnu": p, "jimok": j, "geometry": box(0, 0, w, h)} for p, j, w, h in specs]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:5186")


def test_old_landlocked_small_to_redevelopment():
    """낡은 소형 필지 클러스터(노후연면적1.0·과소·맹지) → 재개발."""
    par = _parcels([("A", "대", 5, 10), ("B", "대", 5, 10)])     # 각 50㎡(과소<90)
    bld = pd.DataFrame([
        {"pnu": "A", "approval_year": 1980, "structure": "rc", "gross_floor_area": 100},
        {"pnu": "B", "approval_year": 1980, "structure": "rc", "gross_floor_area": 100},
    ])
    out = score_cluster(["A", "B"], par, bld, current_year=2026)
    m = out["metrics"]
    assert m["old_area_ratio"] == 1.0 and m["undersized_ratio"] == 1.0
    assert out["housing_eligible"] and out["path"] == "재개발"
    assert len(out["caveats"]) == 4                       # 한계 4개 동봉


def test_new_large_to_none():
    """신축 대형 필지 → 모든 요건 미달 → 해당없음."""
    par = _parcels([("A", "대", 10, 20), ("B", "대", 10, 20)])   # 각 200㎡
    bld = pd.DataFrame([
        {"pnu": "A", "approval_year": 2020, "structure": "rc", "gross_floor_area": 400},
        {"pnu": "B", "approval_year": 2020, "structure": "rc", "gross_floor_area": 400},
    ])
    out = score_cluster(["A", "B"], par, bld, current_year=2026)
    assert out["metrics"]["old_area_ratio"] == 0.0
    assert not out["housing_eligible"] and not out["urban_eligible"]
    assert out["path"] == "해당없음"


def test_moderately_old_to_moa_taun():
    """노후 0.5(신축혼재)·대형·광역 미달 → 모아타운·소규모정비."""
    par = _parcels([("A", "대", 10, 20)])                 # 200㎡(비과소)
    bld = pd.DataFrame([
        {"pnu": "A", "approval_year": 1980, "structure": "rc", "gross_floor_area": 100},
        {"pnu": "A", "approval_year": 2020, "structure": "rc", "gross_floor_area": 100},
    ])  # 면적가중 노후 = 100/200 = 0.5
    out = score_cluster(["A"], par, bld, current_year=2026)
    assert out["metrics"]["old_area_ratio"] == 0.5
    assert not out["housing_eligible"] and not out["urban_eligible"]
    assert out["path"] == "모아타운·소규모정비"


def test_no_buildings_nan_not_zero():
    """건물 없는 클러스터 → 노후 NaN(거짓 0 아님), 해당없음."""
    par = _parcels([("A", "대", 10, 20)])
    m = cluster_metrics(["A"], par, pd.DataFrame(columns=["pnu", "approval_year", "structure", "gross_floor_area"]))
    assert pd.isna(m["old_area_ratio"])                   # 평가불가는 NaN

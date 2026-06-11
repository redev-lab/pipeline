"""graph/build 회귀 테스트 — 노드 제외(수정1)·인접 대칭·라벨 reconcile 고정.

실행: python -m pytest tests/test_build.py
"""
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from redev.graph.build import build_adjacency, node_parcels, reconcile_labels_to_graph

_CFG = {"node_jimok": {"exclude": ["도", "천"], "building_overrides_jimok": False}}


def _gdf(rows):
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:5186")


def test_node_parcels_excludes_jimok():
    """비주거 지목(도·천)은 노드에서 제외, 대·임은 유지(수정1)."""
    g = _gdf([
        {"pnu": "1", "jimok": "대", "geometry": box(0, 0, 1, 1)},
        {"pnu": "2", "jimok": "도", "geometry": box(1, 0, 2, 1)},   # 도로 → 제외
        {"pnu": "3", "jimok": "임", "geometry": box(2, 0, 3, 1)},   # 임야 → 유지
    ])
    keep = node_parcels(g, cfg=_CFG)
    assert set(keep["pnu"]) == {"1", "3"}


def test_adjacency_symmetric_no_selfloop():
    """경계 공유 두 필지 → 양방향 엣지(대칭), 자기루프 0."""
    g = _gdf([
        {"pnu": "A", "geometry": box(0, 0, 1, 1)},
        {"pnu": "B", "geometry": box(1, 0, 2, 1)},   # A와 x=1에서 경계 공유
        {"pnu": "C", "geometry": box(5, 5, 6, 6)},   # 고립
    ])
    ei = build_adjacency(g, buffer_m=0.0)
    edges = set(map(tuple, ei.t().tolist()))
    assert (0, 1) in edges and (1, 0) in edges     # 대칭
    assert all(a != b for a, b in edges)           # 자기루프 0
    assert ei.shape[1] == 2                         # A-B 양방향만(C 고립)


def test_reconcile_drops_non_node_labels():
    """그래프 노드에 없는 라벨(도로 등)은 drop, 나머지 유지."""
    tbl = pd.DataFrame({
        "pnu": ["A", "B", "ROAD"],
        "certainty": ["positive", "uncertain", "positive"],
    })
    kept, rep = reconcile_labels_to_graph(tbl, node_pnus={"A", "B"})
    assert set(kept["pnu"]) == {"A", "B"}
    assert rep["dropped_non_node"] == 1

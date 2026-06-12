"""rail 회귀 테스트 — as-of-t 노선 필터(mock 역, 파일 없이).

실행: python -m pytest tests/test_rail.py
"""
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import redev.data.ingest.rail as rail


def test_asof_t_filters_recent_line(monkeypatch):
    """★2010 라벨은 2017 개통 우이신설역을 못 본다 — as-of-t 누수 차단."""
    st = pd.DataFrame({
        "name": ["old", "new"], "line": ["1호선", "우이신설선"],
        "x": [1000.0, 0.0], "y": [0.0, 0.0], "open_year": [2000, 2017],
        "open_source": ["pre2001", "compiled"],
    })
    monkeypatch.setattr(rail, "load_stations", lambda: st)
    parcels = gpd.GeoDataFrame({"pnu": ["A"]}, geometry=[box(0, 0, 2, 2)], crs="EPSG:5186")  # centroid (1,1)

    r10 = rail.rail_features(pd.DataFrame({"pnu": ["A"], "t": [2010]}), parcels)
    r20 = rail.rail_features(pd.DataFrame({"pnu": ["A"], "t": [2020]}), parcels)
    assert r10["rail_dist_m"].iloc[0] > 900          # 2010: old만(우이신설 미개통) → 멀다
    assert r20["rail_dist_m"].iloc[0] < 100           # 2020: 우이신설 개통 → 가깝다
    assert r20["rail_src"].iloc[0] == "compiled"      # 최근접이 compiled(2001+) 노선

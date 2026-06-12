"""zoning 회귀 테스트 — 세분류 우선·ordinal 매핑·미스 플래그(합성 폴리곤).

실행: python -m pytest tests/test_zoning.py
"""
import geopandas as gpd
from shapely.geometry import box

import redev.data.ingest.zoning as zoning


def test_specific_over_generic_and_missing(monkeypatch):
    parcels = gpd.GeoDataFrame(
        {"pnu": ["1129000000000000001", "1129000000000000002", "1129000000000000003"]},
        geometry=[box(0, 0, 1, 1), box(10, 0, 11, 1), box(100, 0, 101, 1)], crs="EPSG:5186")
    # A: 도시지역+제2종일반 중첩 / B: 자연녹지 / C: 밖
    zones = gpd.GeoDataFrame(
        {"a6": ["도시지역", "제2종일반주거지역", "자연녹지지역"]},
        geometry=[box(0, 0, 2, 2), box(0, 0, 2, 2), box(10, 0, 12, 2)], crs="EPSG:5186")
    monkeypatch.setattr(zoning, "load_zoning", lambda codes: zones)
    zf = zoning.zoning_features(parcels).set_index("pnu")

    assert zf.loc["1129000000000000001", "zoning_ord"] == 4      # ★세분류(제2종일반=4) 우선, generic 아님
    assert zf.loc["1129000000000000002", "zoning_ord"] == 0      # 자연녹지=비주거 0
    assert zf.loc["1129000000000000002", "zoning_missing"] == 0
    assert zf.loc["1129000000000000003", "zoning_missing"] == 1  # 폴리곤 밖 → 미스

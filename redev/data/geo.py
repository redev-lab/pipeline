"""geo.py — 좌표계 단일 닻 (지뢰1: CRS 불일치 방어).

역할: 모든 ingest 리더가 geometry를 읽으면 **여기를 거쳐 EPSG:5186으로**
통일된 뒤에야 하위(공간조인)로 간다. 레이어마다 좌표계가 다르면(지적도 5186 vs
의제처리 2097) 같은 지점이 수 미터 어긋나, "구역 폴리곤 ∩ 필지" 조인이 *조용히*
틀린 필지에 라벨을 붙인다(에러 없이 결과만 오염 — PNU float 지뢰와 같은 급).

왜 5186인가: 라벨의 등뼈가 연속지적도이고 그 기준계가 EPSG:5186(Korea2000 /
GRS80 / 중부원점2010)이다. 등뼈에 모두를 맞춘다.
"""

import geopandas as gpd

# 좌표계 닻. 모든 공간 레이어를 이 좌표계로 통일한다.
TARGET_CRS = "EPSG:5186"  # Korea 2000 / Central Belt 2010 (연속지적도 기준계)


def to_target_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """GeoDataFrame을 TARGET_CRS(5186)로 재투영한다 (좌표계 관문).

    - CRS가 비어 있으면(미상) 추측하지 않고 raise — .prj 누락은 데이터 문제다.
    - 이미 5186이면 no-op(불필요한 변환·부동소수 오차 방지).
    - 그 외(예: 2097)면 to_crs로 변환. to_crs = 좌표를 다른 측지계로 수학적으로
      재계산하는 호출(단순 단위변환이 아니라 데이텀 변환 포함).
    """
    if gdf.crs is None:
        raise ValueError(
            "CRS 미상 GeoDataFrame — 원천 .prj 확인 필요. 좌표계를 추측하지 않는다."
        )
    if gdf.crs.to_epsg() == 5186:
        return gdf
    return gdf.to_crs(TARGET_CRS)

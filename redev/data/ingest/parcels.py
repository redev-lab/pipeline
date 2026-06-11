"""parcels.py — 연속지적도(AL_D002) 적재 (필지 스파인).

역할: 파이프라인의 등뼈. 모든 라벨·그래프가 올라타는 PNU·geometry의 단일 출처.
더불어 (시군구,동,본번,부번)→PNU 인덱스를 제공해, 해제구역의 위치(지번)을 PNU로
해석한다(지뢰3의 해제 부분).

AL_D002 매핑(실데이터 EDA 확정): A1=PNU, A3=법정동주소, A4=지번, A7=시군구코드.
설계: docs/design/ingest_parcels.md
"""

from __future__ import annotations

import re

import geopandas as gpd
import pandas as pd
from pyogrio import read_dataframe

from redev.data.geo import to_target_crs
from redev.data.pnu import normalize_pnu, parse_pnu

_COLUMN_MAP = {"A1": "pnu", "A3": "dong_addr", "A4": "jibun", "A5": "jimok_raw", "A7": "sigungu"}
_SRC_COLUMNS = list(_COLUMN_MAP.keys())


def _parse_jimok(s) -> str | None:
    """A5('179-3대'/'산39-25임') → 지목 1자('대'/'임'). 끝의 한글.

    역할: 그래프 노드 제외(도로·하천 등 비주거)·접도 피처(지목='도' 인접)의 키.
    """
    import re
    m = re.search(r"([가-힣]+)$", str(s))
    return m.group(1) if m else None


def _where_clause(sigungu_codes) -> str:
    """시군구코드 집합 → OGR SQL where 절 (구 단위만 읽어 메모리 절약, R10).

    개념: SHP을 통째로 안 올리고, 드라이버 레벨에서 A7(시군구) 필터를 태운다.
    """
    quoted = ",".join("'%s'" % c for c in sigungu_codes)
    return f"A7 IN ({quoted})"


def _parse_dong(addr) -> str | None:
    """법정동주소 '서울특별시 은평구 수색동' → '수색동' (마지막 토큰)."""
    if not isinstance(addr, str) or not addr.strip():
        return None
    return addr.strip().split()[-1]


def _parse_jibun(s) -> tuple[int, int] | None:
    """지번 문자열 → (본번, 부번). '179-3'→(179,3), '711'→(711,0).

    역할: 해제 위치('수유동 711')와 같은 키로 맞추기 위한 정규화.
    """
    if not isinstance(s, str):
        return None
    m = re.match(r"^\s*(\d+)(?:-(\d+))?\s*$", s)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or 0)


def load_parcels(
    path: str, sigungu_codes, *, with_geometry: bool = True
) -> tuple[gpd.GeoDataFrame | pd.DataFrame, dict]:
    """연속지적도 SHP → 정제 필지 테이블 + 위생 리포트 (스파인).

    입력:
      path : GDAL 경로('/vsizip/<zip>/<...>.shp' 가능).
      sigungu_codes : 읽을 자치구 코드 집합(config.training_sigungu_codes() 등).
      with_geometry : True면 폴리곤도 읽어 5186 통일(그래프·공간조인용).

    출력:
      gdf : [pnu, dong_addr, jibun, sigungu] (+geometry) — 4구만, 표준 PNU.
      report : 필지 수·PNU drop·동/지번 결측 등.
    """
    where = _where_clause(sigungu_codes)
    if with_geometry:
        # gpd.read_file → pyogrio 백엔드. where로 구 필터, cp949(.cpg 오기재 대비).
        gdf = gpd.read_file(path, columns=_SRC_COLUMNS, where=where, encoding="cp949")
        gdf = to_target_crs(gdf)                 # 좌표계 닻(지뢰1). 이미 5186이면 no-op.
        gdf = gdf.rename(columns=_COLUMN_MAP)
        # 깨진 geometry 복구(self-intersect 등) — make_valid는 위상학적으로 유효화.
        gdf["geometry"] = gdf.geometry.make_valid()
    else:
        gdf = read_dataframe(
            path, columns=_SRC_COLUMNS, where=where, read_geometry=False, encoding="cp949"
        ).rename(columns=_COLUMN_MAP)

    total = len(gdf)
    # PNU 표준화 관문: A1 → 19자리 문자열(불량은 None → drop).
    def _safe(v):
        try:
            return normalize_pnu(v)
        except (TypeError, ValueError):
            return None

    gdf["pnu"] = gdf["pnu"].map(_safe)
    gdf["jimok"] = gdf["jimok_raw"].map(_parse_jimok)   # 지목(노드제외·접도용)
    gdf = gdf.drop(columns="jimok_raw")
    n_bad = int(gdf["pnu"].isna().sum())
    gdf = gdf[gdf["pnu"].notna()].copy()

    report = {
        "total_rows": total,
        "dropped_bad_pnu": n_bad,
        "kept_rows": len(gdf),
        "sigungu_counts": gdf["sigungu"].value_counts().to_dict(),
        "dong_missing": int(gdf["dong_addr"].isna().sum()),
        "jibun_missing": int(gdf["jibun"].isna().sum()),
        "jimok_counts": gdf["jimok"].value_counts().head(15).to_dict(),
    }
    return gdf, report


def build_jibun_index(parcels) -> dict:
    """(시군구, 동, 본번, 부번) → PNU 인덱스 (해제 지번매칭용, 지뢰3 해제 부분).

    역할: 해제구역의 위치('수유동 711')를 PNU로 해석하는 룩업. 지번에서 PNU를
    *재구성하지 않고* 지적도 실값으로 인덱스를 만든다(산/일반 필지구분 깨짐 방지).

    충돌(같은 번-지에 일반·산 둘 다) 시 일반 필지(필지구분 1)를 우선한다 — 해제
    대상은 토지대장(일반)이다.
    """
    index: dict = {}
    for pnu, dong_addr, jibun, sigungu in zip(
        parcels["pnu"], parcels["dong_addr"], parcels["jibun"], parcels["sigungu"]
    ):
        dong = _parse_dong(dong_addr)
        bj = _parse_jibun(jibun)
        if dong is None or bj is None:
            continue
        key = (sigungu, dong, bj[0], bj[1])
        if key in index:
            # 일반(필지구분 1) 우선: 기존이 산이고 새 게 일반이면 교체.
            if parse_pnu(index[key])["is_san"] and not parse_pnu(pnu)["is_san"]:
                index[key] = pnu
            continue
        index[key] = pnu
    return index

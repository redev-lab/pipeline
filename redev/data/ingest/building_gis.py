"""building_gis.py — GIS건물통합정보(AL_D010) 적재 (노후도 원천, R1).

역할: 원천 SHP를 읽어 aging.py 가 먹을 *정제 건물 테이블*로 변환한다. 노후도에
필요한 최소 컬럼만 뽑고(메모리, R10), PNU를 표준화하고, 깨진 행을 거른다.

A-코드 매핑(실데이터 EDA 확정): A2=PNU, A13=사용승인일, A11=구조명,
A14=연면적, A7=토지대장구분(일반/산), A23=시군구코드.
  주의: A7은 처음에 '집합건물 구분'으로 오인했으나 실제 값은 일반/산 →
  필지의 토지/임야 구분(PNU 필지구분과 동일 정보)이다. 집합건물 식별은
  별도 필드 필요(미해결, R10 — 노후도엔 영향 없음).
설계: docs/design/ingest_building_gis.md
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from redev.data.geo import to_target_crs
from redev.data.pnu import normalize_pnu

# 원천 A-코드 → 정제 컬럼명. (읽을 때 이 컬럼만 선택해 1.2GB를 다 안 올린다.)
_COLUMN_MAP = {
    "A2": "pnu",
    "A13": "approval_raw",
    "A11": "structure_raw",
    "A14": "gross_floor_area",
    "A7": "land_div",  # 토지대장구분(일반/산) — 집합건물 플래그 아님(EDA로 확인)
    "A23": "sigungu",
}
_SRC_COLUMNS = list(_COLUMN_MAP.keys())

# 구조명 → 내구성 분류. rc(내구구조)는 긴 경과연수 기준(config rc_years),
# 그 외는 짧은 기준(other_years). ★연수 임계값은 config(규칙5), 분류 키워드만 여기.
_RC_KEYWORDS = ("철근콘크리트", "철골", "강구조", "라멘", "프리캐스트", "피씨", "p.c", "pc조")


def _classify_structure(name) -> str:
    """구조명(cp949 디코드된 문자열) → 'rc' | 'other'.

    경량철골 등 경량구조는 내구성이 낮아 'other'로 뺀다(철골 키워드보다 우선).
    결측·미상은 'other'(보수적: 더 짧은 경과연수 → 노후 판정이 관대 → R2 오탐↓).
    """
    if not isinstance(name, str) or not name.strip():
        return "other"
    s = name.strip()
    if "경량" in s:  # 경량철골조 등
        return "other"
    return "rc" if any(k in s for k in _RC_KEYWORDS) else "other"


def _parse_approval_year(s) -> object:
    """A13 사용승인일('1991-09-02' 등) → 연도(int). 결측·이상치 → pd.NA.

    노후도(R1)의 시점 닻은 '연도'면 충분(labels.md 결정3). 앞 4자리를 연도로.
    """
    if s is None:
        return pd.NA
    s = str(s).strip()
    if len(s) < 4 or not s[:4].isdigit():
        return pd.NA
    y = int(s[:4])
    if y < 1900 or y > 2100:  # 0000·9999 같은 더미·오류값 차단
        return pd.NA
    return y


def _safe_normalize_pnu(v) -> object:
    """normalize_pnu를 적용하되 실패(타입·길이 오류)는 None으로 표시(드롭 대상)."""
    try:
        return normalize_pnu(v)
    except (TypeError, ValueError):
        return None


def load_buildings(path: str, *, with_geometry: bool = False) -> tuple[pd.DataFrame, dict]:
    """GIS건물통합정보 SHP → 정제 건물 테이블 + 위생 리포트.

    역할: ingest의 첫 리더. 출력 df는 aging.old_ratio_by_parcel 이 바로 먹는다.

    입력:
      path : GDAL이 읽는 경로. zip 내부는 '/vsizip/<zip>/<...>.shp' 형식.
      with_geometry : True면 건물 외곽 폴리곤도 읽어 5186으로 reproject(지뢰1).
        노후도(R1)엔 geometry가 불필요 → 기본 False로 1.2GB를 다 안 올린다(R10).

    출력:
      df : [pnu, approval_year(Int64), structure('rc'/'other'),
            gross_floor_area(float), land_div(일반/산), sigungu] (+ geometry if 요청)
      report : 적재 위생 통계(총건수·드롭·결측률·분포) — 정직성.
    """
    # ── 읽기: 필요한 컬럼만. cp949(.cpg 없어 기본 latin로 깨짐) 명시. ──
    if with_geometry:
        # geopandas.read_file → 내부적으로 pyogrio. geometry 포함.
        gdf = gpd.read_file(path, columns=_SRC_COLUMNS, encoding="cp949")
        gdf = to_target_crs(gdf)            # ★좌표계 닻(지뢰1): 5186으로 통일
        df = gdf.rename(columns=_COLUMN_MAP)
    else:
        # 속성만(geometry 제외) → 빠르고 가볍다.
        from pyogrio import read_dataframe
        df = read_dataframe(
            path, columns=_SRC_COLUMNS, read_geometry=False, encoding="cp949"
        ).rename(columns=_COLUMN_MAP)

    total = len(df)

    # ── 정제: PNU 표준화·연도 파싱·구조 분류·연면적 수치화 ──
    df["pnu"] = df["pnu"].map(_safe_normalize_pnu)
    df["approval_year"] = df["approval_raw"].map(_parse_approval_year).astype("Int64")
    df["structure"] = df["structure_raw"].map(_classify_structure)
    df["gross_floor_area"] = pd.to_numeric(df["gross_floor_area"], errors="coerce")

    # ── 위생(R10): PNU 결측·불량 행만 드롭(approval 결측은 aging이 시점필터로 처리) ──
    n_bad_pnu = int(df["pnu"].isna().sum())
    df = df[df["pnu"].notna()].copy()

    # 불필요한 원본 컬럼 정리
    df = df.drop(columns=["approval_raw", "structure_raw"])

    report = {
        "total_rows": total,
        "dropped_bad_pnu": n_bad_pnu,
        "kept_rows": len(df),
        "approval_year_missing_rate": round(float(df["approval_year"].isna().mean()), 4),
        "structure_counts": df["structure"].value_counts().to_dict(),
        "land_div_counts": df["land_div"].value_counts(dropna=False).to_dict(),
        "n_parcels": int(df["pnu"].nunique()),
    }
    return df, report

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

# ★국가표준 GIS건물(AL_D162 일반 + AL_D164 집합, 공공누리 1유형) 어댑터.
# 서울 AL_D010과 A-코드 레이아웃이 다르다(_experiments/gis_swap 실증: 서울 A13과 사용승인일
# 정확일치 99.9%·연면적 상관 0.9955·구조 100%). A7(토지대장구분) 대응 없음 → land_div는 None.
_COLUMN_MAP_NATIONAL = {
    "A1": "pnu",
    "A35": "approval_raw",     # 사용승인일(A34=허가/착공이 아님 — 실증으로 A35 확정)
    "A28": "structure_raw",
    "A24": "gross_floor_area",
    "A39": "sigungu",
}
_SRC_COLUMNS_NATIONAL = list(_COLUMN_MAP_NATIONAL.keys())

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

    return _finalize(df)


def _finalize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """원천 무관 공통 정제 — 서울(load_buildings)·국가표준(load_buildings_national) 공유.

    입력 df: [pnu, approval_raw, structure_raw, gross_floor_area, sigungu(, land_div)] 원시.
    """
    total = len(df)
    if "land_div" not in df.columns:
        df["land_div"] = None     # 국가표준엔 토지대장구분 없음(노후도·밀도 무관)

    # ── 정제: PNU 표준화·연도 파싱·구조 분류·연면적 수치화 ──
    df["pnu"] = df["pnu"].map(_safe_normalize_pnu)
    df["approval_year"] = df["approval_raw"].map(_parse_approval_year).astype("Int64")
    df["structure"] = df["structure_raw"].map(_classify_structure)
    df["gross_floor_area"] = pd.to_numeric(df["gross_floor_area"], errors="coerce")

    # ── 위생(R10): PNU 결측·불량 행만 드롭(approval 결측은 aging이 시점필터로 처리) ──
    n_bad_pnu = int(df["pnu"].isna().sum())
    df = df[df["pnu"].notna()].copy()

    # ── ★빈 폴리곤 제거: 연면적<=0/결측 AND 사용승인일 없음 = 부속·무허가·미등록 더미.
    #    호수밀도(stage1 len(cbld))를 부풀리는 노이즈 — 실측(_experiments/gis_swap): 빈 폴리곤이
    #    서울 호수밀도 통과율을 9%→23%로 부풀리고, 격자표본 AUC를 가렸다. AND 조건(둘 다 없을 때만)으로
    #    보수적 제거. 노후도(old_ratio)는 approval 결측을 이미 시점필터로 빼므로 이 제거로 불변.
    no_area = df["gross_floor_area"].isna() | (df["gross_floor_area"] <= 0)
    no_approval = df["approval_year"].isna()
    n_empty = int((no_area & no_approval).sum())
    df = df[~(no_area & no_approval)].copy()

    # 불필요한 원본 컬럼 정리
    df = df.drop(columns=["approval_raw", "structure_raw"])

    report = {
        "total_rows": total,
        "dropped_bad_pnu": n_bad_pnu,
        "dropped_empty_polygon": n_empty,
        "kept_rows": len(df),
        "approval_year_missing_rate": round(float(df["approval_year"].isna().mean()), 4),
        "structure_counts": df["structure"].value_counts().to_dict(),
        "land_div_counts": df["land_div"].value_counts(dropna=False).to_dict(),
        "n_parcels": int(df["pnu"].nunique()),
    }
    return df, report


def load_buildings_national(d162_path: str, d164_path: str, *, backfill_path: str | None = None,
                            with_geometry: bool = False) -> tuple[pd.DataFrame, dict]:
    """★국가표준 GIS건물(일반 AL_D162 + 집합 AL_D164) → 정제 건물 테이블 (1유형, 상업가능).

    역할: 서울 4유형(상업금지) load_buildings의 1유형 대체. 어댑터로 A-코드를 서울 레이아웃에
    맞춰 리네임(_COLUMN_MAP_NATIONAL) 후 _finalize 공유. 일반+집합 두 SHP를 합쳐 로드.

    backfill_path: 건축HUB 표제부 보충 parquet([pnu·approval_year·gross_floor_area·structure]).
      ★national이 누락한 PNU(노후밀집 실주거)의 사용승인일을 채워 zone_vectors 기준·점수 안정화(#3-b-2).
      national에 이미 있는 PNU는 중복 추가 안 함(보충만). 어댑터와 동일하게 _finalize가 정제.

    입력: d162_path(일반)·d164_path(집합) — vsizip 경로. with_geometry True면 5186으로 통일.
    출력: load_buildings와 동일 스키마 [pnu, approval_year, structure, gross_floor_area, land_div, sigungu].
    """
    frames = []
    for path in (d162_path, d164_path):
        if with_geometry:
            g = gpd.read_file(path, columns=_SRC_COLUMNS_NATIONAL, encoding="cp949")
            g = to_target_crs(g)            # ★EPSG:5186 통일(국가표준은 이미 5186이라 사실상 동일)
            frames.append(g.rename(columns=_COLUMN_MAP_NATIONAL))
        else:
            from pyogrio import read_dataframe
            frames.append(read_dataframe(
                path, columns=_SRC_COLUMNS_NATIONAL, read_geometry=False, encoding="cp949"
            ).rename(columns=_COLUMN_MAP_NATIONAL))
    df = pd.concat(frames, ignore_index=True)

    # ── backfill 통합(있으면): national 누락 PNU만 보충(중복 방지) ──
    import os
    if backfill_path and os.path.exists(backfill_path):
        bf = pd.read_parquet(backfill_path)
        bf = bf[bf["approval_year"].astype(str).str.len() == 4].copy()
        bf["pnu"] = bf["pnu"].astype(str).str.zfill(19)
        bf = bf[~bf["pnu"].isin(set(df["pnu"].astype(str)))]          # national에 없는 PNU만
        add = pd.DataFrame({"pnu": bf["pnu"], "approval_raw": bf["approval_year"].astype(str),
                            "structure_raw": bf["structure"],
                            "gross_floor_area": pd.to_numeric(bf["gross_floor_area"], errors="coerce"),
                            "sigungu": bf["pnu"].str[:5]})
        df = pd.concat([df, add], ignore_index=True)
    return _finalize(df)

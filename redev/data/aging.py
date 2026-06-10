"""aging.py — 노후도 as-of-t (시점 정합성의 단일 정의처, R1의 심장).

역할: "필지(또는 구역) 위 건물들을, **시점 t에 존재하던 것만으로**, t 시점 기준
노후 비율"을 계산한다. 이 한 가지 규율이 §5-R1(노후도 시점 누수)을 막는다.

왜 별도 모듈인가: 이 계산은 두 곳에서 똑같이 필요하다 —
  (1) labels.py 의 R2 오염 판정(positive인데 t-노후도가 컷 미만이면 완공/철거),
  (2) graph/features.py 의 노드 피처(Phase 2).
두 곳이 각자 계산하면 정의가 갈라진다. 시점 정합성은 단일 정의처에서만 나온다.

★raw → 정제 변환은 ingest(building_gis.py)의 몫. 이 모듈은 *정제된* 건물
스키마만 본다 (아래 계약 참조). 그래서 원천 파일 포맷과 무관하다.

────────────────────────────────────────────────────────────────────────────
정제 건물 테이블 계약 (building_gis.py 가 내놓아야 하는 컬럼):
  - pnu            : str(19)   필지 키 (redev.data.pnu 표준형)
  - approval_year  : int|NA    사용승인 연도 (원천 A13 사용승인일에서 파생).
                               결측 가능 — 결측 건물은 노후 판정에서 제외(아래 §주의).
  - structure      : str       'rc' | 'other' 로 *이미 분류된* 구조.
                               (철근콘크리트·철골 등 내구구조 → 'rc',
                                조적·목조·블록 등 → 'other'. 매핑은 ingest 책임.)
  - gross_floor_area : float|NA 연면적(㎡). 연면적 가중 노후비율에 사용(선택).
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import pandas as pd

from redev.config import load_thresholds

# 정제 건물 테이블의 필수 컬럼 (계약을 코드로 고정 — 깨지면 즉시 드러나게).
REQUIRED_COLUMNS = ("pnu", "approval_year", "structure")

_RC = "rc"        # 내구성 구조 (철근콘크리트·철골 등) → 더 긴 경과연수 기준
_OTHER = "other"  # 그 외 구조 (조적·목조 등) → 더 짧은 경과연수 기준


def _aging_thresholds() -> dict:
    """config에서 구조별 노후 경과연수 기준을 읽는다 (매직넘버 금지, 규칙5).

    반환: {'rc': rc_years, 'other': other_years}  (단위: 년)
    """
    th = load_thresholds()["building_aging"]
    return {_RC: th["rc_years"], _OTHER: th["other_years"]}


def _is_old_at(approval_year: int, structure: str, t: int, thr: dict) -> bool:
    """건물 하나가 시점 t에 '노후'인가 (구조별 경과연수 기준).

    t 시점 경과연수 = t - 사용승인연도. 구조별 기준(rc:30, other:20 등) 이상이면
    노후. ★t 이후 건물은 애초에 호출 전에 걸러진다(아래 old_ratio_as_of).
    """
    years_elapsed = t - approval_year
    # 알 수 없는 구조는 보수적으로 'other'(짧은 기준)로 보지 않고, 명시적으로 분기.
    threshold = thr.get(structure, thr[_OTHER])
    return years_elapsed >= threshold


def old_ratio_as_of(
    buildings: pd.DataFrame,
    t: int,
    *,
    weight: str = "area",
) -> float:
    """주어진 건물 집합의 '시점 t 노후 비율'을 계산한다 (R1의 핵심).

    역할: 한 필지(또는 한 구역)에 속한 건물들을 받아, **t에 존재하던 건물만으로**
    노후 비율을 낸다. 호출부가 buildings 를 어떻게 고르냐(필지 단위 / 구역 폴리곤
    내부)는 자유 — 이 함수는 순수하게 비율만 계산한다(테스트 용이).

    시점 정합성(R1): approval_year > t 인 건물(=t 이후 신축)과 approval_year 결측
    건물은 '존재하지 않은 것'으로 보고 제외한다. "t에 알 수 있던 정보만"(규칙8).

    weight:
      - 'area' : 연면적 가중 (노후·불량 *연면적* 비율 — housing_redevelopment 요건과 정합).
                 gross_floor_area 결측이 있으면 'count'로 자동 폴백.
      - 'count': 동수 기준 단순 비율.

    반환: 0.0~1.0 노후 비율. t에 존재한 건물이 하나도 없으면 NaN(판정 불가).
    """
    thr = _aging_thresholds()

    # ① 시점 필터: t 이후 신축·승인연도 결측 제외 (R1 닻).
    existed = buildings[buildings["approval_year"].notna()]
    existed = existed[existed["approval_year"] <= t]
    if existed.empty:
        return float("nan")  # t 시점에 평가할 건물이 없음 — 거짓 0을 내지 않는다.

    # ② 각 건물의 t 시점 노후 여부.
    is_old = existed.apply(
        lambda r: _is_old_at(int(r["approval_year"]), r["structure"], t, thr),
        axis=1,
    )

    # ③ 가중 비율. 연면적 가중이 원칙이나 결측 있으면 동수로 폴백.
    use_area = weight == "area" and "gross_floor_area" in existed.columns \
        and existed["gross_floor_area"].notna().all()
    if use_area:
        w = existed["gross_floor_area"]
        return float(w[is_old].sum() / w.sum())
    return float(is_old.mean())  # 동수 비율


def old_ratio_by_parcel(
    buildings: pd.DataFrame,
    t: int,
    *,
    weight: str = "area",
) -> pd.Series:
    """필지(PNU)별 t 시점 노후 비율 (배치 편의 헬퍼).

    역할: 건물 테이블 전체를 PNU로 묶어, 각 필지의 t 시점 노후 비율을 한 번에.
    labels.py 가 라벨 PNU들의 R2 오염을 일괄 판정할 때, features.py 가 노드
    피처를 만들 때 쓴다.

    반환: index=pnu, value=노후비율(0~1, 평가불가는 NaN) 인 Series.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in buildings.columns]
    if missing:
        raise KeyError(f"정제 건물 테이블에 필수 컬럼 없음: {missing} (계약 위반)")

    # groupby-apply: 각 PNU 그룹에 위 순수 함수를 적용. (개념: 필지 단위 집계)
    # include_groups=False: 그룹키('pnu')를 연산 프레임에서 빼 경고 제거(pandas 2.2+).
    # old_ratio_as_of는 pnu 컬럼을 안 쓰므로 빼도 안전하다.
    return buildings.groupby("pnu", sort=False).apply(
        lambda g: old_ratio_as_of(g, t, weight=weight),
        include_groups=False,
    )

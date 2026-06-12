"""stage1.py — Stage1 요건 룰셋 + 경로 분류 (Phase 4, 결정론 루브릭).

역할: 필지 클러스터(PNU 집합)를 받아 법적 정비 요건 충족도를 *투명하게* 점수화하고
경로(재개발 / 모아타운·소규모정비 / 해당없음)를 분류한다. 학습 아님 — 숫자는 전부
config 룩업·결정론 계산(규칙4·5). 노후·불량은 사용승인일 근사임을 출력에 명시(규칙4).

★시점: 추론용이라 *현재* 노후도 사용(학습 아니므로 R1 누수 개념 없음 — 현황 판정).
★모델 무관(R9): 클러스터를 인자로 받는 순수 함수 — infer 없이 단독 테스트. 설계: stage1.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from redev.config import load_thresholds
from redev.data.aging import old_ratio_as_of
from redev.graph.features import _road_abutting

# ★데이터 한계 — 출력에 항상 동봉(규칙4·R15 정직성). v1.1 정석데이터 입수 시 교체.
CAVEATS = [
    "노후·불량은 사용승인일 기반 근사(실제 구조안전 진단 아님).",
    "접도율은 도로폭 무관 인접(touches) 근사 — 조례 '4m 도로폭' 미반영(v1.1 도로망 폭 교체).",
    "호수밀도는 건물 '동' 단위 — '호(세대)' 아님, 집합건물 세대수 미반영(v1.1 교체).",
    "저밀이용은 절대 용적률 근사 — 정석은 법정용적률 대비(v1.1 용도지역 데이터 교체).",
]


def _f(x, n=2) -> str:
    return "NaN" if x is None or pd.isna(x) else f"{x:.{n}f}"


def cluster_metrics(pnu_set, parcels, buildings, *, current_year: int = 2026, cfg=None) -> dict:
    """클러스터의 요건 지표를 결정론 계산(순수). 설계: stage1.md §3.

    입력: pnu_set(클러스터), parcels(geom·지목·도로 포함), buildings(정제), current_year.
    출력: 지표 dict. 평가 불가(건물·필지 없음)는 NaN으로 — 거짓 0을 내지 않는다.
    """
    th = cfg or load_thresholds()
    ps = set(pnu_set)
    cpar = parcels[parcels["pnu"].isin(ps)].copy()
    cbld = buildings[buildings["pnu"].isin(ps)]
    n = len(cpar)

    cpar["_area"] = cpar.geometry.area
    area_ha = float(cpar["_area"].sum()) / 10_000.0

    old_area_ratio = old_ratio_as_of(cbld, current_year, weight="area")     # 현재시점 노후연면적(NaN 가능)
    abut_ratio = float(_road_abutting(cpar, parcels[parcels["jimok"] == "도"]).mean()) if n else float("nan")
    house_density = (len(cbld) / area_ha) if area_ha > 0 else float("nan")  # 동/ha

    under_cut = th["urban_redevelopment"]["undersized_lot_area_m2"]
    undersized_ratio = float((cpar["_area"] < under_cut).mean()) if n else float("nan")

    # 저밀이용: 필지별 용적률(Σ연면적/대지면적) < 컷. 건물 없는 필지=용적률0=저밀 포함.
    if len(cbld):
        gfa = pd.to_numeric(cbld["gross_floor_area"], errors="coerce").fillna(0.0).groupby(cbld["pnu"]).sum()
    else:
        gfa = pd.Series(dtype=float)
    far = cpar["pnu"].map(gfa).fillna(0.0).to_numpy() / cpar["_area"].replace(0, np.nan).to_numpy()
    far_cut = th["urban_redevelopment"]["low_density_far"]
    low_density_ratio = float((np.nan_to_num(far, nan=0.0) < far_cut).mean()) if n else float("nan")

    return {
        "n_parcels": n, "n_buildings": int(len(cbld)), "area_ha": round(area_ha, 3),
        "old_area_ratio": old_area_ratio, "abut_ratio": abut_ratio,
        "house_density": house_density, "undersized_ratio": undersized_ratio,
        "low_density_ratio": low_density_ratio,
    }


def _housing_eligible(m: dict, th: dict) -> tuple[bool, list]:
    """주택정비형: 노후연면적≥60% AND (접도율≤40% OR 호수밀도≥60). stage1.md §4."""
    h = th["housing_redevelopment"]
    oa, ab, hd = m["old_area_ratio"], m["abut_ratio"], m["house_density"]
    old_ok = pd.notna(oa) and oa >= h["old_building_area_ratio"]
    abut_ok = pd.notna(ab) and ab <= h["abutting_road_ratio_max"]
    dens_ok = pd.notna(hd) and hd >= h["house_density_min"]
    elig = bool(old_ok and (abut_ok or dens_ok))
    reasons = [
        f"노후연면적 {_f(oa)} {'≥' if old_ok else '<'} {h['old_building_area_ratio']}",
        f"접도율 {_f(ab)} {'≤' if abut_ok else '>'} {h['abutting_road_ratio_max']}",
        f"호수밀도 {_f(hd, 0)} {'≥' if dens_ok else '<'} {h['house_density_min']}",
    ]
    return elig, reasons


def _urban_eligible(m: dict, th: dict) -> tuple[bool, int, list]:
    """도시정비형: {노후도≥30%, 과소필지≥40%, 저밀≥50%} 중 2개 이상. stage1.md §4."""
    u = th["urban_redevelopment"]
    oa, us, ld = m["old_area_ratio"], m["undersized_ratio"], m["low_density_ratio"]
    c1 = pd.notna(oa) and oa >= u["old_building_ratio"]
    c2 = pd.notna(us) and us >= u["undersized_lot_ratio"]
    c3 = pd.notna(ld) and ld >= u["low_density_use_ratio"]
    count = int(c1) + int(c2) + int(c3)
    elig = count >= u["required_count"]
    reasons = [
        f"노후도 {_f(oa)} {'≥' if c1 else '<'} {u['old_building_ratio']}",
        f"과소필지 {_f(us)} {'≥' if c2 else '<'} {u['undersized_lot_ratio']}",
        f"저밀이용 {_f(ld)} {'≥' if c3 else '<'} {u['low_density_use_ratio']}",
        f"충족 {count}/{u['required_count']}개",
    ]
    return elig, count, reasons


def classify_path(m: dict, th: dict) -> dict:
    """요건 충족 → 경로 분류. 재개발 / 모아타운·소규모정비 / 해당없음. stage1.md §4."""
    h_elig, h_reasons = _housing_eligible(m, th)
    u_elig, u_count, u_reasons = _urban_eligible(m, th)
    oa = m["old_area_ratio"]
    if h_elig or u_elig:
        path = "재개발"
    elif pd.notna(oa) and oa >= th["moa_taun"]["min_old_ratio"]:
        path = "모아타운·소규모정비"           # 광역 미달이나 노후(신축혼재) — 광흥창 케이스
    else:
        path = "해당없음"
    return {
        "path": path, "housing_eligible": h_elig, "urban_eligible": u_elig,
        "housing_reasons": h_reasons, "urban_reasons": u_reasons,
    }


def score_cluster(pnu_set, parcels, buildings, *, current_year: int = 2026, cfg=None) -> dict:
    """공개 진입점 — 클러스터 → 지표·경로·근거·caveat. infer가 클러스터를 주면 호출.

    출력: {metrics, path, *_eligible, *_reasons, caveats}. 모든 한계는 caveats로 동봉(정직성).
    """
    th = cfg or load_thresholds()
    m = cluster_metrics(pnu_set, parcels, buildings, current_year=current_year, cfg=th)
    out = {"metrics": m, "caveats": list(CAVEATS)}
    out.update(classify_path(m, th))
    return out

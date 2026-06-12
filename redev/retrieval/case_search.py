"""case_search.py — 사례검색 (Phase 7, ① retrieval). 설계: docs/design/retrieval.md.

후보지 → "○○구역과 N% 유사 + 진행 이력". 비교 51구역뿐 → ★numpy 전수 코사인(DB·pgvector는
과한 도구, v2 전역서). 물리 피처 4개(노후도·면적·호수밀도·접도율)만 — t는 메타로만(질의측
미지정이라 유사도에 섞으면 '최근 선호' 몰래 가중). 숫자는 데이터 계산만(규칙4, LLM 미사용).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from redev.config import load_thresholds
from redev.rules.stage1 import cluster_metrics

# 유사도 물리 피처(키, 한글명). ★t 제외(설계 §1).
_FEATURES = [("old_area_ratio", "노후도"), ("area_ha", "면적"),
             ("house_density", "호수밀도"), ("abut_ratio", "접도율")]


@dataclass
class ZoneVectors:
    Z: np.ndarray          # [n_zones, 4] L2-정규화된 표준화 벡터
    meta: list             # [{zone_id, display_name, t, zone_type, completed}]
    mean: np.ndarray       # 표준화 통계(쿼리도 동일 적용)
    std: np.ndarray


def _raw_vec(m: dict) -> np.ndarray:
    return np.array([m.get(k, np.nan) for k, _ in _FEATURES], dtype=float)


def _zone_display_name(pnus, pnu2dong: dict, t) -> str | None:
    """★구역 표시명(§B-3) — 원시 zone_id(11590NTC…) 대신 사용자 표기. 구역의 최빈 법정동주소
    ('서울특별시 동작구 노량진동')에서 '서울특별시' 접두를 떼고 '자치구 동 일대 (지정연도)'로.
    zone_id 원시코드는 메타에만 남긴다(노출 안 함)."""
    from collections import Counter
    addrs = [pnu2dong.get(p) for p in pnus]
    addrs = [a.strip() for a in addrs if isinstance(a, str) and a.strip()]
    if not addrs:
        return None
    top = Counter(addrs).most_common(1)[0][0]
    parts = top.split()
    label = " ".join(parts[1:]) if parts and parts[0].startswith("서울") else top  # '서울특별시' 접두 제거
    return f"{label} 일대 ({t})" if t else f"{label} 일대"


def build_zone_vectors(zones: list, parcels, buildings, *, cfg=None) -> ZoneVectors:
    """51 지정구역 → 표준화·정규화 벡터 + 메타. zones=[{zone_id,pnus,t,zone_type}].

    완공여부 = 현재 노후도 < 컷(R2 탐지기 재사용 — 완공구역은 신축이라 낮음).
    """
    th = cfg or load_thresholds()
    cut = th["label_hygiene"]["min_old_ratio_for_positive"]
    pnu2dong = dict(zip(parcels["pnu"], parcels["dong_addr"]))     # §B-3 표시명용
    raws, meta = [], []
    for z in zones:
        m = cluster_metrics(z["pnus"], parcels, buildings, cfg=th)
        raws.append(_raw_vec(m))
        oa = m["old_area_ratio"]
        meta.append({"zone_id": z["zone_id"],
                     "display_name": _zone_display_name(z["pnus"], pnu2dong, z.get("t")),
                     "t": z.get("t"), "zone_type": z.get("zone_type"),
                     "completed": bool(oa is not None and not np.isnan(oa) and oa < cut)})
    raw = np.vstack(raws)
    mean = np.nanmean(raw, axis=0)
    std = np.nanstd(raw, axis=0)
    std[std == 0] = 1.0                                  # 상수 축 보호
    # ★표준화 *후* NaN→0(=평균): 표준화 전 0치환은 평균을 민다(수검 교정).
    Z = _normalize(np.nan_to_num((raw - mean) / std, nan=0.0))
    return ZoneVectors(Z, meta, mean, std)


def _normalize(X: np.ndarray) -> np.ndarray:
    """행별 L2 정규화(코사인 = 정규화 벡터의 내적)."""
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def _featurize(m: dict, zv: ZoneVectors) -> np.ndarray:
    """쿼리 metrics → 구역과 동일 통계로 표준화·정규화한 벡터(표준화 후 NaN→0=평균)."""
    v = np.nan_to_num((_raw_vec(m) - zv.mean) / zv.std, nan=0.0)
    return _normalize(v[None, :])[0]


def cosine_topk(qvec: np.ndarray, zv: ZoneVectors, *, k: int = 3) -> list:
    """상위 K 유사 구역 + 유사도 + ★기여축(표준화 벡터 per-축 곱 상위)."""
    sims = zv.Z @ qvec                                   # 정규화 내적 = 코사인
    order = np.argsort(-sims)[:k]
    out = []
    for i in order:
        contrib = zv.Z[i] * qvec                         # per-축 기여
        top_ax = [_FEATURES[j][1] for j in np.argsort(-contrib)[:2] if contrib[j] > 0]
        out.append({**zv.meta[i], "similarity": round(float(sims[i]), 3), "top_similar_axes": top_ax})
    return out


def search_cases(query_pnus, parcels, buildings, zv: ZoneVectors, *,
                 k: int = 3, sort_by_recency: bool = False, cfg=None) -> dict:
    """공개 진입점 — 후보 클러스터 → 유사 구역 + 이력 + 근거(LLM이 언어화할 구조화 입력).

    sort_by_recency: ★유사도와 분리된 명시적 정렬 스위치(최근 사례 우선) — 유사도에 안 섞음.
    """
    th = cfg or load_thresholds()
    qm = cluster_metrics(query_pnus, parcels, buildings, cfg=th)
    matches = cosine_topk(_featurize(qm, zv), zv, k=k)
    if sort_by_recency:
        matches = sorted(matches, key=lambda x: (x["t"] is not None, x["t"]), reverse=True)
    return {"query_metrics": {k_: qm.get(k_) for k_, _ in _FEATURES}, "matches": matches}

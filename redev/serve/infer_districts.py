"""infer_districts.py — 전역 추론 점수 캐시 (Phase 8 [0] → C 서울 전역). 설계: demo.md §0-1.

★학습 동결: production B1+(4구 학습)로 추론 구 노드를 *점수만* 낸다(inductive — 학습 안 한 구
추론). 지적도·건물은 서울 전체분이라 클립만. ★조리 배치(C): 구별로 클립→그래프→전노드 피처→
점수→누적. 그래프 인접은 구간 엣지 0(구 배치)이라 per-gu가 전역과 동치 + 메모리 구 1개로 바운드.
재학습 0. 캐시명은 자치구 수와 무관(infer_scores.parquet) — 규칙7 config 확장.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from redev.paths import DATA

_CACHE = DATA / "processed/infer_scores.parquet"
_KEEP = ["pnu", "sigungu", "score", "score_pct", "cx", "cy", "aging"]


def build_inference_scores(*, force_rebuild: bool = False, log=print) -> pd.DataFrame:
    """전역 추론 전 노드 점수 캐시(구별 조리 배치). ★production B1+ 동결, inductive 스코어.

    반환: [pnu, sigungu, score, score_pct(구내 백분위), cx, cy, aging]. backend /screen·/report가 조회.
    log: 구별 조리 시간 측정 출력(측정 보고용). force_rebuild=False면 캐시 재사용.
    """
    if not force_rebuild and _CACHE.exists():
        return pd.read_parquet(_CACHE)

    from redev.config import inference_sigungu_codes
    from redev.data.ingest.building_gis import load_buildings
    from redev.data.ingest.parcels import load_parcels
    from redev.graph.build import build_graph
    from redev.models.baseline import _SRC, _vsizip, prepare_baseline_matrix
    from redev.models.infer import build_all_node_features, score_all, train_production_b1

    codes = sorted(inference_sigungu_codes())
    t0 = time.time()
    parcels, _ = load_parcels(_vsizip(*_SRC["parcels"]), codes, with_geometry=True)   # 전역 1회 적재
    buildings, _ = load_buildings(_vsizip(*_SRC["buildings"]), with_geometry=False)   # 서울 전체(조인서 클립)
    log(f"[load] {len(codes)}구 parcels {len(parcels):,} + buildings {len(buildings):,} ({time.time()-t0:.0f}s)")

    aug = prepare_baseline_matrix()                                  # 4구 라벨(학습 동결)
    model, fc = train_production_b1(aug)                             # production B1+ −용도지역(1회)
    log(f"[train] production B1+ 동결 ({time.time()-t0:.0f}s 누적)")

    parts = []
    for code in codes:                                              # ★구별 조리(클립→그래프→피처→점수)
        tc = time.time()
        gp = parcels[parcels["sigungu"] == code].copy()
        if gp.empty:
            log(f"  [{code}] parcels 0 — 건너뜀"); continue
        graph, pnu_to_idx, _ = build_graph(gp)                      # 구 인접 그래프(구간 엣지 0)
        allf = build_all_node_features(gp, buildings, pnu_to_idx, graph.edge_index)
        allf["score"] = score_all(model, allf, fc)                  # inductive 스코어
        allf["score_pct"] = allf["score"].rank(pct=True)            # 구내 백분위(표시용)
        centroid = gp.set_index("pnu").geometry.centroid
        allf["cx"] = allf["pnu"].map(centroid.x).values
        allf["cy"] = allf["pnu"].map(centroid.y).values
        parts.append(allf[_KEEP])
        log(f"  [{code}] {len(gp):,}필지 점수 중앙{allf['score'].median():.3f} ({time.time()-tc:.0f}s)")

    out = pd.concat(parts, ignore_index=True)
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(_CACHE, index=False)
    log(f"[done] {len(out):,}필지 / {len(parts)}구 / 총 {time.time()-t0:.0f}s / 캐시 {_CACHE.stat().st_size/1e6:.1f}MB")
    return out

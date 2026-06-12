"""infer_districts.py — 추론 구 확장(마포·강남) 점수 캐시 (Phase 8 [0]). 설계: demo.md §0-1.

★학습 동결: production B1+(4구 학습)로 추론 구(6구) 노드를 *점수만* 낸다(inductive — 학습 안 한
구 추론). 지적도·건물은 서울 전체분이라 클립만. 6구 graph→현재시점 피처→점수→캐시. 재학습 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from redev.paths import DATA

_CACHE = DATA / "processed/infer_scores_6gu.parquet"


def build_inference_scores(*, force_rebuild: bool = False) -> pd.DataFrame:
    """추론 6구(학습 4구 + 마포·강남) 전 노드 점수 캐시. ★production B1+ 동결, inductive 스코어.

    반환: [pnu, sigungu, score, score_pct(구내 백분위), + 피처]. backend /screen·/report가 조회.
    """
    if not force_rebuild and _CACHE.exists():
        return pd.read_parquet(_CACHE)

    from redev.config import inference_sigungu_codes
    from redev.data.ingest.building_gis import load_buildings
    from redev.data.ingest.parcels import load_parcels
    from redev.graph.build import build_graph
    from redev.models.baseline import _SRC, _vsizip, prepare_baseline_matrix
    from redev.models.infer import build_all_node_features, score_all, train_production_b1

    codes = sorted(inference_sigungu_codes())                        # 6구(4 학습 + 마포·강남)
    parcels, _ = load_parcels(_vsizip(*_SRC["parcels"]), codes, with_geometry=True)
    buildings, _ = load_buildings(_vsizip(*_SRC["buildings"]), with_geometry=False)

    graph, pnu_to_idx, _ = build_graph(parcels)                      # 6구 인접 그래프(구 배치, 구간 엣지 0)
    allf = build_all_node_features(parcels, buildings, pnu_to_idx, graph.edge_index)   # 현재시점 10피처+nb

    aug = prepare_baseline_matrix()                                  # 4구 라벨(학습 동결)
    model, fc = train_production_b1(aug)                             # production B1+ −용도지역
    allf["score"] = score_all(model, allf, fc)                       # ★inductive 스코어
    allf["score_pct"] = allf.groupby("sigungu")["score"].rank(pct=True)   # 구내 백분위(표시용)

    centroid = parcels.set_index("pnu").geometry.centroid            # 지도 표시용 좌표(5186)
    allf["cx"] = allf["pnu"].map(centroid.x).values
    allf["cy"] = allf["pnu"].map(centroid.y).values
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    keep = ["pnu", "sigungu", "score", "score_pct", "cx", "cy", "aging"]
    allf[keep].to_parquet(_CACHE, index=False)
    return allf[keep]

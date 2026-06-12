"""infer.py — 전 노드 스코어링 → 확률 히트맵 + 후보 클러스터 (Phase 6, R12). 설계: pipeline.md.

추론은 "현재 시점"(2026) 스냅샷 — 사용자는 오늘의 판단을 묻는다. production B1(전 라벨 학습)으로
전 노드(uncertain 포함) 확률을 내고, 고확률 노드의 그래프 연결요소를 후보 클러스터로 묶는다.
★하이퍼파라미터는 Phase 3 inner 공간CV에서 고른 세트 동결(전 데이터 재튜닝=미검증 배포).
★운영 임계값은 Phase 3 pooled OOF best-F1에서(임의 0.5 금지).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Phase 3 inner 공간CV 선정 결과 동결(재튜닝 금지). run_xgb_cv selection: depth=5, lr=0.1.
PROD_B1_PARAMS = (5, 0.1)


def build_all_node_features(parcels, buildings, pnu_to_idx, edge_index, *, current_year: int = 2026) -> pd.DataFrame:
    """★전 노드(141K) 현재시점 피처(self + 이웃집계) — infer·B0 공유 캐시. 전역 idx 순서 정렬.

    학습행렬은 라벨뿐 → 추론은 전 노드 필요. 단일 t(현재)라 build_neighbor_features가 단순해진다.
    """
    from redev.graph.features import node_features
    from redev.models.baseline import build_neighbor_features

    n = len(pnu_to_idx)
    idx_to_pnu = np.empty(n, dtype=object)
    for p, i in pnu_to_idx.items():
        idx_to_pnu[i] = p
    base = pd.DataFrame({"pnu": idx_to_pnu, "t": current_year})       # 전역 idx 순서
    self_feat = node_features(base, parcels, buildings)               # 현재 self 피처
    aug = build_neighbor_features(self_feat, edge_index, pnu_to_idx, parcels, buildings, hops=2)
    aug["sigungu"] = aug["pnu"].str[:5]
    return aug


def train_production_b1(aug_labeled, *, params=PROD_B1_PARAMS):
    """전 라벨 학습 B1(추론용). ★하파 동결(Phase 3 선정), 공간 inner holdout로 early stopping만.

    검증(LODO)은 Phase 3에서 끝 — 여기선 배포 모델을 전 데이터로 적합(재튜닝 아님).
    """
    from redev.eval.spatial_cv import spatial_zone_groups
    from redev.models.baseline import _make_xgb, _spw, production_feature_set

    fc = production_feature_set(aug_labeled)        # ★B1+ −용도지역(측정: 누수 제외, baseline 주석)
    g = spatial_zone_groups(np.arange(len(aug_labeled)), aug_labeled, k=2)
    va = g[0]
    tr = np.concatenate(g[1:]) if len(g) > 1 else g[0]
    X = aug_labeled[fc].to_numpy(np.float32)
    y = aug_labeled["y"].to_numpy()
    m = _make_xgb(params[0], params[1], spw=_spw(y[tr]))
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    return m, fc


def operating_threshold(aug_labeled, edge_index, pnu_to_idx) -> float:
    """★운영 임계값 = Phase 3 pooled OOF best-F1 (임의값 금지). B1 OOF에서 결정론 산출."""
    from redev.eval.metrics import best_f1
    from redev.models.feasibility import oof_scores
    oof = oof_scores(aug_labeled, edge_index, pnu_to_idx)
    y = aug_labeled["y"].to_numpy()
    m = np.isfinite(oof)
    _, thr = best_f1(y[m], oof[m])
    return float(thr)


def score_all(model, all_feats: pd.DataFrame, fc: list) -> np.ndarray:
    """전 노드 확률(전역 idx 순서)."""
    return model.predict_proba(all_feats[fc].to_numpy(np.float32))[:, 1]


def heatmap_percentile(scores: np.ndarray) -> np.ndarray:
    """★히트맵 표시값 = 백분위(0~100). raw 확률은 81%가 0.97 포화 → 온통 빨강(정보 0).

    백분위는 "상대적으로 어디가 더 노후 환경인가"를 보이고 Phase 5 "상위 X%" 언어와 통일.
    """
    from scipy.stats import rankdata
    return rankdata(scores, method="average") / len(scores) * 100.0


def percentile_threshold(scores: np.ndarray, *, top_pct: float) -> float:
    """상위 top_pct% 점수 컷(★타이트 클러스터용 — 표시용·미검증 컷). 포화 확률의 사용성 보강."""
    return float(np.percentile(scores, 100.0 - top_pct))


def candidate_clusters(scores: np.ndarray, pnu_to_idx: dict, edge_index, *, thr: float, min_nodes: int = 5) -> list:
    """고확률(≥thr) 노드의 그래프 연결요소 → 후보 클러스터(PNU 집합). 최소크기 필터.

    인접한 고확률 노드 = 한 후보(블록). 고립 고확률(이웃 저확률)은 min_nodes로 걸러짐.
    """
    n = len(pnu_to_idx)
    idx_to_pnu = np.empty(n, dtype=object)
    for p, i in pnu_to_idx.items():
        idx_to_pnu[i] = p
    high = scores >= thr
    src, dst = np.asarray(edge_index[0]), np.asarray(edge_index[1])
    keep = high[src] & high[dst]                                      # 양 끝 고확률 엣지만
    g = coo_matrix((np.ones(keep.sum()), (src[keep], dst[keep])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    clusters = []
    for c in np.unique(comp[high]):
        members = np.where((comp == c) & high)[0]
        if len(members) >= min_nodes:
            clusters.append(set(idx_to_pnu[members].tolist()))
    return clusters


def cluster_polygon(cluster_pnus, parcels, *, buffer_m: float = 3.0, min_area_m2: float = 500.0):
    """(보너스, R12) 클러스터 필지 → 매끈 경계. 합집합+버퍼 닫힘+최소면적. 도로 스냅핑은 v1.1.

    노드 분류를 이어붙이면 구멍·삐죽 → buffer 후 unbuffer로 작은 틈 닫고 최소면적 미만 제거.
    """
    sub = parcels[parcels["pnu"].isin(set(cluster_pnus))]
    if sub.empty:
        return None
    merged = sub.geometry.buffer(buffer_m).union_all().buffer(-buffer_m)   # 닫힘 연산
    if merged.is_empty or merged.area < min_area_m2:
        return None
    return merged

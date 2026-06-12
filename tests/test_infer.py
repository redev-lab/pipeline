"""infer/iou/B0 회귀 테스트 — 클러스터·백분위·region-grow·IoU(순수, 합성).

실행: python -m pytest tests/test_infer.py
"""
import numpy as np

from redev.eval.iou import core_capture, set_iou
from redev.models.baseline import region_grow
from redev.models.infer import candidate_clusters, heatmap_percentile, percentile_threshold

_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
_EI = np.array([[0, 1, 2, 3], [1, 0, 3, 2]])    # A-B, C-D


def test_candidate_clusters_connected_high():
    """인접 고확률만 한 클러스터. 고립 고확률은 min_nodes로 탈락."""
    scores = np.array([0.9, 0.9, 0.9, 0.1])     # A,B,C 고 / D 저 → C는 고립(C-D만)
    cl = candidate_clusters(scores, _IDX, _EI, thr=0.5, min_nodes=2)
    assert len(cl) == 1 and cl[0] == {"A", "B"}


def test_heatmap_percentile_monotone():
    p = heatmap_percentile(np.array([0.1, 0.5, 0.9, 0.97]))
    assert p[0] < p[1] < p[2] < p[3] and p[-1] <= 100.0


def test_percentile_threshold_top():
    s = np.arange(100, dtype=float)             # 0..99
    assert percentile_threshold(s, top_pct=10) == np.percentile(s, 90)


def test_region_grow_seed_and_grow():
    """노후 seed(≥0.6) 품은 grow(≥0.4) 연결요소만."""
    aging = np.array([0.7, 0.5, 0.3, 0.1])      # A seed, B grow, C·D 미달
    cl = region_grow(aging, _EI, _IDX, seed_cut=0.6, grow_cut=0.4, min_nodes=2)
    assert len(cl) == 1 and cl[0] == {"A", "B"}


def test_iou_and_core_capture():
    assert set_iou({"A", "B"}, {"B", "C"}) == 1 / 3      # ∩1 ∪3
    assert core_capture({"A", "B"}, {"B", "C"}) == 0.5   # B를 {B,C}의 절반 포착

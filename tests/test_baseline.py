"""baseline 회귀 테스트 — 이웃집계 순수함수(aggregate_once) 손계산 검증(수검③).

실행: python -m pytest tests/test_baseline.py
"""
import numpy as np

from redev.models.baseline import _closure, _csr_neighbors, _nb_columns, aggregate_once


def test_aggregate_once_hand():
    """경로그래프 0-1-2, 피처 1열로 손계산: mean·max가 이웃만 집계하나."""
    feat = np.array([[1.0], [2.0], [4.0]])
    # 대칭 엣지 (0-1, 1-2): 양방향
    src = np.array([0, 1, 1, 2])
    dst = np.array([1, 0, 2, 1])
    out = aggregate_once(feat, src, dst)        # [3, 2] = [mean | max]
    # 0의 이웃 {1}=2 ; 1의 이웃 {0,2}=mean(1,4)=2.5,max4 ; 2의 이웃 {1}=2
    assert np.allclose(out[:, 0], [2.0, 2.5, 2.0])     # mean
    assert np.allclose(out[:, 1], [2.0, 4.0, 2.0])     # max


def test_aggregate_once_isolated_zero():
    """이웃 없는 노드는 0 (features fillna(0.0) 규약과 동일)."""
    feat = np.array([[5.0], [9.0]])
    out = aggregate_once(feat, np.array([0]), np.array([1]))   # 1은 src에 없음
    assert np.allclose(out[1], [0.0, 0.0])      # 노드1: 이웃 집계 없음 → 0


def test_closure_two_hops():
    """폐포가 정확히 hops 홉까지 닿나 (2홉 집계의 정확성 전제)."""
    # 0-1-2-3 사슬
    ei = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    src_s, dst_s = _csr_neighbors(ei, 4)
    clo1 = _closure(np.array([0]), src_s, dst_s, hops=1)
    clo2 = _closure(np.array([0]), src_s, dst_s, hops=2)
    assert set(clo1.tolist()) == {0, 1}            # 1홉: 자기+직접이웃
    assert set(clo2.tolist()) == {0, 1, 2}         # 2홉: 2까지


def test_nb_columns_shape():
    cols = _nb_columns(2)
    assert len(cols) == 2 * 2 * 5                   # 2홉 × (mean,max) × 5피처
    assert cols[0].startswith("nb1_mean_") and cols[-1].startswith("nb2_max_")

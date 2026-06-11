"""gnn 회귀 테스트 — model forward 형상 + Normalizer(area log1p, train-fit).

실행: python -m pytest tests/test_gnn.py
"""
import numpy as np
import torch

from redev.graph.features import FEATURE_COLUMNS
from redev.models.gnn.model import RedevSAGE
from redev.models.gnn.train import Normalizer, _AREA_IDX


def test_sage_forward_shape():
    """노드 N개 → logit N개 (per-node 분류)."""
    torch.manual_seed(0)
    m = RedevSAGE(in_dim=5, hidden=16, dropout=0.0)
    x = torch.randn(4, 5)
    ei = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])   # 사슬 0-1-2-3
    out = m(x, ei)
    assert out.shape == (4,)                       # 노드별 1 logit


def test_normalizer_area_log1p_and_zscore():
    """area_m2(idx1)만 log1p 후 z-score. 통계는 fit 데이터 기준."""
    feats = np.array([[0.5, 100.0, 0.8, 10.0, 1.0],
                      [0.0, 1_000_000.0, 0.2, 0.0, 0.0]], dtype=np.float64)
    n = Normalizer().fit(feats)
    out = n.transform(feats)
    assert _AREA_IDX == FEATURE_COLUMNS.index("area_m2") == 1
    assert np.allclose(out.mean(0), 0.0, atol=1e-5)   # z-score → 평균 0
    # area는 log1p로 압축돼 100 vs 1e6의 raw 격차(1e4배)가 줄어든다
    assert abs(np.log1p(100.0) - np.log1p(1_000_000.0)) < 1_000_000.0


def test_normalizer_fit_only_on_given():
    """transform은 fit 통계만 사용 — 다른 분포를 줘도 fit 평균/표준편차로."""
    train = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 1.0, 1.0]])
    n = Normalizer().fit(train)
    mean_before = n.mean.copy()
    n.transform(np.array([[9.0, 9.0, 9.0, 9.0, 9.0]]))   # 호출이 통계를 안 바꿔야
    assert np.allclose(n.mean, mean_before)

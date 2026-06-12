"""pu 회귀 테스트 — 가중치(함정 §0: uncertain만 저가중)·식별 컷(순수).

실행: python -m pytest tests/test_pu.py
"""
import numpy as np
import pandas as pd

from redev.models.pu import UNDESIGNATED_CUT, pu_weights


def test_pu_weights_only_uncertain_low():
    """★positive·reliable_neg=1.0, uncertain만 w(저가중) — 함정 §0 대비."""
    cert = pd.Series(["positive", "reliable_negative", "uncertain", "uncertain"])
    w = pu_weights(cert, w=0.3)
    assert np.allclose(w, [1.0, 1.0, 0.3, 0.3])


def test_undesignated_cut_matches_labels():
    """uncertain 식별 컷 = labels §4 노후미지정(0.5)."""
    assert UNDESIGNATED_CUT == 0.5

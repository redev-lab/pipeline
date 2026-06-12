"""feasibility 회귀 테스트 — 보정·랭킹·명칭 정직성(순수, 합성).

실행: python -m pytest tests/test_feasibility.py
"""
import numpy as np

from redev.models.feasibility import (ENV_SCORE_LABEL, calibrate, reliability,
                                       risk_signal_placeholder, score_feasibility, topk_precision)


def test_calibrate_monotone():
    """isotonic 보정은 단조 비감소(순서 보존)."""
    oof = np.array([0.1, 0.4, 0.6, 0.9])
    y = np.array([0, 0, 1, 1])
    ir = calibrate(oof, y)
    p = ir.predict([0.1, 0.5, 0.9])
    assert p[0] <= p[1] <= p[2]


def test_reliability_ece():
    # 완벽 보정(예측=실제율): conf 0=acc 0, conf 1=acc 1 → ECE 0
    assert reliability(np.array([0.0, 0.0, 1.0, 1.0]), np.array([0, 0, 1, 1]))["ece"] == 0.0
    # 과신: 0.1 예측인데 실제 0 → ECE 0.1
    assert abs(reliability(np.array([0.1, 0.1]), np.array([0, 0]))["ece"] - 0.1) < 1e-9


def test_topk_precision_concentrates():
    """상위 50%가 양성을 앞에 모으나."""
    tk = topk_precision(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0]))
    assert tk["top50%"] == 1.0


def test_score_feasibility_naming_honesty():
    """★점수 정체='재개발 환경 유사도' → 이름에 '추진 가능성' 금지(R15·R18)."""
    ref = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    out = score_feasibility(0.8, ref)
    assert out["label"] == ENV_SCORE_LABEL and "추진" not in out["label"]
    assert out["rank_top_pct"] == 20.0                       # 0.8은 5개 중 상위 20%
    assert out["risk_signals"]["status"] == "v1_미학습"      # R18 미학습 명시


def test_risk_placeholder_no_assertion():
    rs = risk_signal_placeholder()
    assert rs["status"] == "v1_미학습" and "R18" in rs["reason"]

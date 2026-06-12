"""feasibility 회귀 테스트 — 보정·랭킹·명칭 정직성(순수, 합성).

실행: python -m pytest tests/test_feasibility.py
"""
import numpy as np

from redev.models.feasibility import (ENV_SCORE_LABEL, _rank_phrase, calibrate, reliability,
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


def test_rank_phrase_top_and_bottom():            # §B-1 표시 표기
    assert _rank_phrase(8.0) == "상위 8.0%"
    assert _rank_phrase(50) == "상위 50%"          # 경계 = 상위
    assert _rank_phrase(77.3) == "하위 22.7%"      # 50 초과 → 하위
    assert _rank_phrase(100.0) == "하위 0.0%"      # 최하위


def test_raw_rank_splits_saturated_scores():      # §B-2 raw 순위는 동률 뭉침 없음
    # 보정확률이면 상단 포화로 0.993·0.994가 같은 백분위로 뭉쳤다. raw 순위는 갈라져야 한다.
    ref = np.array([0.1, 0.5, 0.9, 0.993, 0.994, 0.999])
    a = score_feasibility(0.993, ref, calibrated_prob=0.977)
    b = score_feasibility(0.994, ref, calibrated_prob=0.977)
    assert a["rank_top_pct"] != b["rank_top_pct"]              # 동률 뭉침 해소
    assert a["rank_top_pct"] > b["rank_top_pct"]               # 0.994가 더 상위(작은 값)
    assert a["calibrated_prob"] == 0.977 and "calibrated_prob" not in a["rank_phrase"]  # 보정확률은 메타

"""retrieval 회귀 테스트 — 코사인·정규화·t제외(순수, 합성).

실행: python -m pytest tests/test_retrieval.py
"""
import numpy as np

from redev.retrieval.case_search import _FEATURES, ZoneVectors, _normalize, cosine_topk


def test_normalize_unit_norm():
    v = _normalize(np.array([[3.0, 4.0, 0.0, 0.0]]))
    assert abs(np.linalg.norm(v[0]) - 1.0) < 1e-9


def test_features_exclude_t():
    """★t(지정연도)는 유사도 축에서 제외 — 물리 4피처만."""
    keys = [k for k, _ in _FEATURES]
    names = [n for _, n in _FEATURES]
    assert names == ["노후도", "면적", "호수밀도", "접도율"]
    assert "t" not in keys and "designated_year" not in keys


def test_cosine_topk_nearest_and_axes():
    Z = _normalize(np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [1.0, 1.0, 0, 0]]))
    meta = [{"zone_id": "A", "t": 2008, "zone_type": "UQ1221", "completed": False},
            {"zone_id": "B", "t": 2010, "zone_type": "UQ1221", "completed": False},
            {"zone_id": "C", "t": 2009, "zone_type": "UQ1221", "completed": True}]
    zv = ZoneVectors(Z, meta, np.zeros(4), np.ones(4))
    q = _normalize(np.array([[1.0, 0, 0, 0]]))[0]
    top = cosine_topk(q, zv, k=2)
    assert top[0]["zone_id"] == "A" and top[0]["similarity"] == 1.0   # 노후도축 정렬
    assert "노후도" in top[0]["top_similar_axes"]
    assert top[0]["completed"] is False and top[0]["t"] == 2008       # 메타(이력) 동봉

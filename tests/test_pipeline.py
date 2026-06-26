"""pipeline 회귀 테스트 — 주소 파싱(지번/도로명/PNU직접) + 단계 래퍼(순수).

실행: python -m pytest tests/test_pipeline.py
"""
import numpy as np
import pytest

from redev.orchestration.pipeline import Context, _confidence, _stage, address_to_pnu


def test_confidence_not_inverted_by_score():
    """★신뢰도 = 임계값 거리(점수 높낮이 아님). 최하위/최상위=고신뢰, 경계 근처=저신뢰."""
    thr, margin = 0.618, 0.15
    assert _confidence(0.058, thr, margin) == "고신뢰"   # 최하위 = 확실히 아님(역전 버그 수정)
    assert _confidence(0.99, thr, margin) == "고신뢰"    # 최상위 = 확실히 후보환경
    assert _confidence(0.65, thr, margin) == "저신뢰"    # 임계값 바로 위 = 애매
    assert _confidence(0.55, thr, margin) == "저신뢰"    # 임계값 바로 아래 = 애매


def _ctx():
    # 최소 Context (주소 파싱만 — 나머지 필드는 더미)
    return Context(
        parcels=None, buildings=None, pnu_to_idx={}, edge_index=None,
        jibun_index={("11290", "정릉동", 170, 1): "1129013300101700001"},
        scores=np.array([]), calibrated=np.array([]), pnu_cluster={}, thr=0.5,
        target=None, agg_level=None, comp=None,
        name2code={"성북구": "11290", "동작구": "11590", "은평구": "11380", "구로구": "11530"},
    )


def test_jibun_address_to_pnu():
    assert address_to_pnu("성북구 정릉동 170-1", _ctx()) == "1129013300101700001"
    assert address_to_pnu("서울특별시 성북구 정릉동 170-1", _ctx()) == "1129013300101700001"


def test_pnu_direct_input():
    assert address_to_pnu("1129013300101700001", _ctx()) == "1129013300101700001"


def test_road_address_friendly_error():
    """★도로명주소 미지원 → 친절한 에러(v1 한계)."""
    with pytest.raises(ValueError, match="도로명주소 미지원"):
        address_to_pnu("성북구 보문로 100", _ctx())


def test_out_of_scope_gu():
    with pytest.raises(ValueError, match="구 미인식"):       # 4구 _ctx name2code 밖(문구: 서울 25구 밖이거나 구 미인식)
        address_to_pnu("강남구 역삼동 100", _ctx())


def test_scoped_full_and_partial():                          # ★부분 전역화 — 7구 full vs 전역 사이드카 partial
    from redev.orchestration.pipeline import address_to_pnu_scoped
    from redev.serve.global_index import _key_hash
    ctx = _ctx()
    assert address_to_pnu_scoped("성북구 정릉동 170-1", ctx, None) == ("1129013300101700001", "full")
    kh = _key_hash("11290", "장위동", 68, 422)               # jibun_index엔 없고 gidx엔 있음 → partial
    gidx = {"kh": np.array([kh], dtype=np.int64), "pn": np.array([1129010600100680422], dtype=np.int64),
            "zones": set(), "clusters": set()}
    assert address_to_pnu_scoped("성북구 장위동 68-422", ctx, gidx) == ("1129010600100680422", "partial")


def test_stage_wrapper_catches():
    """단계 래퍼는 예외를 status='error'로 흡수(부분 실패 견고)."""
    ok = _stage(lambda x: x + 1, 1)
    bad = _stage(lambda: 1 / 0)
    assert ok == {"status": "ok", "result": 2}
    assert bad["status"] == "error" and "ZeroDivision" in bad["reason"]

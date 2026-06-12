"""pipeline 회귀 테스트 — 주소 파싱(지번/도로명/PNU직접) + 단계 래퍼(순수).

실행: python -m pytest tests/test_pipeline.py
"""
import numpy as np
import pytest

from redev.orchestration.pipeline import Context, _stage, address_to_pnu


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
    with pytest.raises(ValueError, match="4구"):
        address_to_pnu("강남구 역삼동 100", _ctx())


def test_stage_wrapper_catches():
    """단계 래퍼는 예외를 status='error'로 흡수(부분 실패 견고)."""
    ok = _stage(lambda x: x + 1, 1)
    bad = _stage(lambda: 1 / 0)
    assert ok == {"status": "ok", "result": 2}
    assert bad["status"] == "error" and "ZeroDivision" in bad["reason"]

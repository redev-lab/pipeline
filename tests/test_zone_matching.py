"""zone_matching 회귀 테스트 — t-전쟁에서 잡은 함정 6개를 영구히 지킨다.

각 test 함수 = 잡은 버그 1개. 한 번 잡은 함정은 테스트가 다시 안 빠지게 한다.
실행: python -m pytest tests/test_zone_matching.py
"""
from redev.data.zone_matching import (
    clean_text,
    cycle_done,
    is_redev_title,
    normalize_zone_name,
    parent_of,
    region_of,
)


def test_bug1_region_of_dong_strip():
    """함정1(동맵 가드의 토큰 측): 지번형 토큰의 지역이 동맵 키와 맞아야."""
    assert region_of("응암동700") == "응암"   # trailing 동 strip → 동맵 키 '응암'과 일치
    assert region_of("돈암6") == "돈암"
    assert region_of("가2") == "가"           # 2자 미만으로 줄면 보존(strip 안 함)


def test_bug2_je_not_swallowed():
    """함정2: 그리디 한글이 '제'를 삼키면 안 된다(돈암제6→돈암6, 돈암제6 아님)."""
    assert normalize_zone_name("돈암제6 주택재개발정비구역 결정") == {"돈암6"}
    assert "돈암제6" not in normalize_zone_name("돈암제6 주택재개발")


def test_normalize_paren_multi_zone():
    """괄호 안 다중 구역 '(흑석2,9구역)' → {흑석2, 흑석9}."""
    toks = normalize_zone_name("흑석재정비촉진지구 재정비촉진계획(흑석2,9구역) 변경결정")
    assert {"흑석2", "흑석9"} <= toks


def test_bug3_reconstruction_excluded():
    """함정3: 재건축 결정이 '정비구역' 키워드로 새면 안 된다."""
    assert is_redev_title("정릉3 주택재건축 정비구역지정 및 지형도면 고시") is False
    assert is_redev_title("돈암제6 주택재개발정비구역 결정 및 지형도면 고시") is True


def test_bug4_encoding_connector():
    """함정4: cp949 깨짐(?)·middle-dot(·)을 일반 규칙으로 제거 → 동일 구역명."""
    assert clean_text("수색?증산") == "수색증산"
    assert clean_text("수색·증산") == "수색증산"
    # 정규화가 토큰 추출까지 이어지는지(부모 매칭 일관성)
    assert parent_of("수색?증산재정비촉진지구 재정비촉진계획(수색13구역) 변경") == "수색증산"


def test_bug5_later_stage_excluded():
    """함정5: 시행인가 등 후속단계는 지정이 아니다(길음1=1998 시행인가 누수)."""
    assert is_redev_title("길음제1구역주택재개발사업시행인가") is False
    assert is_redev_title("흑석제2구역(3차지역)주택개량재개발사업관리처분계획인가") is False


def test_bug6_cycle_done_detection():
    """함정6: 사이클 완료고시(관리처분/준공) 식별 — 멀티사이클 가드의 입력."""
    assert cycle_done("흑석제2구역(3차지역)주택개량재개발사업관리처분계획인가") is True
    assert cycle_done("돈암제6 주택재개발정비구역 결정 및 지형도면 고시") is False


def test_parent_of_promotion():
    """촉진 sub→부모 매핑(근사 t용). 비촉진은 None."""
    assert parent_of("흑석재정비촉진지구 재정비촉진계획(흑석2구역) 변경결정") == "흑석"
    assert parent_of("돈암제6 주택재개발정비구역 결정") is None

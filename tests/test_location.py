"""location 파서 회귀 테스트 — cancelled.py 해제 위치 파싱 함정 고정.

parse_location(공백없음·일대·산·행정동) + admin_to_legal_dong(법정동 보존).
실행: python -m pytest tests/test_location.py
"""
from redev.data.location import admin_to_legal_dong, parse_location


def test_parse_basic_and_nospace():
    assert parse_location("수유동 711") == ("수유동", 711, 0, False)
    assert parse_location("정릉동170-1") == ("정릉동", 170, 1, False)   # 공백없음


def test_parse_ildae_suffix():
    assert parse_location("장위동 233-42일대") == ("장위동", 233, 42, False)
    assert parse_location("증산동 205-33 일대") == ("증산동", 205, 33, False)


def test_parse_san():
    assert parse_location("상도동 산65") == ("상도동", 65, 0, True)


def test_parse_legal_dong_with_digit_preserved():
    """법정동의 동 *뒤* 숫자/가(안암동2가)는 동명의 일부 — 보존."""
    assert parse_location("안암동2가59") == ("안암동2가", 59, 0, False)
    assert parse_location("삼선동1가 512-34") == ("삼선동1가", 512, 34, False)


def test_admin_to_legal_strips_only_pre_dong_digit():
    """행정동(동 *앞* 숫자)만 strip, 법정동(동 뒤 가/숫자)은 보존."""
    assert admin_to_legal_dong("상도2동") == "상도동"      # 행정동→법정동
    assert admin_to_legal_dong("정릉3동") == "정릉동"
    assert admin_to_legal_dong("안암동2가") == "안암동2가"  # 법정동 보존(깨먹지 않음)
    assert admin_to_legal_dong("삼선동1가") == "삼선동1가"


def test_parse_reject_non_jibun():
    assert parse_location("") is None
    assert parse_location("구역명만있음") is None

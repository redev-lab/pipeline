"""transactions 회귀 테스트 — 파싱·PNU매핑 순수로직(네트워크 없음, 합성 XML).

실행: python -m pytest tests/test_transactions.py
"""
from redev.data.ingest.transactions import _num, _parse_items, _to_pnu

_XML = """<response><body><items>
  <item><umdNm>석관동</umdNm><jibun>338-505</jibun><sggCd>11290</sggCd>
    <dealAmount>32,500</dealAmount><excluUseAr>41.85</excluUseAr><landAr>30.71</landAr>
    <buildYear>2018</buildYear><dealYear>2024</dealYear><dealMonth>5</dealMonth><cdealType></cdealType></item>
  <item><umdNm>석관동</umdNm><jibun>10</jibun><sggCd>11290</sggCd>
    <dealAmount>50,000</dealAmount><cdealType>O</cdealType></item>
</items></body></response>"""


def test_num_parses_comma_and_blank():
    assert _num("32,500") == 32500.0
    assert _num("") is None and _num("-") is None
    assert _num("30.71") == 30.71


def test_parse_items_excludes_cancelled():
    """cdealType='O'(계약해제) 제외 — 2건 중 1건만."""
    items = _parse_items(_XML, "villa")
    assert len(items) == 1
    assert items[0]["land"] == "30.71" and items[0]["amount"] == "32,500"


def test_to_pnu_direct_and_admin_fallback():
    idx = {("11290", "석관동", 338, 505): "P1", ("11290", "상도동", 159, 1): "P2"}
    assert _to_pnu("석관동", "338-505", "11290", idx) == "P1"           # 직접 조회
    assert _to_pnu("상도2동", "159-1", "11290", idx) == "P2"            # 행정동→법정동 폴백
    assert _to_pnu("없는동", "1-1", "11290", idx) is None              # 미존재

"""zone_attrs 매칭 회귀 — 구역명 정규화 + 컨텍스트 zone_id 강건 매칭 (순수).

실행: python -m pytest tests/test_zone_attrs.py
"""
from redev.data.zone_attrs import norm_zone_name, resolve_to_context


def test_norm_zone_name_absorbs_variants():
    assert norm_zone_name("가리봉제1구역") == "가리봉1"
    assert norm_zone_name("가리봉1구역") == "가리봉1"
    assert norm_zone_name("가리봉 제1구역") == "가리봉1"
    assert norm_zone_name("장위15구역 주택재개발정비사업") == "장위15"
    assert norm_zone_name("흑석2재정비촉진구역") == "흑석2"


def test_resolve_direct_match_high_conf():
    store = {"11290NTC0001": {"zone_name": "장위15구역", "attrs": {}}}
    res = resolve_to_context(store, ["11290NTC0001"], {"11290NTC0001": "장위15구역 변경"})
    assert res["resolved"]["11290NTC0001"]["match"] == {"confidence": "고", "method": "고시관리코드 일치", "gosi_zone_id": "11290NTC0001"}
    assert res["unmatched"] == []


def test_resolve_name_match_mid_conf():            # 고시관리코드 다르지만 구역명으로 연결
    store = {"11530NTC_GOSI": {"zone_name": "가리봉1구역", "attrs": {}}}
    ctx_ids = ["11530NTC_CTX"]                       # 같은 자치구(11530), 다른 NTFC_SN
    titles = {"11530NTC_CTX": "가리봉제1구역 주택정비형 재개발사업 정비계획 결정"}
    res = resolve_to_context(store, ctx_ids, titles)
    m = res["resolved"]["11530NTC_CTX"]["match"]
    assert m["confidence"] == "중" and m["method"] == "구역명 정규화" and m["gosi_zone_id"] == "11530NTC_GOSI"


def test_resolve_unmatched_not_forced():           # ★오매칭보다 무매칭 — 후보 없으면 연결 안 함
    store = {"99999NTC_X": {"zone_name": "없는구역", "attrs": {}}}
    res = resolve_to_context(store, ["11290NTC_A"], {"11290NTC_A": "장위15구역 변경"})
    assert res["resolved"] == {} and res["unmatched"][0]["reason"] == "무매칭"


def test_resolve_gu_guard_blocks_cross_district():  # 자치구 다르면 구역명 같아도 매칭 안 함
    store = {"11530NTC_GOSI": {"zone_name": "중앙1구역", "attrs": {}}}
    res = resolve_to_context(store, ["11680NTC_OTHER"], {"11680NTC_OTHER": "중앙1구역 정비"})
    assert res["resolved"] == {} and res["unmatched"][0]["reason"] == "무매칭"

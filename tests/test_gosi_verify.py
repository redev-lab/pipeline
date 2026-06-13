"""gosi_verify 회귀 — verbatim diff·범위 가드·출처·미기재 (순수, LLM/PDF 없음).

실행: python -m pytest tests/test_gosi_verify.py
"""
from redev.nlp.gosi_verify import _value_token, verify_attr, verify_extraction

_SRC = "정비계획 결정 내용. 기준용적률 : 190.0% 이며, 계획세대수 : 2,464세대 로 한다. 건폐율 60% 이하."


def test_value_token_parsing():
    assert _value_token("190.0%") == "190.0"
    assert _value_token("2,464세대") == "2464"
    assert _value_token("13,000.0㎡") == "13000.0"
    assert _value_token("미기재") is None


def test_verified_when_in_source_and_range():
    item = {"raw": "190.0%", "label": "기준용적률", "변경구분": "변경후",
            "sentence": "기준용적률 : 190.0%"}
    v = verify_attr("용적률", item, _SRC, 고시번호="2024-448", 고시일자="2024-09-19")
    assert v["grade"] == "verified" and v["value"] == 190.0
    assert v["고시번호"] == "2024-448" and v["고시일자"] == "2024-09-19"   # 출처 보존


def test_rejected_when_number_not_in_source():   # ★환각 차단
    item = {"raw": "250%", "label": "용적률", "변경구분": "변경후", "sentence": "용적률 250%"}
    v = verify_attr("용적률", item, _SRC, 고시번호="x", 고시일자="y")
    assert v["grade"] == "rejected" and v["checks"]["in_source"] is False


def test_flagged_when_out_of_range():
    # 900%는 원문에 넣되 용적률 상식범위(50~600) 밖
    src = _SRC + " 허용용적률 900%"
    item = {"raw": "900%", "label": "허용용적률", "변경구분": "변경후", "sentence": "허용용적률 900%"}
    v = verify_attr("용적률", item, src, 고시번호="x", 고시일자="y")
    assert v["grade"] == "flagged" and v["checks"]["in_range"] is False


def test_flagged_when_sentence_not_in_source():   # 숫자는 있으나 출처 문장이 가짜
    item = {"raw": "190.0%", "label": "용적률", "변경구분": "변경후",
            "sentence": "이 문장은 원문에 없다 190.0%"}
    v = verify_attr("용적률", item, _SRC, 고시번호="x", 고시일자="y")
    assert v["grade"] == "flagged" and v["checks"]["sentence_in_source"] is False


def test_missing_passthrough():
    v = verify_attr("건폐율", "미기재", _SRC, 고시번호="x", 고시일자="y")
    assert v["grade"] == "missing" and v["value"] is None


def test_verify_extraction_summary():
    ex = {"zone_name": "장위15구역", "고시번호": "2024-448", "고시일자": "2024-09-19",
          "attrs": {"용적률": {"raw": "190.0%", "label": "기준용적률", "변경구분": "변경후",
                              "sentence": "기준용적률 : 190.0%"},
                    "계획세대수": {"raw": "2,464세대", "label": "계획세대수", "변경구분": "변경후",
                               "sentence": "계획세대수 : 2,464세대"},
                    "건폐율": "미기재", "구역면적": "미기재"}}
    out = verify_extraction(ex, _SRC)
    assert out["summary"]["verified"] == 2 and out["summary"]["missing"] == 2
    assert out["results"]["계획세대수"]["value"] == 2464.0

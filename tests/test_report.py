"""report 회귀 테스트 — 표시포맷·환각검증·템플릿폴백(순수/mock, 실 LLM 없음).

실행: python -m pytest tests/test_report.py
"""
from redev.llm.report import (
    _display_facts, _template_report, _user_caveats, generate_report, verify_numbers,
)

_DATA = {
    "candidate": True, "b1_score": 0.94, "confidence": "고신뢰",
    "stages": {
        "예언_환경점수": {"status": "ok", "result": {"label": "재개발 환경 점수", "rank_top_pct": 77.3, "rank_phrase": "하위 22.7%", "caveats": ["환경점수 caveat"]}},
        "진단_요건": {"status": "ok", "result": {"path": "재개발", "metrics": {"old_area_ratio": 0.94, "abut_ratio": 0.5}}},
        "진단_시세맥락": {"status": "ok", "result": {"land_share_pyung_man": 1658.0, "newbuild_exclu_pyung_man": 5446.0, "caveats": ["시세 caveat"]}},
    },
    "caveats": ["투자 권유 아님(R15)"],
}


def test_display_facts_preformats_strings():
    f = _display_facts(_DATA)
    assert f["환경점수"] == "재개발 환경 점수 하위 22.7%"     # §B-1: 77.3%(상위) → 하위 22.7%
    assert f["노후도"] == "노후·불량 연면적 94%"
    assert "1,658만원" in f["시세맥락"] and "5,446만원" in f["시세맥락"]


def test_verify_numbers_catches_hallucination():
    f = _display_facts(_DATA)
    assert verify_numbers("환경 점수 하위 22.7%, 노후도 94%.", f)["ok"]         # 표시값 그대로 → 합격
    bad = verify_numbers("수익률 200% 보장, 9억 상승.", f)                       # 200·9 창작
    assert not bad["ok"] and "200" in bad["unmatched"]
    assert verify_numbers("### 1. 될까\n하위 22.7%\n### 2. 얼마", f)["ok"]       # ★절 번호 1.2. 면제


def test_template_report_has_facts_and_caveats():
    f = _display_facts(_DATA)
    cav = ["투자 권유 아님(R15)", "시세 caveat"]
    rep = _template_report(f, cav)
    assert "재개발 환경 점수 하위 22.7%" in rep and "투자 권유 아님(R15)" in rep
    assert verify_numbers(rep, f, cav)["ok"]                                   # 템플릿은 환각 0 보장(caveat 포함)


def test_generate_report_llm_and_fallback():
    # mock LLM: 표시값 그대로 → 환각 0, source llm
    ok = generate_report(_DATA, complete_fn=lambda s, u: "환경 점수 하위 22.7% [환경점수]")
    assert ok["source"] == "llm" and ok["hallucination"]["ok"]
    # LLM 실패 → 템플릿 폴백
    def boom(s, u):
        raise RuntimeError("fail")
    fb = generate_report(_DATA, complete_fn=boom)
    assert fb["source"] == "template" and fb["hallucination"]["ok"]


# ── 계약 v1.1 (6대 결함) 회귀 ──────────────────────────────────────────────

# candidate=False인데 점수 높은 비후보 케이스(예언_환경점수는 항상 산출)
_NONCAND = {
    "candidate": False, "b1_score": 0.92, "confidence": "고신뢰",
    "verdict": {"class": "관심 권역(후보 경계 밖)",
                "headline": "환경 점수 상위 8.0%이나 후보 군집 미포함 — 현 시점 관망 권역. 단정 아님."},
    "stages": {
        "예언_환경점수": {"status": "ok", "result": {"label": "재개발 환경 점수", "rank_top_pct": 8.0, "rank_phrase": "상위 8.0%", "caveats": []}},
        "진단_요건": {"status": "na", "reason": "후보 군집 미형성 — 단일 필지로 요건 판정 불가"},
        "진입_eligibility": {"status": "ok", "result": {
            "진단_토허": {"toheo_applies": True, "gap_investment_possible": False},
            "예언_잔여기간": {"known": False, "stage": None, "note": "해당 구역 아님 — 사업 단계 없음"}}},
    },
    "caveats": ["모든 수치 추정·참고치이며 투자 권유 아님(R15).",
                "v1 후보경계는 거친 필터(코어 ~39% 포착) — 정밀 경계 아님(R13)."],
}


def test_facts_always_filled_no_dash():           # 결함 2
    f = _display_facts(_NONCAND)
    assert "산출 불가" in f["요건판정"]            # "—" 아님, 사유로
    assert f["환경점수"].startswith("재개발 환경 점수 상위")
    assert f.get("결론")                            # 결함 6: 결론 머리문장


def test_judgment_label_uses_confidence_not_hardcoded():  # 신뢰도 역전 수정
    f = _display_facts(_NONCAND)                    # confidence=고신뢰(점수 명확)
    assert f["판정"] == "후보 클러스터 아님(고신뢰)"   # 하드코딩 '(저신뢰)' 제거
    fc = _display_facts({**_NONCAND, "confidence": "저신뢰"})
    assert fc["판정"] == "후보 클러스터 아님(저신뢰)"


def test_report_returns_translated_caveats_user():  # 패널용 번역 caveat 노출
    out = generate_report(_NONCAND, complete_fn=lambda s, u: "ok")
    assert out["caveats_user"] and all("R1" not in c and "§" not in c for c in out["caveats_user"])
    assert any("투자 권유가 아니라" in c for c in out["caveats_user"])


def test_noncandidate_contradiction_explained():  # 결함 1
    f = _display_facts(_NONCAND)                   # rank 8% → 점수 높은데 후보 아님 → 설명 붙음
    assert "후보판정설명" in f and "미포함" in f["후보판정설명"]


def test_low_score_noncandidate_no_contradiction_note():  # 결함 1 경계 — 모순 없을 땐 미부착
    low = {**_NONCAND, "stages": {**_NONCAND["stages"],
           "예언_환경점수": {"status": "ok", "result": {"label": "재개발 환경 점수", "rank_top_pct": 100.0, "rank_phrase": "하위 0.0%"}}}}
    f = _display_facts(low)
    assert "후보판정설명" not in f                  # 점수 낮음 → '대상 아님'이 설명, 모순 문장 없음
    assert f["환경점수"] == "재개발 환경 점수 하위 0.0%"      # §B-1: 상위 100% → 하위 0.0%(결함2 채움 유지)


def test_stage_not_leaked_for_noncandidate():     # 결함 3
    f = _display_facts(_NONCAND)
    assert "잔여기간" not in f                      # 기간·단계 누수 없음
    assert "단계상태" in f and "구역 아님" in f["단계상태"]


def test_plan_info_verified_flagged_and_latest_flag():  # §5 계획정보 표시 분기
    d = {**_DATA, "stages": {**_DATA["stages"], "진단_계획정보": {"status": "ok", "result": {
        "zone_name": "흑석2구역", "고시번호": "2025-426", "고시일자": "2025-07-31",
        "flags": ["최신 미반영(서울시 2025-659 변경안 협의중·미입수)"],
        "attrs": {"용적률": {"value": 599.96, "raw": "599.96", "label": "용적률", "grade": "verified"},
                  "계획세대수": {"value": 1012, "raw": "1,012세대", "label": "건립예정세대수", "grade": "verified"},
                  "건폐율": {"value": 52.26, "raw": "52.26", "label": "건폐율", "grade": "flagged"}}}}}}
    f = _display_facts(d)
    assert "용적률 599.96" in f["계획정보"] and "계획세대수 1,012세대" in f["계획정보"]
    assert "건폐율 52.26(잠정)" in f["계획정보"]              # ★flagged → 잠정(단정 금지)
    assert "서울고시 2025-426 기준, 후속 변경 미반영" in f["계획정보"]   # ★출처 + 최신 플래그
    # 환각검증: 계획정보 숫자가 표시값에 있어 리포트 인용 시 통과
    assert verify_numbers("용적률 599.96, 1,012세대 (서울고시 2025-426 기준)", f)["ok"]


def test_similar_case_uses_display_name_not_raw_code():  # §B-3
    d = {**_DATA, "retrieval": {"matches": [
        {"zone_id": "11590NTC202409250002", "display_name": "동작구 노량진동 일대 (2009)",
         "similarity": 0.91, "t": 2009}]}}
    f = _display_facts(d)
    assert "동작구 노량진동 일대 (2009)" in f["유사사례"]
    assert "11590NTC" not in f["유사사례"]          # 원시코드 노출 안 함


def test_user_caveats_strip_internal_codes():     # 결함 4
    uc = _user_caveats(_NONCAND["caveats"])
    blob = " ".join(uc)
    assert uc and all(code not in blob for code in ("R1", "R13", "R15", "§", "★", "39%"))
    assert "투자 권유가 아니라" in blob


def test_d2_preservation_caveat_translated():     # D-2 유형 밖(보존·상업) 경고
    internal = ["보존지구·상업지역 등은 점수가 높아도 정비 대상이 아닐 수 있음 — 용도지역 미반영(D-2 수검)."]
    uc = _user_caveats(internal)
    assert len(uc) == 1 and "보존지구" in uc[0] and "재개발 대상이 아닐" in uc[0]
    assert "D-2" not in uc[0] and "수검" not in uc[0]      # 내부코드 D-2·'수검' 제거


def test_noncandidate_report_hallucination_zero():  # 결함 5·6 통합(환각 0 유지)
    rep = generate_report(_NONCAND, complete_fn=lambda s, u: _template_report(
        _display_facts(_NONCAND), _user_caveats(_NONCAND["caveats"])))
    assert rep["hallucination"]["ok"]
    assert "R15" not in rep["report_text"] and "39%" not in rep["report_text"]

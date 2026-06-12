"""report 회귀 테스트 — 표시포맷·환각검증·템플릿폴백(순수/mock, 실 LLM 없음).

실행: python -m pytest tests/test_report.py
"""
from redev.llm.report import _display_facts, _template_report, generate_report, verify_numbers

_DATA = {
    "candidate": True, "b1_score": 0.94,
    "stages": {
        "예언_환경점수": {"status": "ok", "result": {"label": "재개발 환경 점수", "rank_top_pct": 77.3, "caveats": ["환경점수 caveat"]}},
        "진단_요건": {"status": "ok", "result": {"path": "재개발", "metrics": {"old_area_ratio": 0.94, "abut_ratio": 0.5}}},
        "진단_시세맥락": {"status": "ok", "result": {"land_share_pyung_man": 1658.0, "newbuild_exclu_pyung_man": 5446.0, "caveats": ["시세 caveat"]}},
    },
    "caveats": ["투자 권유 아님(R15)"],
}


def test_display_facts_preformats_strings():
    f = _display_facts(_DATA)
    assert f["환경점수"] == "재개발 환경 점수 상위 77.3%"
    assert f["노후도"] == "노후·불량 연면적 94%"
    assert "1,658만원" in f["시세맥락"] and "5,446만원" in f["시세맥락"]


def test_verify_numbers_catches_hallucination():
    f = _display_facts(_DATA)
    assert verify_numbers("환경 점수 상위 77.3%, 노후도 94%.", f)["ok"]          # 표시값 그대로 → 합격
    bad = verify_numbers("수익률 200% 보장, 9억 상승.", f)                       # 200·9 창작
    assert not bad["ok"] and "200" in bad["unmatched"]
    assert verify_numbers("### 1. 될까\n상위 77.3%\n### 2. 얼마", f)["ok"]        # ★절 번호 1.2. 면제


def test_template_report_has_facts_and_caveats():
    f = _display_facts(_DATA)
    cav = ["투자 권유 아님(R15)", "시세 caveat"]
    rep = _template_report(f, cav)
    assert "재개발 환경 점수 상위 77.3%" in rep and "투자 권유 아님(R15)" in rep
    assert verify_numbers(rep, f, cav)["ok"]                                   # 템플릿은 환각 0 보장(caveat 포함)


def test_generate_report_llm_and_fallback():
    # mock LLM: 표시값 그대로 → 환각 0, source llm
    ok = generate_report(_DATA, complete_fn=lambda s, u: "환경 점수 상위 77.3% [환경점수]")
    assert ok["source"] == "llm" and ok["hallucination"]["ok"]
    # LLM 실패 → 템플릿 폴백
    def boom(s, u):
        raise RuntimeError("fail")
    fb = generate_report(_DATA, complete_fn=boom)
    assert fb["source"] == "template" and fb["hallucination"]["ok"]

"""client/layer3 회귀 테스트 — 재시도 로직·JSON파싱·무신호·폴백(mock, 실 LLM 없음).

실행: python -m pytest tests/test_nlp.py
"""
from redev.llm.client import LLMError, _is_transient, _retry_delay
from redev.nlp.layer3 import _parse_json, social_signals


# ── client (순수 로직) ──
def test_is_transient_classifies_429():
    assert _is_transient("429 RESOURCE_EXHAUSTED") and _is_transient("503 UNAVAILABLE")
    assert not _is_transient("400 INVALID_ARGUMENT")


def test_retry_delay_parses_server_hint():
    assert _retry_delay("... 'retryDelay': '55s' ...") == 55.0
    assert _retry_delay("no hint") is None


# ── layer3 ──
def test_parse_json_strips_fence():
    assert _parse_json('```json\n{"signals": []}\n```') == {"signals": []}


def test_social_signals_extracts_and_attaches_source():
    """mock LLM JSON → 신호 + 출처URL 부착, status '신호 있음'."""
    corpus = [{"text": "...", "source": "S", "url": "http://x"}]
    mock = lambda sys, usr: '{"signals":[{"type":"갈등","direction":"악재","evidence":"인용문"}]}'
    r = social_signals("가상구역", corpus=corpus, complete_fn=mock)
    assert r["status"] == "신호 있음" and r["signals"][0]["type"] == "갈등"
    assert r["signals"][0]["source_url"] == "http://x"


def test_social_signals_no_corpus_is_normal():
    """★무신호 = '신호 없음' 정상(억지 생성 0)."""
    r = social_signals("x", corpus=[])
    assert r["status"] == "신호 없음" and r["signals"] == []


def test_social_signals_fallback_on_llm_error():
    """LLM 실패 → 폴백(빈 신호, 안 죽음)."""
    def boom(s, u):
        raise LLMError("fail")
    r = social_signals("x", corpus=[{"text": "a", "url": "u"}], complete_fn=boom)
    assert r["status"] == "신호 없음" and "폴백" in r["reason"]

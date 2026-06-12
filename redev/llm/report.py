"""report.py — ⑨ 종합 리포트 (Phase 7, ③). 설계: docs/design/report.md.

run() 7단계 JSON → 5종 판단(될까·얼마·언제·리스크·진입) 한국어 리포트. ★LLM은 숫자를 짓지
않는다(규칙4): 파이프라인이 ★표시용 문자열을 미리 만들고("상위 77.3%"), LLM은 그걸 글자 그대로
쓴다(변형 금지) → verify_numbers가 정확 문자열 대조(반올림·단위변환 모호매칭 불필요, 기준 엄격).
LLM 실패 시 템플릿 폴백(환각 0 보장). 모든 주장에 [모듈.값] 태그, caveat 누락 금지(정직성).
"""
from __future__ import annotations

import re

# 환각검증 allowlist — 구조적 서수만(절 번호 1~5, "5종 판단"의 5). 의미수치는 면제 안 함.
_ALLOWLIST = {"1", "2", "3", "4", "5"}


def _display_facts(data: dict) -> dict:
    """run() 결과 → ★표시용 문자열 사전(LLM이 글자 그대로 쓸 최종 표기). 숫자 포맷 고정."""
    f, st = {}, data.get("stages", {})
    f["판정"] = "후보 클러스터 속함" if data.get("candidate") else "후보 클러스터 아님(저신뢰)"
    if data.get("b1_score") is not None:
        f["환경유사도점수"] = f"{data['b1_score']}"
    fe = st.get("예언_환경점수", {}).get("result")
    if fe:
        f["환경점수"] = f"{fe['label']} 상위 {fe['rank_top_pct']}%"
    rq = st.get("진단_요건", {}).get("result")
    if rq:
        m = rq["metrics"]
        f["요건판정"] = f"{rq['path']}"
        f["노후도"] = f"노후·불량 연면적 {_pct(m.get('old_area_ratio'))}"
        f["접도율"] = f"접도율 {_pct(m.get('abut_ratio'))}"
    mc = st.get("진단_시세맥락", {}).get("result")
    if mc:
        f["시세맥락"] = (f"인근 빌라 대지지분 평당 {_won(mc.get('land_share_pyung_man'))} / "
                      f"신축 아파트 전용 평당 {_won(mc.get('newbuild_exclu_pyung_man'))}")
    el = st.get("진입_eligibility", {}).get("result")
    if el:
        t = el["진단_토허"]
        f["토허"] = f"토허 {'적용' if t['toheo_applies'] else '미적용'}, 갭투자 {'불가' if not t['gap_investment_possible'] else '가능'}"
        sr = el["예언_잔여기간"]
        if sr.get("known"):
            r = sr["remaining_years"]
            f["잔여기간"] = f"{sr['stage']} 단계, 잔여 {r['min']}~{r['max']}년(통상 {r['typical']}년)"
    cases = data.get("retrieval", {}).get("matches", [])
    if cases:
        c = cases[0]
        f["유사사례"] = f"{c['zone_id']} {int(round(c['similarity']*100))}% 유사({c.get('t')} 지정)"
    soc = data.get("social", {})
    f["사회신호"] = soc.get("status", "신호 없음")
    return f


def _pct(x):
    return "값없음" if x is None or (isinstance(x, float) and x != x) else f"{round(x*100)}%"


def _won(x):
    return "값없음" if x is None or (isinstance(x, float) and x != x) else f"{int(round(x)):,}만원"


def _all_caveats(data: dict) -> list:
    """입력 전체의 caveats 수집(누락 0 검사·리포트용)."""
    cav = list(data.get("caveats", []))
    for s in data.get("stages", {}).values():
        r = s.get("result")
        if isinstance(r, dict):
            cav += r.get("caveats", [])
            for v in r.values():
                if isinstance(v, dict):
                    cav += v.get("caveats", [])
    return list(dict.fromkeys(cav))         # 순서보존 중복제거


def _nums(text: str) -> set:
    """숫자 토큰 추출(콤마·소수·%·연도) → 정규화(콤마/% 제거, ★서수 trailing dot 제거).

    '### 1.' 같은 절 번호는 '1.'로 잡혀 allowlist '1'과 어긋난다 → 끝 dot 제거('1.'→'1').
    소수('0.94')는 끝이 숫자라 영향 없음.
    """
    return {t.replace(",", "").rstrip("%").rstrip(".") for t in re.findall(r"\d[\d,]*\.?\d*%?", text)}


def verify_numbers(report: str, facts: dict, caveats=()) -> dict:
    """★환각 검증 — 리포트 숫자가 원본(표시값 facts + caveats)에 전부 있나. 불일치=환각.

    caveats도 verbatim 원본(R15·39% 등 코드·수치 포함)이라 allowed에 넣는다 — 그 텍스트를
    그대로 옮긴 건 환각 아님. allowlist는 구조적 서수(절 번호)만.
    """
    src = list(facts.values()) + list(caveats)
    allowed = (set().union(*[_nums(v) for v in src]) if src else set()) | _ALLOWLIST
    unmatched = sorted(_nums(report) - allowed)
    return {"ok": not unmatched, "unmatched": unmatched}


def _template_report(facts: dict, caveats: list) -> str:
    """결정론 템플릿(폴백·환각 0 기준선). 표시문자열을 고정 양식에 채움."""
    L = ["[재개발 투자 판단 리포트 — 템플릿]"]
    order = ["판정", "환경점수", "요건판정", "노후도", "접도율", "시세맥락", "토허", "잔여기간", "유사사례", "사회신호"]
    for k in order:
        if k in facts:
            L.append(f"- {k}: {facts[k]}")
    L.append("\n[한계·주의]")
    L += [f"- {c}" for c in caveats]
    return "\n".join(L)


_SYSTEM = (
    "너는 재개발 투자 판단 리포트 작성자다. 주어진 '표시값'과 caveats만으로 한국어 리포트를 쓴다. "
    "★절대 규칙: (1) 5종 판단(될까·얼마·언제·리스크·진입) 구조. (2) 모든 주장 문장 끝에 근거 태그 "
    "[키=표시값]을 붙여라. (3) ★표시값의 숫자·문자열을 글자 그대로 사용하고 절대 변형(반올림·단위변환·"
    "재계산)하지 마라 — JSON에 없는 숫자를 새로 만들면 안 된다. (4) 입력 caveats를 '한계·주의' 절에 "
    "하나도 빠짐없이 옮겨라. (5) 투자 권유 표현 금지(참고치). 설명은 간결하게."
)


def generate_report(data: dict, *, complete_fn=None) -> dict:
    """LLM 언어화 + 환각검증 + 폴백. 반환: {report_text, source, hallucination}."""
    import json
    facts = _display_facts(data)
    caveats = _all_caveats(data)
    if complete_fn is None:
        try:
            from redev.llm.client import complete as complete_fn
        except Exception:
            complete_fn = None
    if complete_fn is not None:
        try:
            user = json.dumps({"표시값": facts, "caveats": caveats}, ensure_ascii=False, indent=2)
            text = complete_fn(_SYSTEM, user)
            return {"report_text": text, "source": "llm", "hallucination": verify_numbers(text, facts, caveats)}
        except Exception as e:
            data = {**data, "_llm_error": str(e)[:80]}
    text = _template_report(facts, caveats)
    return {"report_text": text, "source": "template", "hallucination": verify_numbers(text, facts, caveats)}

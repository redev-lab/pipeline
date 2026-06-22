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
    """run() 결과 → ★표시용 문자열 사전(LLM이 글자 그대로 쓸 최종 표기). 숫자 포맷 고정.

    계약 v1.1(report.md §11): ① 결론 머리문장 ② 환경점수·요건판정 항상 채움("—" 금지, 없으면
    '산출 불가(사유)') ③ candidate=False+점수높음이면 후보판정설명으로 모순 해소 ④ 잔여기간은
    known일 때만, 아니면 단계상태 문구(단계·기간 누수 금지).
    """
    f, st = {}, data.get("stages", {})
    cand = data.get("candidate")
    v = data.get("verdict") or {}
    if v.get("headline"):
        f["결론"] = v["headline"]                          # ⑥ 맨 위 한 문장 결론(결정론)
    conf = data.get("confidence")                          # ★신뢰도(점수-라벨 역전 방지)
    tag = f"({conf})" if conf else ""
    # ★in_zone(실제 지정구역) ≠ cand(환경 유사 군집) — '지정됨' 오독 차단(§defect 2)
    # ★지정구역은 candidate 여부·환경점수 순위와 무관하게 '지정 정비구역'이 주 라벨(환경점수는 부가).
    if data.get("in_zone"):
        f["판정"] = f"지정 정비구역{tag}"
    elif cand:
        f["판정"] = f"환경 유사 군집 속함 — 지정 아님{tag}"
    else:
        f["판정"] = f"환경 유사 군집 아님{tag}"
    # ★raw 환경유사도점수(b1_score)는 표시 안 함 — 점수가 0.97+에 포화돼 절대값이 오도(0.977이 '높아
    #   보이나' 상대순위는 중하위). 사용자엔 rank_phrase(상대순위)만 노출. raw는 out['b1_score'] 메타로 보존.

    # ② 환경점수 — 항상 채움. ★상대순위(rank_phrase) — 절대값 아님을 명시(포화 오도 차단).
    fe = (st.get("예언_환경점수", {}) or {}).get("result")
    if fe:
        f["환경점수"] = f"{fe['label']} {fe['rank_phrase']}(전 구역 상대순위)"
    else:
        f["환경점수"] = "산출 불가(그래프 노드 외 — 점수 미산출)"
    # ① 모순 해소 — ★점수는 높은데(상위 N% 이내) 후보 아님일 때만: 두 사실의 관계를 한 문장으로.
    #    점수가 낮으면 모순이 없으므로(verdict '대상 아님'이 설명) 이 문장을 붙이지 않는다.
    if not cand and fe:
        from redev.config import load_infer_config
        cfg = load_infer_config()["cluster"]
        if fe["rank_top_pct"] <= cfg["tight_top_pct"]:
            f["후보판정설명"] = (f"환경 점수는 {fe['rank_phrase']}이나 "
                              f"연결 군집 기준(최소 {cfg['min_nodes']}필지) 미달로 후보 경계엔 미포함")

    # ② 요건판정 — 항상 채움(클러스터 없으면 사유)
    rq = (st.get("진단_요건", {}) or {}).get("result")
    if rq:
        m = rq["metrics"]
        f["요건판정"] = f"{rq['path']}"
        f["노후도"] = f"노후·불량 연면적 {_pct(m.get('old_area_ratio'))}"
        f["접도율"] = f"접도율 {_pct(m.get('abut_ratio'))}"
    else:
        f["요건판정"] = f"산출 불가({st.get('진단_요건', {}).get('reason', '판정 불가')})"

    mc = (st.get("진단_시세맥락", {}) or {}).get("result")
    if mc:
        def _wp(x):                                       # 평당가 — 결측은 '거래 부족'(어색한 '값없음/평' 회피)
            return "거래 부족" if x is None or (isinstance(x, float) and x != x) else f"{int(round(x)):,}만원/평"
        lp, npv = mc.get("land_provenance"), mc.get("newbuild_provenance")   # ★출처·표본(환경점수 수준 정직성)
        f["시세맥락"] = (f"빌라 대지지분 {_wp(mc.get('land_share_pyung_man'))}" + (f"({lp})" if lp else "")
                      + f" · 신축 전용 {_wp(mc.get('newbuild_exclu_pyung_man'))}" + (f"({npv})" if npv else ""))
    else:
        f["시세맥락"] = "시세 산출 불가(반경 내 거래 부족)"

    # ★계획정보(고시 추출, §5) — verified만 단정, flagged는 '(잠정)', 출처·최신 플래그 동봉
    pi = (st.get("진단_계획정보", {}) or {}).get("result")
    if pi and pi.get("attrs"):
        parts = []
        for a in ["용적률", "계획세대수", "구역면적", "건폐율"]:
            it = pi["attrs"].get(a)
            if it:
                g = it.get("grade")
                # verified·manual_verified=단정 / ocr_검토필요=OCR 잠정 / flagged=잠정
                tag = "" if g in ("verified", "manual_verified") else "(OCR 잠정)" if g == "ocr_검토필요" else "(잠정)"
                parts.append(f"{a} {_attr_display(a, it['raw'])}{tag}")
        if parts:
            src = f"서울고시 {pi['고시번호']} 기준"
            if pi.get("flags"):
                src += ", 후속 변경 미반영"           # ★흑석2 최신 미반영 등
            f["계획정보"] = " · ".join(parts) + f" ({src})"
    el = (st.get("진입_eligibility", {}) or {}).get("result")
    if el:
        t = el["진단_토허"]
        f["토허"] = f"토허 {'적용' if t['toheo_applies'] else '미적용'}, 갭투자 {'불가' if not t['gap_investment_possible'] else '가능'}"
        sr = el["예언_잔여기간"]
        if sr.get("known"):                                # ③ 후보 구역+단계 입력일 때만 기간 노출
            r = sr["remaining_years"]
            f["잔여기간"] = f"{sr['stage']} 단계, 잔여 {r['min']}~{r['max']}년(통상 {r['typical']}년)"
        else:                                              # 구역 아님/단계 미입력 → 기간 없이 사유만
            f["단계상태"] = sr.get("note", "사업 단계 정보 없음")
    cases = data.get("retrieval", {}).get("matches", [])
    if cases:
        c = cases[0]
        name = c.get("display_name") or c.get("zone_id")   # §B-3: 표시명 우선, 원시코드는 폴백·메타
        f["유사사례"] = f"{name} {int(round(c['similarity']*100))}% 유사"
    soc = data.get("social", {})
    f["사회신호"] = soc.get("status", "신호 없음")
    if v.get("class"):
        f["행동분류"] = v["class"]                          # ⑥ 5절 행동 관점 요약 기준(숫자 없음)
    return f


# ★내부 caveat(R##·§·★·설계메모) → 사용자 언어 번역표(계약 §11-4). 문장에 숫자 없음 → 환각검증 안전.
#   규칙: 어떤 내부 caveat이 키워드를 담으면 그 사용자 문장을 노출(순서 고정·중복 제거). 매칭 안 된
#   dev/ops caveat(상업배포·법률검토)은 사용자 한계가 아니므로 메타 전용(노출 안 함).
_CAVEAT_RULES = [
    (("투자 권유",), "이 리포트는 투자 권유가 아니라 데이터 기반 참고치입니다."),
    (("거친 필터", "정밀 경계 아님"), "후보 경계는 대략적 추정이며 정밀한 사업 경계가 아닙니다."),
    (("추진 성공", "환경 유사도"), "환경 점수는 '닮은 동네인지'를 볼 뿐, 실제 사업 추진·성공 여부는 예측하지 않습니다."),
    (("보존지구", "정비 대상이 아닐"), "보존지구·상업지역 등은 점수가 높아도 재개발 대상이 아닐 수 있습니다(용도지역 미반영)."),
    (("상승여력",), "예상 수익(상승여력)은 분담금 등 정보가 더 필요해 현재 산정하지 않습니다."),
    (("토허",), "토지거래허가 규제는 수시로 바뀌니 사용 시점의 최신 고시를 확인하세요."),
    (("범위·변동 큼",), "사업 단계별 잔여 기간은 분쟁·경기 등 외부 요인으로 크게 달라질 수 있습니다."),
    (("의제처리 재개발구역 데이터 기준",),
     "‘지정 아님’은 ‘우리 데이터에 없음’을 뜻합니다 — 지정 판정은 의제처리 재개발구역 기준이라 일반 주택재개발·가로주택 등 일부 지정구역은 누락될 수 있습니다(실제 지정은 정비사업 정보몽땅 등에서 확인)."),
]


def _user_caveats(internal: list) -> list:
    """내부 caveat 리스트 → 사용자 언어 caveat(번역표 매칭, 순서 고정·중복 제거)."""
    out = []
    for keys, sentence in _CAVEAT_RULES:
        if sentence not in out and any(any(k in c for k in keys) for c in internal):
            out.append(sentence)
    return out


_UNIT = {"용적률": "%", "건폐율": "%", "구역면적": "㎡", "계획세대수": "세대"}


def _attr_display(attr: str, raw: str) -> str:
    """계획정보 표시 — 속성 단위 부착(면적 ㎡·건폐율 % 등). 이미 단위 있으면 원형 유지.

    예: '30이하'(건폐율)→'30% 이하', '187,669.0'(면적)→'187,669㎡', '231.54%'→그대로.
    숫자 자릿값은 보존(환각검증 — 계획정보 표시값이 곧 허용 원천이라 self-allowed).
    """
    u = _UNIT.get(attr, "")
    s = (raw or "").strip()
    if u and u in s:                                      # 이미 단위(% ㎡ 세대) 있으면 그대로
        return s
    m = re.match(r"\s*([\d,]+(?:\.\d+)?)(.*)$", s)
    if not m:
        return s
    num, rest = m.group(1), m.group(2).strip()
    if attr == "구역면적" and "." in num:                  # 187,669.0 → 187,669 (불필요 소수 제거)
        num = num.rstrip("0").rstrip(".")
    return f"{num}{u}" + (f" {rest}" if rest else "")


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


# 본문 5절 — (라벨, 표시값 키들). 결론·환경점수·판정은 카드(화면)로 빠지므로 본문엔 없음.
_SECTIONS = [
    ("될까", ["후보판정설명", "요건판정", "노후도", "접도율"]),
    ("얼마", ["시세맥락", "계획정보"]),
    ("언제", ["잔여기간", "단계상태"]),
    ("리스크", ["유사사례", "사회신호"]),
    ("진입", ["토허"]),
]


def _template_report(facts: dict) -> str:
    """결정론 템플릿(폴백·환각 0 기준선). 깨끗한 5절(태그·결론·한계 없음 — 화면이 카드/접힘으로 표시)."""
    L = []
    for label, keys in _SECTIONS:
        vals = [facts[k] for k in keys if k in facts]
        if vals:
            L.append(f"### {label}")
            L.append(" · ".join(vals))
    return "\n".join(L)


# 본문에서 LLM에 주지 않는 '카드' 키(화면 상단 카드·칩이 따로 표시 — 본문 중복 금지).
_CARD_KEYS = {"결론", "판정", "환경점수", "행동분류"}

_SYSTEM = (
    "너는 재개발 투자 판단 리포트 작성자다. 주어진 '표시값'만으로 한국어 본문을 쓴다.\n"
    "★출력 형식 — 정확히 5개 절, 각 절 머리는 '### ' + 라벨, 그 아래 ★1~2줄:\n"
    "### 될까\n### 얼마\n### 언제\n### 리스크\n### 진입\n"
    "★규칙:\n"
    "(1) ★대괄호 태그([키=값])·내부 코드(R숫자·§·★)·인사말·머리말 절대 쓰지 마라 — 깨끗한 평서문만.\n"
    "(2) ★각 절 1~2줄, 명사형으로 간결하게(군더더기·수식어 금지). 예: '빌라 대지지분 6,360만원/평'.\n"
    "(3) 결론 문장·환경 순위·후보/지정 판정은 ★쓰지 마라 — 화면 상단 카드가 따로 표시한다(중복 금지).\n"
    "(4) ★표시값의 숫자·문자열 그대로(변형·창작 금지). 표시값에 없으면 쓰지 마라.\n"
    "(4-1) ★시세값에는 표시값 '시세맥락'의 괄호 출처(예: '동 평균 N건', '반경 100m 실거래 N건', '반경 1km 신축 N건')를 "
    "반드시 그대로 함께 적어라 — 생략 금지(사용자가 '주변 실거래'로 오인 방지).\n"
    "(5) '언제'에 '잔여기간' 없으면 단계·기간을 만들지 말고 '단계상태' 문구 그대로.\n"
    "(6) '한계·주의'는 쓰지 마라 — 화면이 따로 접힘으로 표시한다. 투자 권유 표현 금지(참고치)."
)


def generate_report(data: dict, *, complete_fn=None) -> dict:
    """LLM 언어화 + 환각검증 + 폴백. 반환: {report_text, source, hallucination, caveats_internal}.

    ★사용자 노출 caveat은 번역표(_user_caveats, 숫자 없음)만. 내부 caveat(코드 포함)은 메타로만 보존.
    환각검증의 허용 숫자원천 = 표시값 + 사용자 caveat(숫자 없음) → 내부 코드 숫자는 허용집합에서 빠진다.
    """
    import json
    facts = _display_facts(data)
    body_facts = {k: v for k, v in facts.items() if k not in _CARD_KEYS}   # 카드 키 제외(본문 중복 방지)
    internal = _all_caveats(data)
    user_cav = _user_caveats(internal)
    if complete_fn is None:
        try:
            from redev.llm.client import complete as complete_fn
        except Exception:
            complete_fn = None
    # ★공통 메타: source_facts(출처 — 화면 툴팁/클릭용, 표시 안 함) + caveats. 환각검증은 facts 전체 기준.
    meta = {"caveats_user": user_cav, "caveats_internal": internal, "source_facts": facts}
    if complete_fn is not None:
        try:
            user = json.dumps({"표시값": body_facts}, ensure_ascii=False, indent=2)
            text = complete_fn(_SYSTEM, user)
            return {"report_text": text, "source": "llm",
                    "hallucination": verify_numbers(text, facts, user_cav), **meta}
        except Exception as e:
            data = {**data, "_llm_error": str(e)[:80]}
    return {"report_text": _template_report(body_facts), "source": "template",
            "hallucination": verify_numbers(_template_report(body_facts), facts, user_cav), **meta}

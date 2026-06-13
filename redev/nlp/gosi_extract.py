"""gosi_extract.py — 고시 본문 → 구조화 추출 (LLM). 설계: docs/design/gosi_parse.md §3·§3-1.

★LLM은 숫자를 짓지 않는다(규칙4): 본문에서 **해당 구역의 변경후(최종) 확정값**만 원문 그대로
뽑고(`raw`), 근거 문장(`sentence`)을 첨부한다. 없으면 `미기재`. 값(숫자) 파싱·범위·검증은
gosi_verify가 결정론으로 한다(LLM이 계산하면 천장 리스크 — 트랙 취지 위반).

핵심 난관(§3-1): 한 문서에 수치 여럿(흑석2 694/1,012/807세대) → 구역명 앵커·변경전후 라벨·
헤더 변종으로 '해당 구역 변경후'를 고른다. 이 판단을 LLM이 하고, 근거 문장으로 손대조가 닫는다.
"""
from __future__ import annotations

import json
import re

# 추출 대상(키, 단위 힌트). 분양가·단계일자는 보너스(있으면 줍되 없어도 OK) — v1 핵심 4종.
ATTRS = ["용적률", "건폐율", "계획세대수", "구역면적"]

_SYSTEM = (
    "너는 정비계획 결정고시 본문에서 사실 수치를 추출하는 도구다. 추정·창작 절대 금지 — 본문에 "
    "적힌 것만 원문 그대로 뽑는다.\n"
    "★추출 대상(해당 구역): 용적률·건폐율·계획세대수·구역면적.\n"
    "★선택 규칙(한 문서에 수치 여럿일 때):\n"
    "1) 구역명 앵커: 주어진 '대상 구역명'이 명시된 표/문장의 값만. 지구 전체나 다른 구역(예 흑석9) 값은 제외.\n"
    "2) 변경전후: '기정|변경' 또는 '당초|변경후'로 나뉘면 반드시 변경(후) 값을 택한다. 신규 지정 고시면 그대로.\n"
    "3) 헤더 변종: 용적률은 기준/허용/계획/상한 중 라벨을 'label'에 적고 값은 계획(없으면 허용)을 우선.\n"
    "4) 다중 획지면 구역 총계(계획세대수 합), 면적은 정비구역 전체.\n"
    "5) 해당 구역에 명시 없으면 그 항목은 \"미기재\".\n"
    "★출력: 아래 JSON만(설명 금지). 각 항목은 {\"raw\":\"원문표기 그대로(예 190.0%, 2,464세대)\","
    " \"label\":\"기준용적률 등\", \"변경구분\":\"변경후|신규|기정\", \"sentence\":\"근거 문장 원문 일부\"}"
    " 또는 문자열 \"미기재\".\n"
    "{\"용적률\":..., \"건폐율\":..., \"계획세대수\":..., \"구역면적\":...}"
)


def _parse_json(text: str) -> dict:
    """LLM 응답에서 JSON 객체만 안전 추출(코드펜스·잡텍스트 방어)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def extract_attrs(text: str, *, zone_name: str, 고시번호: str, 고시일자: str,
                  complete_fn=None) -> dict:
    """본문 텍스트 → 해당 구역 추출 결과. 반환:
    {zone_name, 고시번호, 고시일자, attrs: {용적률: {raw,label,변경구분,sentence}|"미기재", ...}, source: "llm"|"none"}.

    ★검증 전 단계 — 여기서 나온 raw·sentence를 gosi_verify가 원문 대조·범위·등급한다.
    """
    if complete_fn is None:
        try:
            from redev.llm.client import complete as complete_fn
        except Exception:
            complete_fn = None
    if complete_fn is None:
        return {"zone_name": zone_name, "고시번호": 고시번호, "고시일자": 고시일자,
                "attrs": {}, "source": "none"}

    from redev.data.ingest.gosi_body import focus_text   # 목표 수치 주변으로 압축(프롬프트 축소·정확도)
    focused = focus_text(text, zone_name=zone_name)
    user = f"대상 구역명: {zone_name}\n고시번호: {고시번호}\n\n[본문(발췌)]\n{focused}"
    raw = complete_fn(_SYSTEM, user)
    parsed = _parse_json(raw)
    attrs = {}
    for k in ATTRS:
        v = parsed.get(k)
        if isinstance(v, dict) and v.get("raw"):
            attrs[k] = {"raw": str(v.get("raw")), "label": v.get("label"),
                        "변경구분": v.get("변경구분"), "sentence": str(v.get("sentence") or "")}
        else:
            attrs[k] = "미기재"
    return {"zone_name": zone_name, "고시번호": 고시번호, "고시일자": 고시일자,
            "attrs": attrs, "source": "llm"}

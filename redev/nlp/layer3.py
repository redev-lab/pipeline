"""layer3.py — 사회신호 추출 (Phase 7, ② 직교 축). 설계: nlp_layer3.md §2.

노후도(물리)로 안 잡히는 축 — 갈등·정체·진행 신호를 텍스트에서 뽑는다(R18 프록시). ★LLM은
숫자·사실을 짓지 않고(규칙4) 주어진 문장에서 구조화 신호만 분류·인용. ★신호 부재 = "신호 없음"
이 정상 출력(억지 신호=불합격). 추론 시점 피처(학습 피처 아님 — §4 아키텍처).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_CORPUS_DIR = Path("_data/corpus")

_SYSTEM = (
    "너는 한국 도시정비사업(재개발) 뉴스/공고에서 '사회신호'만 뽑는 추출기다. "
    "주어진 기사 텍스트에서 갈등(주민반대·소송·분쟁)·정체(중단·표류·무산)·진행(속도·인가) 신호를 찾는다. "
    "반드시 JSON만 출력: {\"signals\":[{\"type\":\"갈등|정체|진행\",\"direction\":\"악재|호재|중립\","
    "\"evidence\":\"원문에서 인용한 근거 문장\"}]}. "
    "★규칙: evidence는 반드시 원문 문장 그대로 인용. 사실·숫자·내용을 창작하지 마라. "
    "신호가 없으면 빈 배열 {\"signals\":[]}. 설명·서술 금지, JSON만."
)


def load_corpus(zone_id: str) -> list:
    """데모 코퍼스 로드(_data/corpus/<zone>.jsonl). 없으면 빈 리스트(→ '신호 없음' 정상).

    ★실존 구역엔 실제 기사 인용만(가짜 부정정보 금지). 보도 없으면 '신호 없음'이 정직한 데모.
    """
    f = _CORPUS_DIR / f"{zone_id}.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_json(text: str) -> dict:
    """LLM 응답에서 JSON 추출(```json 펜스 제거 후 파싱). 실패 시 빈 신호."""
    s = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        return json.loads(m.group(0)) if m else {"signals": []}


def _extract_one(item: dict, complete_fn) -> list:
    """기사 1건 → 신호 목록(LLM 구조화 추출). 출처 URL을 신호에 붙인다."""
    out = _parse_json(complete_fn(_SYSTEM, item["text"]))
    sigs = out.get("signals", []) if isinstance(out, dict) else []
    for s in sigs:
        s["source_url"] = item.get("url")
        s["source"] = item.get("source")
    return sigs


def social_signals(zone_id: str | None = None, *, corpus: list | None = None, complete_fn=None) -> dict:
    """공개 진입점 — 구역의 사회신호 집계. ★무신호·코퍼스부재·LLM실패 모두 '신호 없음' 정상.

    corpus 직접 주입 가능(테스트). complete_fn 미지정 시 client.complete(폴백은 호출부가 잡음).
    """
    docs = corpus if corpus is not None else load_corpus(zone_id or "")
    if not docs:
        return {"zone_id": zone_id, "signals": [], "status": "신호 없음", "reason": "코퍼스 없음"}
    if complete_fn is None:
        from redev.llm.client import complete as complete_fn
    signals = []
    try:
        for item in docs:
            signals.extend(_extract_one(item, complete_fn))
    except Exception as e:                            # LLM 실패 → 폴백(빈 신호, 안 죽음)
        return {"zone_id": zone_id, "signals": [], "status": "신호 없음",
                "reason": f"LLM 실패(폴백): {type(e).__name__}"}
    return {"zone_id": zone_id, "signals": signals,
            "status": "신호 있음" if signals else "신호 없음", "n_docs": len(docs)}

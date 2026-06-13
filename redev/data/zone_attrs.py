"""zone_attrs.py — 고시 추출 계획정보를 zone 단위로 영속화(ZoneTable 보강). 설계 §5.

고시 본문(gosi_body)→추출(gosi_extract)→검증(gosi_verify) 결과를 zone_id별로 모아 캐시.
리포트·retrieval이 이 스토어를 조회(LLM 재호출 없이). ★verified만 단정 표시, flagged는 잠정,
흑석2 등 '최신 미반영' 플래그·출처(고시번호·고시일자) 보존(리포트 인용·환각검증 유지).
"""
from __future__ import annotations

import json

from redev.paths import DATA

_CACHE = DATA / "processed/zone_attrs.json"
_DIR = DATA / "raw/고시정보"

# 표본 등록부(고시번호↔zone_id↔파일↔플래그). 전역 확장 시 ZoneTable 조인으로 자동 생성 대체.
SAMPLES = [
    {"zone_id": "11290NTC202409250002", "zone": "장위15구역", "고시번호": "2024-448", "고시일자": "2024-09-19",
     "file": "[서고시 제2024-448호] 장위재정비촉진지구 변경 지정, 재정비촉진계획(장위15구역) 변경결정 및 지형도면 고시(2024. 9. 19.)성북구.pdf", "flags": []},
    {"zone_id": "11590NTC202508040004", "zone": "흑석2구역", "고시번호": "2025-426", "고시일자": "2025-07-31",
     "file": "서울특별시_제2025-426호_고시.pdf", "flags": ["최신 미반영(서울시 2025-659 변경안 협의중·미입수)"]},
    {"zone_id": "11170NTC202411110006", "zone": "청파2구역", "고시번호": "2024-519", "고시일자": "2024-10-31",
     "file": "용산2024-519.pdf", "flags": []},
    {"zone_id": "11530NTC202603170002", "zone": "가리봉1구역", "고시번호": "2026-18", "고시일자": "2026-02-12",
     "file": "구로구_제2026-18호_고시.pdf", "flags": []},
    {"zone_id": "11380NTC202008310004", "zone": "응암제2구역", "고시번호": "2020-133", "고시일자": "2020-07-30",
     "file": "응암.pdf", "flags": []},
]


def build_zone_attrs(*, complete_fn=None, samples=None) -> dict:
    """표본 고시 본문 → 추출+검증 → zone_id별 계획정보 스토어 저장. (LLM 추출, 빌드 1회.)"""
    from redev.data.ingest.gosi_body import read_gosi
    from redev.nlp.gosi_extract import ATTRS, extract_attrs
    from redev.nlp.gosi_verify import verify_extraction

    store = {}
    for s in (samples or SAMPLES):
        g = read_gosi(_DIR / s["file"])
        ex = extract_attrs(g["text"], zone_name=s["zone"], 고시번호=s["고시번호"],
                           고시일자=s["고시일자"], complete_fn=complete_fn)
        vr = verify_extraction(ex, g["text"], table_rows=g["rows"], grids=g["grids"])
        attrs = {}
        for a in ATTRS:
            r = vr["results"][a]
            if r["grade"] in ("verified", "flagged"):        # missing/rejected는 스토어 제외
                attrs[a] = {"value": r["value"], "raw": r["raw"], "label": r["label"], "grade": r["grade"]}
        store[s["zone_id"]] = {"zone_name": s["zone"], "고시번호": s["고시번호"], "고시일자": s["고시일자"],
                               "flags": s["flags"], "attrs": attrs}
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    return store


def load_zone_attrs() -> dict:
    """zone_id → 계획정보 스토어(없으면 빈 dict — 보강 전에도 파이프라인 동작)."""
    if not _CACHE.exists():
        return {}
    return json.loads(_CACHE.read_text(encoding="utf-8"))

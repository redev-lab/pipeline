"""zone_attrs.py — 고시 추출 계획정보를 zone 단위로 영속화(ZoneTable 보강). 설계 §5.

고시 본문(gosi_body)→추출(gosi_extract)→검증(gosi_verify) 결과를 zone_id별로 모아 캐시.
리포트·retrieval이 이 스토어를 조회(LLM 재호출 없이). ★verified만 단정 표시, flagged는 잠정,
흑석2 등 '최신 미반영' 플래그·출처(고시번호·고시일자) 보존(리포트 인용·환각검증 유지).
"""
from __future__ import annotations

import json
import re

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
    # 1차 배치 신규(디지털 5 — 성북1·불광5는 스캔이라 OCR 대기로 제외)
    {"zone_id": "11590NTC202505120002", "zone": "상도14구역", "고시번호": "2025-179", "고시일자": "2025-04-03",
     "file": "서울특별시_제2025-179호_고시.pdf", "flags": []},
    {"zone_id": "11590NTC202505190004", "zone": "상도15구역", "고시번호": "2025-178", "고시일자": "2025-04-03",
     "file": "서울특별시_제2025-178호_고시.pdf", "flags": []},
    {"zone_id": "11290NTC202504180005", "zone": "석관4구역", "고시번호": "2025-194", "고시일자": "2025-04-10",
     "file": "서울특별시_제2025-194호_고시.pdf", "flags": []},
    {"zone_id": "11290NTC202503060003", "zone": "하월곡1구역", "고시번호": "2025-245", "고시일자": "2025-05-01",
     "file": "서울특별시_제2025-245호_고시.pdf", "flags": []},
    {"zone_id": "11380NTC202411130006", "zone": "불광8구역", "고시번호": "2024-484", "고시일자": "2024-10-17",
     "file": "서울특별시_제2024-484호_고시.pdf", "flags": []},
    # ★성북1(2024-475)·불광5(2025-163)는 스캔 PDF — OCR 느림·오인식 위험으로 ★수동 입력 경로(set_manual).
    #   OCR 코드(gosi_body._ocr_pdf)는 전역 확장 대비 보존, 지금은 미사용.
]


def build_zone_attrs(*, complete_fn=None, samples=None, merge: bool = True) -> dict:
    """표본 고시 본문 → 추출+검증 → zone_id별 계획정보 스토어 저장.

    merge=True면 기존 스토어 로드 후 ★신규 zone_id만 LLM 추출(이미 추출분 재호출 안 함 — 한도 절약).
    """
    from redev.data.ingest.gosi_body import read_gosi
    from redev.nlp.gosi_extract import ATTRS, extract_attrs
    from redev.nlp.gosi_verify import verify_extraction

    store = load_zone_attrs() if merge else {}
    for s in (samples or SAMPLES):
        if merge and s["zone_id"] in store:                  # 이미 추출됨 → 건너뜀(LLM 절약)
            continue
        g = read_gosi(_DIR / s["file"], ocr=True)            # ★스캔본은 OCR(잠정 등급)
        ex = extract_attrs(g["text"], zone_name=s["zone"], 고시번호=s["고시번호"],
                           고시일자=s["고시일자"], complete_fn=complete_fn)
        vr = verify_extraction(ex, g["text"], table_rows=g["rows"], grids=g["grids"], source=g["source"])
        attrs = {}
        for a in ATTRS:
            r = vr["results"][a]
            if r["grade"] in ("verified", "flagged", "ocr_검토필요"):   # missing/rejected만 제외
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


def set_manual(zone_id: str, *, zone_name: str, 고시번호: str, 고시일자: str, flags=(), **attr_raws) -> dict:
    """★수동 입력 통로 — 사람이 원문(PDF) 직접 대조한 값. 스캔 등 추출 불가 구역용(성북1·불광5 등).

    등급 'manual_verified' = 디지털 verified와 동급 신뢰(사람이 원문 대조). OCR 잠정과 구분.
    verbatim/OCR 검증 불필요(사람이 봄), ★범위 가드만 적용(입력 오타 방지 — 범위밖이면 flagged).
    attr_raws: 용적률='250%', 건폐율='30% 이하', 계획세대수='1,234세대', 구역면적='50,000㎡'(원문 표기 그대로).
    스토어에 병합 저장. 반환: 저장된 엔트리.
    """
    from redev.nlp.gosi_verify import RANGES, _value_token

    store = load_zone_attrs()
    attrs = {}
    for a in ("용적률", "건폐율", "계획세대수", "구역면적"):
        raw = attr_raws.get(a)
        if not raw:
            continue
        tok = _value_token(raw)
        val = float(tok) if tok else None
        lo, hi = RANGES.get(a, (float("-inf"), float("inf")))
        grade = "manual_verified" if (val is not None and lo <= val <= hi) else "flagged"  # 범위밖=입력오타 의심
        attrs[a] = {"value": val, "raw": raw, "label": a, "grade": grade}
    store[zone_id] = {"zone_name": zone_name, "고시번호": 고시번호, "고시일자": 고시일자,
                      "flags": list(flags), "attrs": attrs}
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    return store[zone_id]


_SUFFIX = re.compile(r"(주택재개발|주택재건축|도시환경|주택정비형|재정비촉진|재개발|재건축|정비사업|정비형|구역|뉴타운)")


def norm_zone_name(name: str) -> str:
    """구역명 정규화 — 표기 흔들림 흡수. '가리봉제1구역'·'가리봉1구역'·'가리봉 제1구역' → '가리봉1'.

    제N→N, 사업·구역 접미사 제거, 공백 제거. 무매칭이 오매칭보다 안전하므로 과도 일반화는 피함.
    """
    if not name:
        return ""
    s = re.sub(r"\s+", "", str(name))
    s = re.sub(r"제(\d)", r"\1", s)          # 제1 → 1 (제기동 등은 숫자 앞 '제'만)
    s = _SUFFIX.sub("", s)
    return s


def resolve_to_context(store: dict, ctx_zone_ids, title_map: dict) -> dict:
    """★고시-키 스토어를 *컨텍스트 zone_id*로 재매핑(강건 매칭). 의제처리 NTFC_SN ≠ 고시관리코드 흡수.

    1차 직접: 고시관리코드(스토어 키)가 컨텍스트 zone_id면 그대로(고신뢰).
    2차 구역명: 정규화 구역명이 컨텍스트 zone의 제목에 유일 포함 → 매칭(중신뢰, 자치구 일치 가드).
    다중후보·무매칭 → 연결 안 함(flagged, 사람 확인). ★오매칭보다 무매칭이 안전.
    반환: {context_zone_id: {...attrs, match:{confidence, method, gosi_zone_id}}}, unmatched 리스트는 로그.
    """
    ctx_ids = set(ctx_zone_ids)
    ctx_norm = {zid: norm_zone_name(_zone_name_from_title(title_map.get(zid, ""))) for zid in ctx_ids}
    resolved, unmatched = {}, []
    for gosi_zid, entry in store.items():
        if gosi_zid in ctx_ids:                                   # 1차 직접
            resolved[gosi_zid] = {**entry, "match": {"confidence": "고", "method": "고시관리코드 일치", "gosi_zone_id": gosi_zid}}
            continue
        n = norm_zone_name(entry.get("zone_name", ""))
        gu = gosi_zid[:5]
        cands = [zid for zid in ctx_ids if n and zid[:5] == gu and n == ctx_norm.get(zid)]   # 구역명+자치구
        if len(cands) == 1:                                       # 2차 구역명(유일)
            resolved[cands[0]] = {**entry, "match": {"confidence": "중", "method": "구역명 정규화", "gosi_zone_id": gosi_zid}}
        else:
            unmatched.append({"gosi_zone_id": gosi_zid, "zone_name": entry.get("zone_name"),
                              "reason": "다중후보" if len(cands) > 1 else "무매칭", "n_cands": len(cands)})
    return {"resolved": resolved, "unmatched": unmatched}


def _zone_name_from_title(title: str) -> str:
    """고시 제목에서 구역명 토큰 추출(정규화 입력용). 제목에 'XX제N구역'/'XXN구역' 패턴이 흔함."""
    m = re.search(r"([가-힣]+제?\d+구역)", title or "")
    return m.group(1) if m else (title or "")

"""gosi_verify.py — 고시문 추출 검증 (v1_3-gosi, 본 프로젝트 신규·임포트 0). 설계 §4.

추출(gosi_extract)이 뽑은 각 수치를 결정론으로 검증한다:
1. ★verbatim diff: 추출 raw의 숫자 토큰 + 근거 문장이 **원문에 글자 그대로 존재**하는가
   (원문에 없으면 환각 → reject). 검증은 '맞는 칸인가'가 아니라 '원문에 있나'다 — '맞는 칸'은 손대조(§4).
2. 상식 범위 가드: 용적률 50~600% / 건폐율 20~80% / 세대수≥1 / 면적>0. 밖이면 flagged.
3. 출처 보존: 값마다 고시번호·고시일자·근거 문장 → 리포트가 "용적률 190%(서울고시 2024-448)" 인용.
등급: verified(verbatim+범위 통과) / flagged(범위밖·출처 불일치) / rejected(원문에 숫자 없음=환각) / missing.
"""
from __future__ import annotations

import re

# 상식 범위(규칙5 — 매직넘버 아님, 물리/제도 상식). 밖이면 추출 오류로 플래그.
RANGES = {"용적률": (50.0, 600.0), "건폐율": (20.0, 80.0),
          "계획세대수": (1.0, 1e7), "구역면적": (1.0, 1e8)}

# 표 셀 출처 확인용 라벨 키워드(값이 이 라벨과 같은 표 행에 있으면 출처 확정).
_ATTR_KW = {"용적률": "용적률", "건폐율": "건폐율", "계획세대수": "세대", "구역면적": "면적"}


def _norm(s: str) -> str:
    """공백 전부 제거(셀/줄바꿈 분리에 강건한 substring 대조용)."""
    return re.sub(r"\s+", "", s or "")


def _value_token(raw: str) -> str | None:
    """원문 표기에서 핵심 숫자 토큰 추출(콤마 제거). '2,464세대'→'2464', '190.0%'→'190.0'."""
    m = re.search(r"\d[\d,]*\.?\d*", raw or "")
    return m.group(0).replace(",", "") if m else None


def _cell_has(cell: str, tok: str) -> bool:
    return bool(tok) and tok in _norm(cell).replace(",", "")


def _table_provenance(attr: str, tok: str, *, table_rows=None, grids=None) -> bool:
    """★출처 확정(결정론): 값 tok이 속성 라벨과 같은 표에서 *같은 열/행*에 있나. 산문 평탄화 뭉갬 우회.

    검증을 느슨하게 하는 게 아니다 — 환각 차단(in_source)은 그대로 엄격. 이건 '진짜 표'에서 값+라벨
    위치 일치를 확인하는 *추가* 출처 경로다:
    (A) 헤더-격자: 라벨이 든 헤더 셀의 열 인덱스 j → 같은 열(다른 행) 데이터 셀에 tok (장위15 양식).
    (B) 인라인: 라벨과 tok이 같은 행 (흑석2 '건폐율(%) 54.30 52.26' 양식).
    진짜 깨진 표(라벨·값 위치 못 잡음)는 둘 다 실패 → flagged 정직 유지(억지 통과 금지).
    """
    if not tok:
        return False
    kw = _ATTR_KW.get(attr, attr)
    for table in (grids or []):
        cols = {j for row in table for j, c in enumerate(row) if c and kw in c}   # 라벨 열
        for row in table:
            if cols and any(j in cols and _cell_has(c, tok) for j, c in enumerate(row)):
                return True                                                       # (A) 같은 열
            joined = " ".join(c for c in row if c)
            if kw in joined and _cell_has(joined, tok):
                return True                                                       # (B) 같은 행
    return any(kw in r and _cell_has(r, tok) for r in (table_rows or []))         # 폴백(평탄 행)


def verify_attr(attr: str, item, source_text: str, *, 고시번호: str, 고시일자: str,
                table_rows=None, grids=None) -> dict:
    """한 항목 검증 → {value, raw, label, 변경구분, sentence, 고시번호, 고시일자, grade, checks}."""
    base = {"고시번호": 고시번호, "고시일자": 고시일자}
    if not isinstance(item, dict):                       # "미기재"
        return {**base, "grade": "missing", "raw": None, "value": None}

    raw, sentence = item.get("raw", ""), item.get("sentence", "")
    tok = _value_token(raw)                                    # 콤마 제거된 숫자(예 '2464')
    src, sent = _norm(source_text), _norm(sentence)
    in_source = bool(tok) and tok in src.replace(",", "")      # ★숫자 원문 존재(콤마 무시, 환각 차단)
    sent_in_source = len(sent) >= 8 and sent in src            # 근거 문장 원문 존재(출처 진짜)
    tok_in_sent = bool(tok) and tok in sent.replace(",", "")   # 값이 그 문장에서 나왔나
    table_prov = _table_provenance(attr, tok, table_rows=table_rows, grids=grids)   # ★표 열/행 출처
    value = float(tok) if tok else None
    lo, hi = RANGES.get(attr, (float("-inf"), float("inf")))
    in_range = value is not None and lo <= value <= hi
    provenance = (sent_in_source and tok_in_sent) or table_prov

    if not in_source:
        grade = "rejected"                               # 원문에 그 숫자 없음 = 환각
    elif not in_range:
        grade = "flagged"                                # 범위 밖 = 추출 오류 의심
    elif not provenance:
        grade = "flagged"                                # 출처(문장·표행) 어디서도 못 잡음 = 손대조 필요
    else:
        grade = "verified"
    return {**base, "raw": raw, "value": value, "label": item.get("label"),
            "변경구분": item.get("변경구분"), "sentence": sentence, "grade": grade,
            "checks": {"in_source": in_source, "sentence_in_source": sent_in_source,
                       "tok_in_sentence": tok_in_sent, "table_provenance": table_prov, "in_range": in_range}}


def verify_extraction(extracted: dict, source_text: str, *, table_rows=None, grids=None) -> dict:
    """추출 dict 전체 검증 → 항목별 등급 + 요약. verified만 표시 채택(호출부)."""
    고시번호, 고시일자 = extracted.get("고시번호"), extracted.get("고시일자")
    results, summary = {}, {"verified": 0, "flagged": 0, "rejected": 0, "missing": 0}
    for attr, item in (extracted.get("attrs") or {}).items():
        v = verify_attr(attr, item, source_text, 고시번호=고시번호, 고시일자=고시일자,
                        table_rows=table_rows, grids=grids)
        results[attr] = v
        summary[v["grade"]] = summary.get(v["grade"], 0) + 1
    return {"zone_name": extracted.get("zone_name"), "고시번호": 고시번호, "고시일자": 고시일자,
            "results": results, "summary": summary}

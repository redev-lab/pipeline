"""gosi_body.py — 고시 본문 텍스트 추출 (v1_3-gosi). 설계: docs/design/gosi_parse.md §6.

역할: 고시 PDF → 정제 텍스트(고시문 추출의 입력). ★소스 추상화 — 지금은 로컬 수동 파일,
나중에 토지이음/정보몽땅 입수기로 교체해도 (경로)→(텍스트) 계약만 지키면 됨.

스캔 이미지(텍스트층 없음)는 빈/짧은 문자열이 나온다 → is_scanned()로 OCR 필요 판정(거짓 추출 방지).
"""
from __future__ import annotations

import re


def read_gosi(path, *, ocr: bool = False) -> dict:
    """PDF → {text, rows, grids, source}. source=digital|ocr|scan.

    디지털: 텍스트+표 셀(grids로 헤더 컬럼 매칭). ★스캔본(텍스트층 0)은 ocr=True면 이미지 OCR로
    텍스트 복원(source='ocr', rows/grids 없음) — ★OCR 값은 자동 verified 금지(verify가 'OCR 검토필요'로
    강등). ocr=False면 source='scan'(빈 텍스트, OCR 미수행).
    """
    import pdfplumber  # PDF 텍스트/표 레이어 파서(스캔 이미지엔 텍스트층이 없어 빈 결과)

    prose, rows, grids = [], [], []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            prose.append(page.extract_text() or "")
            for tbl in (page.extract_tables() or []):
                grid = [[(str(c).replace("\n", " ").strip() if c else "") for c in row] for row in tbl]
                grids.append(grid)                            # 구조 보존(헤더 컬럼 매칭용)
                for row in grid:
                    cells = [c for c in row if c]
                    if cells:
                        rows.append(" | ".join(cells))
    text = _clean("\n".join(prose))
    if rows:
        text += "\n\n[표 셀]\n" + _clean("\n".join(rows))     # LLM·산문대조에도 깨끗한 셀 포함

    if is_scanned(text):                                      # 텍스트층 없음 = 스캔
        if ocr:
            return {"text": _ocr_pdf(path), "rows": [], "grids": [], "source": "ocr"}
        return {"text": text, "rows": [], "grids": [], "source": "scan"}
    return {"text": text, "rows": rows, "grids": grids, "source": "digital"}


def _ocr_pdf(path, *, dpi: int = 300, langs=("ko", "en")) -> str:
    """스캔 PDF → 이미지 렌더(pypdfium2) → OCR(EasyOCR, 한국어) → 텍스트. ★숫자 오인식 위험 — verify가 잠정 처리.

    dpi 300(숫자 가독), CPU. 첫 호출 시 EasyOCR 모델 다운로드(~100MB). 표 구조는 복원 안 함(산문 텍스트).
    """
    import easyocr  # torch 기반 OCR(한국어+영문). 시스템 바이너리 불필요.
    import numpy as np
    import pypdfium2 as pdfium

    reader = easyocr.Reader(list(langs), gpu=False)
    pdf = pdfium.PdfDocument(str(path))
    lines = []
    for i in range(len(pdf)):
        img = pdf[i].render(scale=dpi / 72).to_pil()
        lines.extend(reader.readtext(np.array(img), detail=0, paragraph=True))   # detail=0: 텍스트만
    return _clean("\n".join(lines))


def extract_text(path) -> str:
    """호환 진입점 — read_gosi(path)['text']."""
    return read_gosi(path)["text"]


def _clean(text: str) -> str:
    """널문자·과공백 정리(숫자·% 토큰은 보존). PDF 추출 흔한 \\x00 삽입·다중 공백을 줄인다."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t ]+", " ", text)        # 연속 공백 → 1칸(190 . 0 같은 분리 최소화는 §추출에서)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_scanned(text: str, *, min_chars: int = 200) -> bool:
    """텍스트층 사실상 없음(스캔 이미지) 판정 → OCR 필요. 거짓 추출(빈 본문에 환각) 차단용."""
    return len(text.strip()) < min_chars


# 목표 수치가 모이는 키워드(정비계획 결정 조서/표 근처). 구역명도 추가로 앵커.
_FOCUS_KW = ["용적률", "건폐율", "세대", "연면적", "면적", "획지", "정비계획", "건립", "밀도"]


def focus_text(text: str, *, zone_name: str | None = None, max_chars: int = 14000, window: int = 2) -> str:
    """본문을 목표 수치 주변으로 압축(LLM 프롬프트 축소·정확도↑). 키워드 줄 ±window를 모아 잇는다.

    왜: 고시 1건이 수십 페이지(장위15 32k자)지만 용적률·세대수는 정비계획 결정 조서 일부에 모인다.
    통째로 보내면 느리고(끊김 위험) 타 구역·전문(前文)에 LLM이 헷갈린다. 키워드 윈도로 초점을 좁힌다.
    끊긴 구간은 '…'로 표시(맥락 단절 신호). 키워드 0이면 앞부분 폴백.
    """
    lines = text.split("\n")
    kws = list(_FOCUS_KW) + ([zone_name] if zone_name else [])
    keep = set()
    for i, ln in enumerate(lines):
        if any(k in ln for k in kws):
            keep.update(range(max(0, i - window), min(len(lines), i + window + 1)))
    if not keep:
        return text[:max_chars]
    out, prev = [], None
    for j in sorted(keep):
        if prev is not None and j > prev + 1:
            out.append("…")
        out.append(lines[j]); prev = j
    return "\n".join(out)[:max_chars]

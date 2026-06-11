"""zone_matching.py — 구역명/고시제목 정규화·토큰 매칭 헬퍼 (순수함수).

★canonical 파이프라인 모듈. t-전쟁(2026-06-11, 수검 4/4 통과)에서 검증한 로직을
scripts/ 분석코드에서 승격한 것. zone_boundary(t 마이닝)·공공재개발 매칭·labels가
모두 쓴다. I/O 없는 순수함수 → tests/test_zone_matching.py가 회귀로 고정.

한국 정비구역 이름이 제각각(제6 vs 6, 괄호 '(흑석2,9구역)', 접미사 '구역/지구',
촉진 부모/sub)이라 토큰 추출을 한 곳에서 정의한다.

────────────────────────────────────────────────────────────────────────────
★t-전쟁에서 잡은 함정 6개 (각각 tests/ 회귀 테스트 1개씩 — 한 번 잡은 함정은
테스트가 영구히 지킨다):
  1. [자치구 소스] 결정고시 `고시관리코드[:5]`는 자치구가 아니다(93%가 시레벨
     11000). 한 구역 이력이 11000(옛)/11290(신)으로 쪼개짐 → 자치구 가드는
     **동명→구 매핑(지적도)**으로. 고시코드 금지. (zone_boundary에서 적용)
  2. ['제' 삼킴] 그리디 `[가-힣]{2,4}`가 '제'를 먹어 '돈암제6'→'돈암제6'. →
     비그리디 `[가-힣]{2,4}?`로 '돈암6'. (normalize_zone_name)
  3. [재건축 누출] '정릉3 주택재건축 정비구역지정'이 '정비구역'으로 통과 →
     재개발 토큰 오매칭. → is_redev_title에서 '재건축' 배제.
  4. [인코딩] cp949 깨짐 '수색?증산'·middle-dot '수색·증산' → clean_text가
     '수색증산'으로 일반 정규화.
  5. [후속단계 누수] 길음1=1998 '시행인가'가 지정으로 오인. → 시행인가·관리처분
     등 후속단계 배제(is_redev_title).
  6. [멀티사이클] 흑석2 1985 개량재개발→1986 관리처분(완료)→2025 새 촉진. earliest가
     완료된 옛 사이클로 과도하게 내려감 → 완료고시(cycle_done) 이후만 현재 사이클
     (zone_boundary의 OA 인덱스에서 적용).
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import re

# 정비구역 관련 결정만 거르는 키워드 (지구단위계획·도시계획시설 등 무관 결정 배제).
# '재정비촉진' 포함: 촉진지구 지정·촉진계획 결정도 재개발 결정으로 인정(촉진 부모 회수).
REDEV_TITLE_KW = ("재개발", "정비구역", "정비계획", "정비사업", "재정비촉진")

# 재정비촉진 부모지구명 (sub-구역 t 근사용).
PROMOTION_PARENTS = ("수색증산", "흑석", "노량진", "길음", "장위", "한남", "이문휘경",
                     "신정", "방화", "거여마천", "북아현", "미아", "상계")

# 후속단계 고시(지정 아님) — 최초 지정고시 마이닝에서 배제 (함정 5).
_LATER_STAGE_KW = ("시행인가", "사업시행계획", "관리처분", "준공", "착공", "이전고시")

# 사이클 완료 고시 — 이게 있으면 그 *이전* 지정은 완료된 옛 사이클 (함정 6).
CYCLE_DONE_KW = ("관리처분", "준공", "이전고시", "청산")

# 연결자·cp949 깨짐 문자(함정 4) — 한글 사이 연결자 제거로 동일 구역명 통일.
_CONNECTORS = "·・･‧•∙ㆍ"


def clean_text(s) -> str:
    """제목/구역명에서 연결자·인코딩 깨짐 문자 제거 (정규화 1단계, 함정 4).

    '수색·증산' = '수색?증산'(cp949 깨짐) = '수색증산'.
    """
    s = str(s)
    s = re.sub(f"[{_CONNECTORS}]", "", s)            # middle-dot 계열 제거
    s = re.sub(r"(?<=[가-힣])\?(?=[가-힣])", "", s)   # 한글 사이 cp949 깨짐 ? 제거
    return s


def cycle_done(title) -> bool:
    """제목이 사이클 완료(관리처분/준공/이전고시) 고시인가 (함정 6 가드용)."""
    return any(k in str(title) for k in CYCLE_DONE_KW)


def is_redev_title(title) -> bool:
    """제목이 *재개발 정비구역 지정/계획 결정*인가 (지정 마이닝용 필터).

    배제: 재건축(함정 3, 재개발 토큰 오매칭) + 후속단계 시행인가·관리처분 등
    (함정 5, 지정이 아니라 지정 후 단계).
    """
    s = str(title)
    if "재건축" in s:
        return False
    if any(k in s for k in _LATER_STAGE_KW):
        return False
    return any(k in s for k in REDEV_TITLE_KW)


def normalize_zone_name(title) -> set:
    """제목/구역명 → 구역 토큰 set ('지역명+번호').

    추출: 괄호 안 'XX[제]N[,M]구역'('(흑석2,9구역)'→{흑석2,흑석9}) + 선두 'XX[제]N'
    ('돈암제6 주택재개발'→{돈암6}). 정규화: 제N→N, 공백 제거(함정 2). set(여러 구역 참조).
    """
    s = clean_text(title).replace(" ", "")
    toks = set()
    # ([가-힣]{2,4}?) 비그리디 — '제'를 한글부가 삼키지 않게(함정 2: 돈암제6→돈암6).
    for grp in re.findall(r"[\(（]([^\)）]*)[\)）]", s):
        for m in re.finditer(r"([가-힣]{2,4}?)제?(\d+)(?:,(\d+))?구역", grp):
            toks.add(m.group(1) + m.group(2))
            if m.group(3):
                toks.add(m.group(1) + m.group(3))
    m = re.match(r"([가-힣]{2,4}?)제?(\d+)", s)
    if m:
        toks.add(m.group(1) + m.group(2))
    return toks


def region_of(token) -> str:
    """토큰('돈암6'/'응암동700')에서 지역명('돈암'/'응암')만 분리 — 동명→구 가드용.

    trailing '동'/'가' strip: 지번형 토큰('응암동700')의 지역이 동맵 키('응암')와
    어긋나는 문제 해결(단 2자 미만으로 줄면 보존).
    """
    m = re.match(r"([가-힣]+)", str(token))
    r = m.group(1) if m else ""
    stripped = re.sub(r"(동|가)$", "", r)
    return stripped if len(stripped) >= 2 else r


def parent_of(title) -> str | None:
    """제목이 재정비촉진 sub-구역이면 부모 촉진지구명 반환, 아니면 None."""
    s = clean_text(title)
    if "재정비촉진" not in s:
        return None
    for p in PROMOTION_PARENTS:
        if p in s:
            return p
    return None

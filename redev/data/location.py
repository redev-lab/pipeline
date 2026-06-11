"""location.py — 위치(지번) 문자열 파서 공용 헬퍼 (소스 무관).

해제구역 `위치`("상도2동 159-1"), 공공재개발 `위치_지번` 등 사람이 적은 지번
문자열을 (법정동, 본번, 부번, 산여부)로 정규화한다. parcels의 (시군구,동,본번,부번)
→PNU 인덱스 조회에 바로 쓴다.

★처리하는 변형(실데이터에서 잡힌 미스 패턴):
  공백없음('정릉동170-1') · 일대/일원 접미 · 산('상도동 산65') · 행정동≠법정동
  ('상도2동'→'상도동'). 자세한 근거: docs/design/ingest_cancelled.md §2.
"""
from __future__ import annotations

import re


def parse_location(loc) -> tuple[str, int, int, bool] | None:
    """위치 문자열 → (동, 본번, 부번, is_산). 파싱 불가 시 None.

    반환 `동`은 *원형*(행정동일 수 있음). 행정동→법정동 변환은 호출부가 2단
    조회로 처리한다(원형 먼저 → 실패 시 admin_to_legal_dong). 법정동에 숫자가
    정당하게 든 케이스(안암동2가)를 휴리스틱이 깨먹지 않게 하기 위함.
    """
    s = str(loc).strip()
    if not s:
        return None
    s = re.sub(r"\s*(일대|일원)\s*$", "", s)   # 접미 strip
    s = s.replace(" ", "")                      # 공백없음 정규화
    # 끝의 (산?)본번[-부번]
    m = re.search(r"(산)?(\d+)(?:-(\d+))?$", s)
    if not m:
        return None
    dong = s[: m.start()]
    if not dong or dong[-1] not in "동가":       # 동/가로 끝나야 동명
        return None
    return dong, int(m.group(2)), int(m.group(3) or 0), m.group(1) == "산"


def admin_to_legal_dong(dong: str) -> str:
    """행정동→법정동 근사: 동 *앞* 숫자만 제거 ('상도2동'→'상도동').

    동 *뒤* 숫자/가(법정동 '안암동2가'·'삼선동1가')는 건드리지 않는다 — 숫자
    위치로 행정동/법정동을 구분(매핑표 없는 v1 휴리스틱, docs §2).
    """
    return re.sub(r"(\d+)동$", "동", dong)

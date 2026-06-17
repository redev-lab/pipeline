"""gosi_index.py — 고시목록 인덱스(공공데이터 A축). 설계: docs/design/gosi_index.md.

국토교통부 토지이음 고시목록(data.go.kr 15083101, 상업가능·무료)에서 고시번호→공식 URL·일자·제목을
인덱싱한다. 본문은 이 URL로 ★사람이 수동 다운로드(C, ToS 안전). 자동화의 합법 1축.

★외부접속은 사용자 로컬 실행(Claude 환경 차단) — CSV는 무키 즉시, API는 serviceKey(.env) 로컬.
"""
from __future__ import annotations

import re

import pandas as pd

# CSV 컬럼 후보(소스 표기 흔들림 흡수). 4컬럼: 고시번호·고시일자·접속링크주소(URL)·고시명.
_COLS = {"고시번호": ["고시번호", "고시 번호"], "고시일자": ["고시일자", "고시일"],
         "url": ["접속링크주소(URL)", "접속링크주소", "URL", "접속경로", "url"], "고시명": ["고시명", "제목"]}


def _gosi_no(s) -> str | None:
    """문자열에서 'YYYY-NNN' 고시번호 추출(서울특별시 고시 제2025-179호 → 2025-179)."""
    m = re.search(r"\d{4}-\d+", str(s or ""))
    return m.group(0) if m else None


def _pick_col(df: pd.DataFrame, names) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def load_index(csv_path) -> pd.DataFrame:
    """로컬 토지이음 고시목록 CSV → 정규화 인덱스 [고시번호(YYYY-NNN), 고시일자, url, 고시명].

    ★전국 데이터 — 고시번호는 기관별로 중복(2024-448이 전국 13건). dedup 안 함; resolve가 (번호+일자+명)
    으로 서울 고시를 가려낸다. 무키·즉시(무로그인 다운로드 CSV). cp949/utf-8 자동.
    """
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            df = pd.read_csv(csv_path, encoding=enc, dtype=str)
            break
        except UnicodeDecodeError:
            df = None
    if df is None:
        raise ValueError(f"CSV 인코딩 해독 실패: {csv_path}")
    out = pd.DataFrame()
    for key, names in _COLS.items():
        col = _pick_col(df, names)
        out[key] = df[col] if col else None
    out["고시번호"] = out["고시번호"].map(_gosi_no)        # YYYY-NNN 표준화
    out["고시일자"] = out["고시일자"].map(lambda s: re.sub(r"[^\d]", "", str(s or ""))[:8])  # YYYYMMDD
    return out.dropna(subset=["고시번호"])                  # ★dedup 금지(전국 중복)


def _name_core(s) -> str:
    """구역명/고시명에서 매칭 토큰 추출 — '불광제5'→'불광5', '고척동253번지'→'고척동253', '장위15구역'→'장위15'.
    한글(2+)+옵션'제'+숫자 첫 출현. 표기 흔들림(제N·동N번지·N구역) 흡수."""
    m = re.search(r"([가-힣]{2,})제?\s*(\d+)\s*구역", str(s or "")) or re.search(r"([가-힣]{2,})제?\s*(\d+)", str(s or ""))
    return (m.group(1) + m.group(2)) if m else ""


def resolve_urls(targets, index: pd.DataFrame) -> list:
    """★(고시번호+고시일자+구역명) 3중 매칭으로 정확한 서울 고시 URL 해소(전국 번호중복 차단).

    targets = [{"고시번호","고시일자"(옵션),"구역명"(옵션)}]. 번호+일자로 좁히고, ★구역명 토큰이 고시명에
    일치할 때만 URL 반환(고신뢰). 구역명 줬는데 이름 불일치 = ★미발견(엉뚱한 지역 URL 반환 금지 —
    오매칭보다 무발견). 구역명 없으면 번호+일자 유일행만 채택(중신뢰). 반환 [{...url, matched_title, conf}].
    """
    out = []
    for t in targets:
        no, dt = _gosi_no(t.get("고시번호")), re.sub(r"[^\d]", "", str(t.get("고시일자") or ""))[:8]
        core = _name_core(t.get("구역명") or "")
        c = index[index["고시번호"] == no]
        if dt and len(c[c["고시일자"] == dt]):
            c = c[c["고시일자"] == dt]
        url, mt, conf = None, None, "미발견"
        if core:                                          # 구역명 주면 ★이름 일치할 때만(오매칭 차단)
            nm = c[c["고시명"].fillna("").map(lambda s: _name_core(s) == core)]
            if len(nm):
                r = nm.iloc[0]; url, mt, conf = r["url"], r["고시명"], "고"
        elif len(c) == 1:                                 # 구역명 없으면 번호+일자 유일행만
            r = c.iloc[0]; url, mt, conf = r["url"], r["고시명"], "중"
        out.append({"고시번호": no, "고시일자": t.get("고시일자"), "구역명": t.get("구역명"),
                    "url": url, "matched_title": mt, "conf": conf})
    return out


def fetch_index_api(*, service_key: str, endpoint: str, num_rows: int = 1000,
                    max_pages: int = 500, extra: dict | None = None, sleep_s: float = 0.3) -> pd.DataFrame:
    """data.go.kr 표준 OpenAPI로 고시목록 인덱스 적재(★사용자 로컬 실행 — 외부접속).

    endpoint = 활용신청 후 표기되는 요청주소(operation 포함), service_key = .env DATA_GO_KR_KEY(Encoding 형 그대로).
    표준 응답: {response:{body:{items:{item:[...]}, totalCount}}}. 필드명이 다르면 호출부에서 _COLS 조정.
    rate limit 절제(sleep_s 간격). 반환은 load_index와 동일 정규화 스키마.
    """
    import json
    import time
    import urllib.parse
    import urllib.request

    rows = []
    for page in range(1, max_pages + 1):
        q = {"serviceKey": service_key, "pageNo": page, "numOfRows": num_rows, "type": "json", **(extra or {})}
        url = endpoint + ("&" if "?" in endpoint else "?") + urllib.parse.urlencode(q, safe="%")  # 키 % 보존
        with urllib.request.urlopen(url, timeout=30) as r:            # ★로컬에서만 동작(외부접속)
            data = json.loads(r.read().decode("utf-8"))
        body = (((data or {}).get("response") or {}).get("body") or {})
        items = (body.get("items") or {})
        items = items.get("item", items) if isinstance(items, dict) else items
        if not items:
            break
        rows.extend(items if isinstance(items, list) else [items])
        if len(rows) >= int(body.get("totalCount") or 0):
            break
        time.sleep(sleep_s)
    df = pd.DataFrame(rows)
    out = pd.DataFrame()
    for key, names in _COLS.items():
        col = _pick_col(df, names)
        out[key] = df[col] if col else None
    out["고시번호"] = out["고시번호"].map(_gosi_no)
    return out.dropna(subset=["고시번호"]).drop_duplicates("고시번호")

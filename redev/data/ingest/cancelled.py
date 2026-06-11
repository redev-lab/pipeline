"""cancelled.py — 해제구역 적재 (reliable negative, R5 핵심).

역할: 해제 CSV의 `위치(지번)`을 parcels jibun index로 PNU 해석해 reliable_negative
행을 만든다. "요건은 됐으나 무산된" 최강 네거티브. (R5: 같은 PNU가 이후 t에
positive로 재등장 가능.) 설계: docs/design/ingest_cancelled.md

WHERE = seed(대표지번) only — 해제 CSV는 폴리곤이 없어 구역당 1필지(labels.md §4.6).
"""
from __future__ import annotations

import pandas as pd

from redev.config import training_districts
from redev.data.location import admin_to_legal_dong, parse_location


def load_cancelled(csv_path: str, jibun_index: dict) -> tuple[pd.DataFrame, dict]:
    """해제 CSV → reliable_negative 행 + 매칭 리포트.

    입력:
      csv_path : 해제구역 CSV (자치구·구역명·위치·해제일자…).
      jibun_index : parcels.build_jibun_index() 결과 (시군구,동,본번,부번)→PNU.
    출력:
      df : [pnu, t(연도), t_date, t_source, label, certainty, source, neg_reason,
            zone_id] reliable_negative 행.
      report : 파싱·매칭 위생(파싱실패·지번미존재·strip 회수 수 등).

    ★2단 조회: 원형 동으로 먼저 → 실패 시 admin_to_legal_dong(행정동→법정동)으로
    재조회. 법정동에 정당한 숫자(안암동2가)를 휴리스틱이 깨먹지 않게.
    """
    name2code = {d["name"]: d["sigungu_code"] for d in training_districts()}
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)

    rows = []
    n_total = n_out_scope = n_parse_fail = n_jibun_miss = n_strip_recovered = 0
    for _, r in df.iterrows():
        gu = str(r.get("자치구", "")).strip()
        if gu not in name2code:
            n_out_scope += 1
            continue
        n_total += 1
        sig = name2code[gu]
        parsed = parse_location(r.get("위치"))
        if parsed is None:
            n_parse_fail += 1
            continue
        dong, bonbun, bubun, _is_san = parsed

        # ① 원형 동 조회
        pnu = jibun_index.get((sig, dong, bonbun, bubun))
        # ② 실패 시 행정동→법정동 strip 후 재조회 (회수 집계)
        if pnu is None:
            legal = admin_to_legal_dong(dong)
            if legal != dong:
                pnu = jibun_index.get((sig, legal, bonbun, bubun))
                if pnu is not None:
                    n_strip_recovered += 1
        if pnu is None:
            n_jibun_miss += 1
            continue

        t_date = str(r.get("해제일자", ""))[:10]
        rows.append({
            "pnu": pnu,
            "t": int(t_date[:4]) if t_date[:4].isdigit() else pd.NA,
            "t_date": t_date,
            "t_source": "cancel_date",
            "label": 0,
            "certainty": "reliable_negative",
            "source": "해제",
            "neg_reason": "cancelled",
            "zone_id": str(r.get("구역명", "")).strip(),
        })

    out = pd.DataFrame(rows)
    matched = len(out)
    report = {
        "in_scope_rows": n_total,
        "out_of_scope_rows": n_out_scope,
        "matched_pnu": matched,
        "match_rate": round(matched / n_total, 3) if n_total else 0.0,
        "parse_fail": n_parse_fail,
        "jibun_miss": n_jibun_miss,
        "recovered_by_admin_strip": n_strip_recovered,  # ★휴리스틱 실효
    }
    return out, report

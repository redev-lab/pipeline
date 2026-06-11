"""zone_boundary.py — 의제처리구역 → positive ZoneTable (최초 지정일 t).

역할: 의제처리(UQ181) 폴리곤에 **최초 정비구역 지정일(t)**을 붙여 positive 구역
테이블을 만든다. WHERE=폴리곤(경계), WHEN=OA-20283(결정고시) 이력 마이닝.
labels.py가 이 ZoneTable을 parcels와 ∩ 해서 positive 필지 라벨을 만든다.

t-해소는 t-전쟁(수검 4/4)에서 검증한 로직: OA 이력 마이닝(사이클 인지) + 6소스
earliest-genuine + t_alt 보존. 설계: docs/design/ingest_zone_boundary.md.
★소스 추상화(§11): 출력 ZoneTable은 소스 무관 → 상업 전환 시 의제처리만 교체.
"""
from __future__ import annotations

from collections import defaultdict

import geopandas as gpd
import pandas as pd
from pyogrio import read_dataframe

from redev.data.geo import to_target_crs
from redev.data.zone_matching import (
    cycle_done,
    is_redev_title,
    normalize_zone_name,
    parent_of,
    PROMOTION_PARENTS,
)

# 재개발 사업유형(소분류): 주택정비형(UQ1221) + 도시정비형(UQ1222). 학습엔 둘 다 positive.
REDEV_SCLAS = ("UQ1221", "UQ1222")


def _ndate(d) -> str:
    """정렬용 날짜 정규화: 연도만('2008')이면 '2008-01-01'로 패딩."""
    d = str(d)[:10]
    return d if len(d) >= 10 else d[:4] + "-01-01"


def build_dong_to_sigungu(parcels: pd.DataFrame) -> dict:
    """지적도 dong_addr → {동base: set(시군구코드)} (자치구 가드).

    ★고시관리코드[:5]는 자치구가 아니다(함정1, 93%가 시레벨 11000) → 지적도
    동명으로 자치구를 판정한다. 동base = 동명에서 동/숫자/가 앞 지역명만.
    """
    import re
    base2sig: dict = defaultdict(set)
    for addr, sig in zip(parcels["dong_addr"], parcels["sigungu"]):
        if not isinstance(addr, str):
            continue
        last = addr.split()[-1]
        m = re.match(r"([가-힣]+?)(?:동|\d|가|로|길|$)", last)
        if m:
            base2sig[m.group(1)].add(sig)
    return dict(base2sig)


def build_oa_token_index(gosi: pd.DataFrame, dong_map: dict) -> dict:
    """결정고시(OA-20283) → {(시군구,토큰): (최초지정일, 제목)} — ★사이클 인지.

    멀티사이클 가드(함정6): 한 토큰의 이력에 완료고시(관리처분/준공)가 있으면 그
    *이후* 의 최초 지정결정만 현재 사이클로 본다(흑석2: 1985 옛 사이클 배제→2025/2008).

    입력 gosi 컬럼: 제목, 고시일자(또는 d=YYYY-MM-DD), 고시유형.
    """
    from redev.data.zone_matching import region_of
    isdec = gosi["고시유형"].astype(str).str.contains("결정", na=False)
    d = gosi["고시일자"].astype(str).str[:10]
    ev: dict = defaultdict(list)  # (sig,tok) -> [(date, is_redev, is_done, title)]
    for title, date, dec in zip(gosi["제목"], d, isdec):
        if not dec:
            continue
        rd, dn = is_redev_title(title), cycle_done(title)
        if not (rd or dn):
            continue
        for tok in normalize_zone_name(title):
            for sig in dong_map.get(region_of(tok), ()):
                ev[(sig, tok)].append((date, rd, dn, title))
    tm: dict = {}
    for key, lst in ev.items():
        dones = [x[0] for x in lst if x[2]]
        last_done = max(dones) if dones else None
        reds = [(x[0], x[3]) for x in lst if x[1] and (last_done is None or x[0] > last_done)]
        if reds:
            tm[key] = min(reds, key=lambda x: x[0])
    return tm


def build_promotion_parent_dates(gosi: pd.DataFrame) -> dict:
    """재정비촉진 부모지구별 최초 결정일 {부모: (date, 제목)} (촉진 sub의 근사 t)."""
    from redev.data.zone_matching import clean_text
    isdec = gosi["고시유형"].astype(str).str.contains("결정", na=False)
    sub = gosi[isdec & gosi["제목"].map(is_redev_title)].copy()
    sub["d"] = sub["고시일자"].astype(str).str[:10]
    pmin: dict = {}
    for p in PROMOTION_PARENTS:
        hit = sub[sub["제목"].map(lambda x: p in clean_text(x) and "재정비촉진" in clean_text(x))]
        if len(hit):
            r = hit.sort_values("d").iloc[0]
            pmin[p] = (r["d"], r["제목"])
    return pmin


def resolve_zone_t(title, sig, ntfc_date, *, oa_idx, pmin, xsrc):
    """한 폴리곤의 t 해소 — 6소스 earliest-genuine + t_alt 보존.

    후보: ①OA(사이클인지) ②촉진부모 ③ntfc_direct(제목 '지정'&¬'변경') ④정비사업
    ⑤신통 ⑥공공재개발. 전 소스 통틀어 **최소 날짜 = t**, 나머지 = t_alt.
    "변경"뿐(후보 0)이면 (None,None,None) — 보류(추정 금지).

    xsrc = {"jbk":{(sig,tok):date}, "shk":{(sig,tok):date}, "pub":[(sig,구역명,date)]}
    """
    cand = []  # (date, source)
    cs = [oa_idx[(sig, t)] for t in normalize_zone_name(title) if (sig, t) in oa_idx]
    if cs:
        cand.append((min(cs, key=lambda x: x[0])[0], "oa_first_decision"))
    p = parent_of(title)
    if p and p in pmin:
        cand.append((pmin[p][0], "promotion_parent"))
    if "지정" in str(title) and "변경" not in str(title):
        cand.append((ntfc_date, "ntfc_direct"))
    for t in normalize_zone_name(title):
        if (sig, t) in xsrc["jbk"]:
            cand.append((xsrc["jbk"][(sig, t)], "first_designation_gosi"))
        if (sig, t) in xsrc["shk"]:
            cand.append((xsrc["shk"][(sig, t)], "shintong_select"))
    from redev.data.zone_matching import clean_text
    norm_title = clean_text(title).replace(" ", "")
    for psig, pname, pdate in xsrc["pub"]:
        if psig == sig and pname.replace("일대", "") in norm_title:
            cand.append((pdate, "public_redev"))
    if not cand:
        return None, None, None
    cand.sort(key=lambda x: _ndate(x[0]))
    chosen = cand[0]
    alt = sorted({c[0] for c in cand[1:] if _ndate(c[0]) != _ndate(chosen[0])})
    return chosen[0], chosen[1], (";".join(alt) if alt else None)


def load_zones(
    uq_shp: str,
    gosi_csv: str,
    parcels: pd.DataFrame,
    sigungu_codes,
    *,
    jeonbisaeop_csv: str | None = None,
    shintong_csv: str | None = None,
    public_redev_csv: str | None = None,
) -> tuple[gpd.GeoDataFrame, dict]:
    """의제처리 폴리곤 + OA t → positive ZoneTable + 리포트.

    출력 ZoneTable 컬럼: zone_id(NTFC_SN)·geometry(5186)·t(연도)·t_date·t_source·
    t_alt·zone_type(UQ1221/UQ1222)·sigungu·source. 보류는 dropped 리포트로.
    """
    codes = set(sigungu_codes)
    # 폴리곤(2097→5186) + 재개발·자치구 필터
    zones = gpd.read_file(
        uq_shp, columns=["SIGNGU_SE", "SCLAS_CL", "NTFC_SN"], encoding="cp949"
    )
    zones = zones[zones["SIGNGU_SE"].astype(str).isin(codes)
                  & zones["SCLAS_CL"].astype(str).isin(REDEV_SCLAS)].copy()
    zones = to_target_crs(zones)   # 좌표계 닻(지뢰1)

    gosi = pd.read_csv(gosi_csv, encoding="cp949", dtype=str)
    dong_map = build_dong_to_sigungu(parcels)
    oa_idx = build_oa_token_index(gosi, dong_map)
    pmin = build_promotion_parent_dates(gosi)
    gd = gosi.drop_duplicates("고시관리코드").set_index("고시관리코드")
    zones["title"] = zones["NTFC_SN"].map(gd["제목"])
    zones["ntfc_date"] = zones["NTFC_SN"].map(gd["고시일자"]).astype(str).str[:10]

    # 교차검증 소스(있으면). CSV의 자치구명→코드 매핑(config).
    from redev.config import training_districts
    name2code = {d["name"]: d["sigungu_code"] for d in training_districts()}
    xsrc = {"jbk": {}, "shk": {}, "pub": []}
    if jeonbisaeop_csv:
        jb = pd.read_csv(jeonbisaeop_csv, encoding="cp949", dtype=str)
        jb = jb[jb["시군구명"].isin(name2code) & jb["사업시행방식"].astype(str).str.contains("재개발", na=False)]
        for _, r in jb.iterrows():
            for t in normalize_zone_name(r["정비 구역명"]):
                xsrc["jbk"][(name2code[r["시군구명"]], t)] = str(r["고시일"])[:10]
    if shintong_csv:
        sh = pd.read_csv(shintong_csv, encoding="utf-8-sig", dtype=str)
        sh = sh[sh["자치구"].isin(name2code)]
        for _, r in sh.iterrows():
            if str(r.get("신통선정일", "")).strip():
                for t in normalize_zone_name(r["구역명"]):
                    xsrc["shk"][(name2code[r["자치구"]], t)] = str(r["신통선정일"])[:10]
    if public_redev_csv:
        pb = pd.read_csv(public_redev_csv, encoding="utf-8-sig", dtype=str)
        for _, r in pb.iterrows():
            if str(r.get("t", "")).lower() not in ("nan", "") and r["자치구"] in name2code:
                xsrc["pub"].append((name2code[r["자치구"]], str(r["구역명"]), str(r["t"])[:10]))

    # 폴리곤별 t 해소
    out_rows = []
    n_hold = 0
    for _, r in zones.iterrows():
        t, tsrc, talt = resolve_zone_t(
            str(r["title"]), r["SIGNGU_SE"], r["ntfc_date"],
            oa_idx=oa_idx, pmin=pmin, xsrc=xsrc,
        )
        if t is None:
            n_hold += 1
            continue
        out_rows.append({
            "zone_id": r["NTFC_SN"],
            "geometry": r.geometry,
            "t": int(str(t)[:4]) if str(t)[:4].isdigit() else pd.NA,
            "t_date": str(t)[:10],
            "t_source": tsrc,
            "t_alt": talt,
            "zone_type": r["SCLAS_CL"],
            "sigungu": r["SIGNGU_SE"],
            "source": "의제처리",
        })
    zt = gpd.GeoDataFrame(out_rows, geometry="geometry", crs=zones.crs)
    # 멀티폴리곤(같은 NTFC_SN 여러 행)은 dissolve로 1구역 1행
    if len(zt):
        zt = zt.dissolve(by="zone_id", aggfunc="first").reset_index()
    report = {
        "total_polygons": len(zones),
        "resolved_zones": int(zt["zone_id"].nunique()) if len(zt) else 0,
        "held_no_clean_t": n_hold,
        "t_source_counts": zt["t_source"].value_counts().to_dict() if len(zt) else {},
        "with_t_alt": int(zt["t_alt"].notna().sum()) if len(zt) else 0,
    }
    return zt, report

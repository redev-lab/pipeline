"""transactions.py — 국토부 실거래가 적재 (심장2 AVM 입력, R6·R16). 설계: avm.md §3.

역할: 연립다세대(대지지분 평당가 학습 타깃, R6)·아파트(비교신축 시세 reference, R17) 매매
실거래를 API로 받아 PNU 매핑된 정제 거래 테이블로. 지번→PNU는 location.py 재사용
(cancelled.py에서 검증된 파서). 호출=(시군구,연월) 단위 + XML 디스크 캐시(일일 한도·재호출 회피).

★API 키는 Encoding 형태(%포함)라 raw 전달(urlencode 재인코딩 시 401). .env에서 os.getenv.
"""
from __future__ import annotations

import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from redev.data.location import admin_to_legal_dong, parse_location

_CACHE = Path("_data/cache/trades")
# (엔드포인트, .env 키명) — 2026-06-12 실호출 검증(avm.md §3).
_ENDPOINTS = {
    "villa": ("http://apis.data.go.kr/1613000/RTMSDataSvcRHTrade/getRTMSDataSvcRHTrade", "VILLA_TRADE_API_KEY"),
    "apt": ("http://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade", "APT_TRADE_API_KEY"),
}
TRADE_COLUMNS = ["pnu", "trade_type", "deal_amount", "area_m2", "land_share_m2",
                 "build_year", "deal_ym", "sigungu"]


def _num(s):
    try:
        return float(str(s).replace(",", "")) if str(s).strip() not in ("", "-") else None
    except ValueError:
        return None


def _fetch_xml(kind: str, lawd_cd: str, ym: str, *, use_cache: bool = True) -> str:
    """(kind, 시군구5자리, 연월YYYYMM) 1회 호출 → XML. 캐시 우선(재호출·한도 회피)."""
    cache = _CACHE / f"{kind}_{lawd_cd}_{ym}.xml"
    if use_cache and cache.exists():
        return cache.read_text(encoding="utf-8")
    url, env_key = _ENDPOINTS[kind]
    key = os.getenv(env_key)
    if not key:
        raise RuntimeError(f"{env_key} 미설정 — .env 확인")
    other = urllib.parse.urlencode({"LAWD_CD": lawd_cd, "DEAL_YMD": ym, "numOfRows": "1000", "pageNo": "1"})
    raw = urllib.request.urlopen(f"{url}?serviceKey={key}&{other}", timeout=30).read().decode("utf-8")  # ★raw 키
    _CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(raw, encoding="utf-8")
    return raw


def _parse_items(raw: str, kind: str) -> list[dict]:
    """XML → item dict 목록. cdealType='O'(계약해제) 제외."""
    root = ET.fromstring(raw)
    items = []
    for it in root.findall(".//item"):
        g = lambda t: (it.findtext(t) or "").strip()
        if g("cdealType") == "O":
            continue
        items.append({
            "umd": g("umdNm"), "jibun": g("jibun"), "sgg": g("sggCd"),
            "amount": g("dealAmount"), "exclu": g("excluUseAr"), "land": g("landAr"),
            "year": g("buildYear"), "dy": g("dealYear"), "dm": g("dealMonth"), "kind": kind,
        })
    return items


def _to_pnu(umd: str, jibun: str, sgg: str, jibun_index: dict):
    """umdNm+jibun → PNU. location.parse_location 재사용 + 행정동 폴백(cancelled.py 동일)."""
    parsed = parse_location(f"{umd}{jibun}")
    if parsed is None:
        return None
    dong, bon, bu, _san = parsed
    pnu = jibun_index.get((sgg, dong, bon, bu))
    if pnu is None:
        legal = admin_to_legal_dong(dong)
        if legal != dong:
            pnu = jibun_index.get((sgg, legal, bon, bu))
    return pnu


def load_transactions(jibun_index: dict, *, sigungu_codes, months, kinds=("villa", "apt"),
                      use_cache: bool = True) -> tuple[pd.DataFrame, dict]:
    """API → 정제 거래 테이블 + 위생 리포트. PNU 매핑 실패율·대지권 결측율 기록(R6 수검).

    입력: jibun_index(parcels.build_jibun_index), sigungu_codes(4구), months(['202301',...]).
    출력: df(TRADE_COLUMNS) + report(api건수·매핑실패율·대지권결측율 등).
    """
    rows = []
    n_api = n_pnu_miss = n_amt_fail = n_landar_miss = n_villa = 0
    for kind in kinds:
        for lawd in sigungu_codes:
            for ym in months:
                items = _parse_items(_fetch_xml(kind, lawd, ym, use_cache=use_cache), kind)
                n_api += len(items)
                for d in items:
                    pnu = _to_pnu(d["umd"], d["jibun"], d["sgg"], jibun_index)
                    if pnu is None:
                        n_pnu_miss += 1
                        continue
                    amt = d["amount"].replace(",", "")
                    if not amt.isdigit():
                        n_amt_fail += 1
                        continue
                    land = _num(d["land"])
                    if kind == "villa":
                        n_villa += 1
                        if land is None:
                            n_landar_miss += 1
                    rows.append({
                        "pnu": pnu, "trade_type": kind, "deal_amount": int(amt),
                        "area_m2": _num(d["exclu"]), "land_share_m2": land,
                        "build_year": int(d["year"]) if d["year"].isdigit() else pd.NA,
                        "deal_ym": f"{d['dy']}{int(d['dm']):02d}" if d["dm"].isdigit() else pd.NA,
                        "sigungu": d["sgg"],
                    })
    df = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    matched = len(df)
    report = {
        "api_rows": n_api, "matched": matched,
        "pnu_miss": n_pnu_miss, "pnu_match_rate": round(matched / n_api, 3) if n_api else 0.0,
        "amount_fail": n_amt_fail,
        "villa_rows": n_villa, "villa_landar_miss": n_landar_miss,
        "villa_landar_miss_rate": round(n_landar_miss / n_villa, 3) if n_villa else 0.0,
        "by_type": df["trade_type"].value_counts().to_dict() if matched else {},
    }
    return df, report

"""gosi_index 회귀 — 고시목록 CSV 파싱·번호 정규화·URL 매니페스트 (순수, 네트워크 없음).

실행: python -m pytest tests/test_gosi_index.py
"""
import pandas as pd

from redev.data.ingest.gosi_index import _gosi_no, _name_core, load_index, resolve_urls


def test_gosi_no_extracts_yyyy_nnn():
    assert _gosi_no("서울특별시 고시 제2025-179호") == "2025-179"
    assert _gosi_no("2025-179") == "2025-179"
    assert _gosi_no("제목만 있고 번호 없음") is None


def test_name_core_absorbs_variants():
    assert _name_core("불광제5주택정비형 재개발사업") == "불광5"
    assert _name_core("고척동253번지일대 주택정비형") == "고척동253"
    assert _name_core("재정비촉진계획(장위15구역) 변경") == "장위15"


def test_load_index_keeps_national_dupes(tmp_path):
    csv = tmp_path / "idx.csv"
    pd.DataFrame({  # 같은 고시번호 2025-179가 서울·화성 2건(전국 중복) — dedup 안 함
        "고시번호": ["서울특별시 고시 제2025-179호", "화성시 고시 제2025-179호", "번호없음"],
        "고시일": ["2025-04-03", "2025-04-03", "2024-01-01"],
        "접속링크주소(URL)": ["http://eum/seoul", "http://eum/hwaseong", "http://x"],
        "고시명": ["상도14구역 정비계획", "백노동지구 도로", "기타"],
    }).to_csv(csv, index=False, encoding="utf-8-sig")
    idx = load_index(csv)
    assert (idx["고시번호"] == "2025-179").sum() == 2          # 중복 보존(가려냄은 resolve)
    assert idx["고시일자"].iloc[0] == "20250403"               # 일자 정규화


def test_resolve_strict_name_match_no_wrong_fallback(tmp_path):
    csv = tmp_path / "idx.csv"
    pd.DataFrame({
        "고시번호": ["서울특별시 고시 제2025-179호", "화성시 고시 제2025-179호"],
        "고시일": ["2025-04-03", "2025-04-03"],
        "접속링크주소(URL)": ["http://eum/seoul", "http://eum/hwaseong"],
        "고시명": ["상도14구역 주택정비형 재개발사업 정비계획", "화성 백노동지구 도로"],
    }).to_csv(csv, index=False, encoding="utf-8-sig")
    idx = load_index(csv)
    r = resolve_urls([{"고시번호": "2025-179", "고시일자": "2025-04-03", "구역명": "상도14구역"}], idx)
    assert r[0]["url"] == "http://eum/seoul" and r[0]["conf"] == "고"      # 이름으로 서울 선택(화성 아님)
    bad = resolve_urls([{"고시번호": "2025-179", "고시일자": "2025-04-03", "구역명": "없는99구역"}], idx)
    assert bad[0]["url"] is None and bad[0]["conf"] == "미발견"            # ★오매칭 대신 미발견

"""gosi_index 회귀 — 고시목록 CSV 파싱·번호 정규화·URL 매니페스트 (순수, 네트워크 없음).

실행: python -m pytest tests/test_gosi_index.py
"""
import pandas as pd

from redev.data.ingest.gosi_index import _gosi_no, load_index, urls_for


def test_gosi_no_extracts_yyyy_nnn():
    assert _gosi_no("서울특별시 고시 제2025-179호") == "2025-179"
    assert _gosi_no("2025-179") == "2025-179"
    assert _gosi_no("제목만 있고 번호 없음") is None


def test_load_index_normalizes(tmp_path):
    csv = tmp_path / "idx.csv"
    pd.DataFrame({
        "고시번호": ["서울특별시 고시 제2025-179호", "제2024-475호", "번호없음"],
        "고시일자": ["2025-04-03", "2024-10-04", "2024-01-01"],
        "접속링크주소(URL)": ["http://eum.go.kr/a", "http://eum.go.kr/b", "http://x"],
        "고시명": ["상도14구역 정비계획", "성북1구역 공공재개발", "기타"],
    }).to_csv(csv, index=False, encoding="utf-8-sig")
    idx = load_index(csv)
    assert set(idx["고시번호"]) == {"2025-179", "2024-475"}   # 번호없음 행 drop
    assert idx.set_index("고시번호").loc["2025-179", "url"] == "http://eum.go.kr/a"


def test_urls_for_manifest_and_missing(tmp_path):
    csv = tmp_path / "idx.csv"
    pd.DataFrame({"고시번호": ["2025-179"], "고시일자": ["2025-04-03"],
                  "접속링크주소(URL)": ["http://eum.go.kr/a"], "고시명": ["상도14구역"]}).to_csv(
        csv, index=False, encoding="utf-8-sig")
    idx = load_index(csv)
    man = urls_for(["2025-179", "2099-999"], idx)
    assert man[0] == {"고시번호": "2025-179", "고시명": "상도14구역", "url": "http://eum.go.kr/a"}
    assert man[1]["url"] is None                              # 미발견(연간 갱신 전 등) → None

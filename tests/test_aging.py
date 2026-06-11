"""aging 회귀 테스트 — R1(시점 노후도)의 핵심 성질을 고정.

실행: python -m pytest tests/test_aging.py
"""
import pandas as pd

from redev.data.aging import old_ratio_as_of, old_ratio_by_parcel


def _parcel(rows):
    return pd.DataFrame(rows)


def test_r1_time_anchor_changes_ratio():
    """R1 핵심: 같은 필지가 t에 따라 다른 노후도. 지정 후 신축이 들어오면 낮아진다."""
    A = _parcel([
        {"pnu": "A", "approval_year": 1985, "structure": "other", "gross_floor_area": 100},
        {"pnu": "A", "approval_year": 1990, "structure": "rc", "gross_floor_area": 100},
        {"pnu": "A", "approval_year": 2024, "structure": "rc", "gross_floor_area": 400},
        {"pnu": "A", "approval_year": 2025, "structure": "rc", "gross_floor_area": 400},
    ])
    r2021 = old_ratio_as_of(A, 2021, weight="area")
    r2026 = old_ratio_as_of(A, 2026, weight="area")
    assert r2021 == 1.0           # 2021엔 옛 건물뿐 → 전부 노후
    assert r2026 < r2021          # 2024/25 신축이 희석 → 낮아짐(누수 방향)


def test_future_buildings_excluded_nan():
    """t 이전 건물이 없으면 거짓 0이 아니라 NaN(평가 불가)."""
    F = _parcel([{"pnu": "F", "approval_year": 2024, "structure": "rc", "gross_floor_area": 100}])
    r = old_ratio_as_of(F, 2020)
    assert r != r                 # NaN


def test_batch_matches_single():
    """벡터화 배치(old_ratio_by_parcel) == 단건(old_ratio_as_of)."""
    df = _parcel([
        {"pnu": "A", "approval_year": 1985, "structure": "other", "gross_floor_area": 100},
        {"pnu": "A", "approval_year": 2024, "structure": "rc", "gross_floor_area": 400},
        {"pnu": "F", "approval_year": 2024, "structure": "rc", "gross_floor_area": 100},
    ])
    batch = old_ratio_by_parcel(df, 2026, weight="area")
    assert round(batch["A"], 3) == round(old_ratio_as_of(df[df.pnu == "A"], 2026, weight="area"), 3)
    assert batch["F"] == 0.0      # 2024 신축 단독 → 노후 0

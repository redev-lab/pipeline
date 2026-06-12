"""land_price 회귀 테스트 — as-of-t 선택·결측 처리(mock, 파일 없이).

실행: python -m pytest tests/test_land_price.py
"""
import pandas as pd

import redev.data.ingest.land_price as lp


def test_asof_t_selects_label_year(monkeypatch):
    """★학습 피처는 라벨 t의 연도 백분위(as-of-t) — 2008 라벨은 2008 파일에서."""
    def fake(year):
        return pd.Series({"A": 0.9}) if year == 2008 else pd.Series({"A": 0.1, "B": 0.5})
    monkeypatch.setattr(lp, "_year_percentile", fake)
    out = lp.land_price_features(pd.DataFrame({"pnu": ["A", "B"], "t": [2008, 2026]}))
    assert out["land_pct"].iloc[0] == 0.9      # A@2008 → 2008 백분위
    assert out["land_pct"].iloc[1] == 0.5      # B@2026 → 2026 백분위


def test_missing_is_half_plus_flag(monkeypatch):
    """공시지가 없는 PNU → 백분위 0.5 + 결측 플래그(결측을 신호로 보존)."""
    monkeypatch.setattr(lp, "_year_percentile", lambda year: pd.Series(dtype=float))
    out = lp.land_price_features(pd.DataFrame({"pnu": ["Z"], "t": [2008]}))
    assert out["land_pct"].iloc[0] == 0.5 and out["land_missing"].iloc[0] == 1


def test_future_t_clamped_to_current(monkeypatch):
    """t>현재면 현재연도로(추론은 2026)."""
    seen = {}
    def fake(year):
        seen["y"] = year
        return pd.Series({"A": 0.3})
    monkeypatch.setattr(lp, "_year_percentile", fake)
    lp.land_price_features(pd.DataFrame({"pnu": ["A"], "t": [2030]}), current_year=2026)
    assert seen["y"] == 2026

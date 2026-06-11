"""labels 회귀 테스트 — 충돌해소 + R5(해제→재지정 두 행)를 고정.

실행: python -m pytest tests/test_labels.py
"""
import pandas as pd

from redev.data.labels import LABEL_COLUMNS, _resolve_conflicts


def _row(pnu, t, label, certainty, source):
    base = {c: pd.NA for c in LABEL_COLUMNS}
    base.update(pnu=pnu, t=t, label=label, certainty=certainty, source=source, contaminated=False)
    return base


def test_conflict_same_pnu_t_priority():
    """같은 (pnu,t)에 positive+negative → positive 1행만(우선순위)."""
    rows = pd.DataFrame([
        _row("P", 2020, 0, "uncertain", "노후미지정"),
        _row("P", 2020, 1, "positive", "의제처리"),
        _row("P", 2020, 0, "reliable_negative", "신축파생"),
    ])
    out = _resolve_conflicts(rows)
    assert len(out) == 1
    assert out.iloc[0]["certainty"] == "positive"


def test_r5_different_t_both_kept():
    """같은 PNU라도 t가 다르면 두 행 유지(R5: 해제 neg + 재지정 pos)."""
    rows = pd.DataFrame([
        _row("Q", 2014, 0, "reliable_negative", "해제"),
        _row("Q", 2022, 1, "positive", "의제처리"),
    ])
    out = _resolve_conflicts(rows)
    assert len(out) == 2
    assert set(out["label"]) == {0, 1}


def test_no_duplicate_pnu_t_after_resolve():
    rows = pd.DataFrame([
        _row("A", 2020, 1, "positive", "의제처리"),
        _row("A", 2020, 0, "uncertain", "노후미지정"),
        _row("B", 2026, 0, "reliable_negative", "신축파생"),
    ])
    out = _resolve_conflicts(rows)
    assert out.duplicated(["pnu", "t"]).sum() == 0

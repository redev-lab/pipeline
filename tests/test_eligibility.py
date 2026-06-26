"""eligibility 회귀 테스트 — 토허 물건분기·단계 룩업·진단/예언 분리(수검 케이스).

실행: python -m pytest tests/test_eligibility.py
"""
from redev.rules.eligibility import score_eligibility, stage_remaining, toheo_status


def test_apartment_always_toheo_no_gap():
    """★4구 아파트 → 토허 적용·실거주 의무·갭투자 불가."""
    s = toheo_status("아파트")
    assert s["toheo_applies"] and s["residence_duty"] and not s["gap_investment_possible"]
    assert s["basis_date"] == "2026-06"                     # 기준시점 표기(토허 수시변경)


def test_plain_villa_not_toheo_gap_ok():
    """일반 빌라(단지요건 미충족) → 토허 비대상·갭투자 가능."""
    s = toheo_status("다세대", danji_qualified=False)
    assert not s["toheo_applies"] and s["gap_investment_possible"]


def test_danji_villa_toheo_no_gap():
    """단지요건 충족 연립·다세대(단지 내 아파트 동 포함) → 토허 대상·갭투자 불가."""
    s = toheo_status("연립", danji_qualified=True)
    assert s["toheo_applies"] and not s["gap_investment_possible"]


def test_stage_remaining_order_and_unknown():
    """후반 단계일수록 잔여기간 짧다(순서 정합) + 미등록 단계는 거짓값 안 냄."""
    assert (stage_remaining("관리처분인가")["remaining_years"]["typical"]
            < stage_remaining("조합설립인가")["remaining_years"]["typical"])
    assert stage_remaining("없는단계")["known"] is False


def test_score_splits_diagnosis_and_prophecy():
    """★진단(토허=사실)/예언(잔여기간=추정) 라벨 분리(§6)."""
    out = score_eligibility("아파트", "사업시행인가")
    assert "진단_토허" in out and "예언_잔여기간" in out
    assert out["진단_토허"]["toheo_applies"] is True
    assert out["예언_잔여기간"]["known"] is True


def test_stage_suppressed_when_not_in_zone():
    """★계약 §11-3: 후보 구역 아니면 단계·잔여기간 출력 금지(기본값 누수 차단)."""
    sr = stage_remaining("사업시행인가", in_zone=False)
    assert sr["known"] is False and "remaining_years" not in sr
    assert sr["stage"] is None and "구역이 아니라" in sr["note"]


def test_stage_suppressed_when_no_stage_input():
    """★계약 §11-3: 단계 미입력(None)이면 기본값으로 둔갑시키지 않고 추정 안 함."""
    sr = stage_remaining(None, in_zone=True)
    assert sr["known"] is False and "remaining_years" not in sr
    assert "입력되지 않" in sr["note"]
    # 비후보 + 단계 입력이어도 구역 아님이 우선(잔여기간 안 나옴)
    out = score_eligibility("다세대", "사업시행인가", in_zone=False)
    assert out["예언_잔여기간"]["known"] is False

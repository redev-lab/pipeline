"""eligibility.py — 진입 가능성 룩업(토허·잔여기간). 모델 아님. 설계: eligibility.md.

★토허 = "구역 룩업"이 아니라 "서울 전역 기본 + 물건유형 분기"(§11, 2025.10 대책). 모든 값은
config(규칙5), 토허는 휘발성 최상급이라 기준시점 표기 + 재확인 경고(R15). 진단(현재 사실: 토허)과
예언(미래 추정: 잔여기간)을 분리한다(§6).
"""
from __future__ import annotations

from redev.config import load_eligibility_config

# 연립·다세대로 보는 물건유형 명칭(분기용).
_VILLA = {"villa", "yeonlip", "연립", "다세대", "연립다세대"}


def toheo_status(property_type: str, *, danji_qualified: bool = False, cfg=None) -> dict:
    """★진단(현재 사실): 토허 적용·갭투자 가능 여부. 물건유형 분기(§11).

    아파트=항상 대상 / 연립·다세대=단지요건(단지 내 아파트 동 포함) 충족 시만. 토허 대상이면
    실거주 의무 → 갭투자 불가. 기준시점 동봉(토허는 수시 변경, R15).
    """
    th = (cfg or load_eligibility_config())["toheo"]
    if not th["default_applies"]:
        applies = False
    elif property_type == "apartment" or property_type == "아파트":
        applies = bool(th["apartment_always"])
    elif property_type in _VILLA:
        applies = bool(th["villa_requires_danji"]) and bool(danji_qualified)
    else:
        applies = bool(th["default_applies"])              # 미상 유형 보수적(전역 기본)
    duty = applies and bool(th["residence_duty"])
    return {
        "toheo_applies": applies,
        "gap_investment_possible": not duty,               # 실거주 의무면 갭투자 불가
        "residence_duty": duty,
        "basis_date": th["basis_date"],
        "caveats": [
            f"★토허는 수시 지정·해제 — 위 판정은 {th['basis_date']} 기준. 매 사용 시 현행 고시 재확인.",
            "상업 배포 전 법률 검토 필수(R15).",
        ],
    }


def stage_remaining(stage: str, *, cfg=None) -> dict:
    """★예언(미래 추정): 정비사업 단계 → 잔여기간 범위. 단정 금지, 외생변수 천장(§6·R18)."""
    tbl = (cfg or load_eligibility_config())["stage_remaining_years"]
    r = tbl.get(stage)
    if r is None:
        return {"stage": stage, "known": False,
                "note": "미등록 단계 — 잔여기간 추정 불가(거짓값 내지 않음)."}
    return {
        "stage": stage, "known": True, "remaining_years": r,
        "caveats": ["범위·변동 큼 — 분쟁·경기·분양 등 외생변수가 좌우(R18). 단정 아님(§6 예언)."],
    }


def score_eligibility(property_type: str, stage: str, *, danji_qualified: bool = False, cfg=None) -> dict:
    """진단(토허)+예언(잔여기간) 분리 출력. 추정을 사실로 오해하지 않게 라벨 분리(§6)."""
    cfg = cfg or load_eligibility_config()
    return {
        "진단_토허": toheo_status(property_type, danji_qualified=danji_qualified, cfg=cfg),
        "예언_잔여기간": stage_remaining(stage, cfg=cfg),
        "note": "진단(현재 사실)과 예언(미래 추정)을 분리한다 — 추정을 사실로 읽지 말 것(§6).",
    }

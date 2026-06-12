"""feasibility.py — Stage 2 추진 가능성 (B1 보정·랭킹 레이어). 설계: feasibility.md.

★기대치(§0): v1은 "무산 확률"을 내지 않는다 — R18 외생변수 천장 + hard-neg 해제 n=24
(Phase 3 실측 FPR 0.58)로 학습 불가. 정직한 산출 = calibration된 랭킹 + 리스크 자리표시.
★명칭 정직성: 이 점수는 "재개발 환경 유사도"(B1이 배운 것)지 "추진 성공 가능성"이 아니다.
"""
from __future__ import annotations

import numpy as np

# ★사용자 표시 명칭 — '추진 가능성' 아님(점수 축소한 만큼 이름도 축소, R15).
ENV_SCORE_LABEL = "재개발 환경 점수"


def oof_scores(aug, edge_index, pnu_to_idx, *, feat_cols=None) -> np.ndarray:
    """심장1 B1의 ★LODO out-of-fold 예측(보정 입력). train 확률 아님 → 누수 차단(R3).

    B1 재사용(새 모델 아님, R9 일관) — feasibility는 그 위의 보정·랭킹 레이어.
    """
    from redev.models.baseline import production_feature_set, run_xgb_cv
    fc = feat_cols or production_feature_set(aug)     # ★production B1+ −용도지역(infer와 일관)
    return run_xgb_cv(aug, fc, edge_index, pnu_to_idx, model_name="B1+prod")["all_p"]


def calibrate(oof: np.ndarray, y: np.ndarray):
    """isotonic 보정자 적합 — ★OOF 예측 위에서만(train 확률로 하면 누수). out_of_bounds clip."""
    from sklearn.isotonic import IsotonicRegression
    m = np.isfinite(oof)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(oof[m], y[m])
    return ir


def reliability(prob: np.ndarray, y: np.ndarray, *, bins: int = 10) -> dict:
    """reliability diagram + ECE(기대보정오차). 보정 품질 — 대각선에 가까울수록 정직."""
    prob, y = np.asarray(prob), np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc, w = float(prob[m].mean()), float(y[m].mean()), float(m.mean())
        ece += w * abs(conf - acc)
        rows.append({"bin": b, "n": int(m.sum()), "conf": round(conf, 3), "acc": round(acc, 3)})
    return {"ece": round(ece, 4), "bins": rows}


def topk_precision(prob: np.ndarray, y: np.ndarray, ks=(0.1, 0.2, 0.5)) -> dict:
    """랭킹 품질: 상위 K%가 positive를 얼마나 앞에 모으나(정확도 아님)."""
    order = np.argsort(-np.asarray(prob))
    ys = np.asarray(y)[order]
    n = len(ys)
    return {f"top{int(k*100)}%": round(float(ys[:max(1, int(k * n))].mean()), 3) for k in ks}


def risk_signal_placeholder() -> dict:
    """★R18 — 무산 리스크 v1 미학습 명시(거짓 0/단정 금지). 미래 신호 슬롯."""
    return {
        "status": "v1_미학습",
        "reason": "선정·추진 성공은 외생변수(주민동의율·정치·시장) 영역 — 본 점수 미측정(R18). "
                  "hard-neg 해제 n=24로 학습 불가(labels §13).",
        "future_slots": ["재추진 이력(장위13)", "주민동의율", "조합 분쟁 뉴스(nlp layer3)"],
    }


def _rank_phrase(p: float) -> str:
    """★백분위 표기(계약 §B-1) — p=상위 몇 %(낮을수록 상위). p≤50 '상위 p%', p>50 '하위 (100−p)%'.
    '상위 100%'(최하위) 같은 혼란을 없앤다. 표시 계층 함수(숫자 자체는 안 바꿈, 문구만)."""
    return f"상위 {p}%" if p <= 50 else f"하위 {round(100 - p, 1)}%"


def score_feasibility(raw_score: float, raw_reference: np.ndarray, *,
                      calibrated_prob: float | None = None) -> dict:
    """공개 진입점 — ★'재개발 환경 점수' 랭킹 + 리스크 자리표시 + 명칭 정직성.

    ★백분위(계약 §B-2)는 *raw 점수의 전 노드 순위*로 계산한다 — 보정확률은 상단이 포화돼
    서로 다른 raw(0.993·0.994)가 같은 백분위로 뭉친다. raw 순위는 단조라 동률 뭉침이 사라진다.
    보정확률(calibrated_prob)은 내부 메타로만 보존(표시 안 함). rank_top_pct = 상위 몇 %(낮을수록 상위).
    """
    ref = np.asarray(raw_reference)
    better_than = float((ref < raw_score).mean())
    rank_top_pct = round(100.0 * (1 - better_than), 1)
    return {
        "label": ENV_SCORE_LABEL,                         # ★'추진 가능성' 아님
        "rank_top_pct": rank_top_pct,
        "rank_phrase": _rank_phrase(rank_top_pct),        # ★표시 문구(상위/하위)
        "raw_score": round(float(raw_score), 4),
        "calibrated_prob": round(float(calibrated_prob), 3) if calibrated_prob is not None else None,  # 메타
        "risk_signals": risk_signal_placeholder(),
        "caveats": [
            "이 점수는 '재개발 환경 유사도'이지 '추진 성공 가능성'이 아니다 — 선정·추진은 "
            "외생변수(주민동의·정치) 영역으로 측정 안 함(R18).",
            "추정·참고치이며 투자 권유 아님(R15).",
        ],
    }

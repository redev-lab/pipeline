"""metrics.py — 불균형·격전지 인지 평가 지표 (순수함수). spatial_cv가 호출.

★규칙: 정확도(accuracy) 금지(R8). 헤드라인은 PR-AUC. 추가로 R9 승부처인 격전지
recall(aging=0 positive)·hard/easy negative 분리를 별도 함수로 제공(spatial_cv §6).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def pr_auc(y, p) -> float:
    """PR-AUC(average precision) — 불균형 헤드라인(R8). 한 클래스뿐이면 nan."""
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, p))


def f1_at(y, p, thr: float = 0.5) -> float:
    """임계 thr에서 F1 (양성기준 2·tp/(2·tp+fp+fn))."""
    y = np.asarray(y)
    yhat = (np.asarray(p) >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def best_f1(y, p) -> tuple[float, float]:
    """임계값 스윕 best-F1과 그 임계값(격전지·neg분리 리포트의 공통 작동점)."""
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    i = int(np.nanargmax(f1))
    t = float(thr[i]) if i < len(thr) else 1.0
    return float(f1[i]), t


def battleground_recall(y, p, aging, thr: float) -> tuple[float, int]:
    """★aging=0 positive에 대한 recall (R9 승부처). B-2(aging 임계)는 정의상 0%.

    self aging=0 양성은 자기 피처론 음성과 구분 불가 → 이웃/구조로만 잡힌다. 이 한 칸이
    동어반복 너머 '구조의 값'. (spatial_cv §6.5 / baseline §6.5)
    """
    y, aging = np.asarray(y), np.asarray(aging)
    hit = np.asarray(p) >= thr
    mask = (y == 1) & (aging == 0.0)
    n = int(mask.sum())
    return (float(hit[mask].mean()) if n else float("nan")), n


def neg_split_report(y, p, neg_reason, thr: float) -> dict:
    """hard(해제)/easy(신축) negative 분리 위양성률(FPR)+n.

    ★hard(해제)는 학습 24노드뿐(labels §13) → n을 같이 박아 '통계적 무의미'를 가시화.
    그 수치로 결론 내지 말 것. easy(신축)는 aging로 자명 분리(동어반복).
    """
    y = np.asarray(y)
    hit = np.asarray(p) >= thr
    nr = np.asarray(neg_reason).astype(str)
    out = {}
    for name, key in (("hard_해제", "cancelled"), ("easy_신축", "new_construction")):
        m = (y == 0) & (nr == key)
        n = int(m.sum())
        out[name] = {"fpr": (float(hit[m].mean()) if n else float("nan")), "n": n}
    return out

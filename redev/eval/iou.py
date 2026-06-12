"""iou.py — 후보 클러스터 vs 실제 지정구역 경계 평가 (R13). 설계: pipeline.md §2.

노드 분류를 이어붙인 클러스터는 구멍·삐죽 → 폴리곤 IoU 천장(R13). 그래서 IoU와 함께
★핵심부 포착률(코어 recall)을 본다. 클러스터·구역을 PNU 집합으로 보고 집합 IoU로 단순화
(필지 단위 = 노드 단위, 폴리곤 누적오차 회피).
"""
from __future__ import annotations

import numpy as np


def set_iou(pred: set, target: set) -> float:
    """집합 IoU = |∩| / |∪| (필지 단위)."""
    if not pred and not target:
        return 0.0
    inter = len(pred & target)
    return inter / len(pred | target)


def core_capture(pred: set, target: set) -> float:
    """★핵심부 포착률 = |pred ∩ target| / |target| (구역 코어를 예측이 덮은 비율, R13).

    IoU는 예측이 클수록 합집합이 커져 천장에 부딪힌다 → 코어 recall로 "구역을 놓쳤나"를 본다.
    """
    if not target:
        return float("nan")
    return len(pred & target) / len(target)


def best_match(clusters: list, zone_pnus: set) -> tuple:
    """한 지정구역에 대해 가장 겹치는 클러스터의 (IoU, 핵심부 포착률). 매칭 클러스터 없으면 0."""
    if not clusters:
        return 0.0, 0.0
    ious = [(set_iou(c, zone_pnus), core_capture(c, zone_pnus)) for c in clusters]
    return max(ious, key=lambda x: x[0])      # IoU 최대 매칭


def compare_methods(method_clusters: dict, zones: dict) -> dict:
    """방법별(넓은/타이트/B0) × 지정구역별 IoU·핵심부 포착률 평균. R13 통제 비교표.

    method_clusters: {method: [클러스터 PNU집합...]}. zones: {zone_id: 지정구역 PNU집합}.
    반환: {method: {mean_iou, mean_core_capture, n_zones, n_clusters, avg_cluster_size}}.
    """
    out = {}
    for method, clusters in method_clusters.items():
        ious, cores = [], []
        for z_pnus in zones.values():
            iou, core = best_match(clusters, z_pnus)
            ious.append(iou)
            cores.append(core)
        sizes = [len(c) for c in clusters]
        out[method] = {
            "mean_iou": round(float(np.mean(ious)), 3),
            "mean_core_capture": round(float(np.nanmean(cores)), 3),
            "n_zones": len(zones),
            "n_clusters": len(clusters),
            "avg_cluster_size": round(float(np.mean(sizes)), 0) if sizes else 0,
        }
    return out

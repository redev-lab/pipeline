"""eval 회귀 테스트 — metrics 순수함수 + spatial_cv 버퍼/커버리지(R3).

실행: python -m pytest tests/test_eval.py
"""
import numpy as np
import pandas as pd

from redev.eval.metrics import battleground_recall, best_f1, neg_split_report, pr_auc
from redev.eval.spatial_cv import Fold, apply_buffer, lodo_folds, spatial_zone_groups


# ── metrics ──
def test_pr_auc_perfect():
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_battleground_recall_only_aging0_positive():
    """aging=0 양성만 셈: idx0,1 양성·aging0, 예측 [T,F] → recall 0.5, n=2."""
    r, n = battleground_recall([1, 1, 0], [0.9, 0.1, 0.1], [0.0, 0.0, 0.5], thr=0.5)
    assert n == 2 and r == 0.5


def test_neg_split_hard_easy():
    out = neg_split_report([0, 0], [0.9, 0.1], ["cancelled", "new_construction"], thr=0.5)
    assert out["hard_해제"] == {"fpr": 1.0, "n": 1}      # 해제 1개 위양성
    assert out["easy_신축"] == {"fpr": 0.0, "n": 1}


# ── spatial_cv ──
def _toy_labels():
    # 2구(01 test / 02 train). C는 A의 그래프 1홉 + 5m 근접 → 버퍼로 제거돼야.
    return pd.DataFrame({
        "pnu": ["A", "B", "C", "D"],
        "sigungu": ["01", "02", "02", "02"],
        "centroid_x": [0.0, 1000.0, 5.0, 5000.0],
        "centroid_y": [0.0, 0.0, 0.0, 0.0],
        "y": [1, 0, 0, 0], "zone_id": ["z1", pd.NA, pd.NA, pd.NA],
    })


def test_lodo_coverage():
    folds = lodo_folds(_toy_labels())
    allt = np.concatenate([f.test_idx for f in folds])
    assert len(allt) == len(np.unique(allt)) == 4      # 모든 행 1회씩 test


def test_apply_buffer_removes_graph_and_geom():
    labels = _toy_labels()
    pnu_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    edge_index = np.array([[0, 2], [2, 0]])            # A-C 인접(대칭)
    test_fold = Fold("01", np.array([1, 2, 3]), np.array([0]))
    out = apply_buffer(test_fold, labels, edge_index, pnu_to_idx, hops=1, buffer_m=200)
    # C(행2): A의 1홉 + 5m 근접 → 제거. B·D는 1000·5000m라 유지.
    assert set(out.train_idx.tolist()) == {1, 3}
    assert out.buffer_report["removed_graph_hop"] == 1 and out.buffer_report["removed_geom"] == 1


def test_spatial_zone_groups_never_splits_zone():
    # 두 구역(z1 원점근처, z2 멀리) — k=2면 각 구역이 통째로 한 그룹에.
    labels = pd.DataFrame({
        "pnu": list("abcd"), "y": [1, 1, 1, 1],
        "zone_id": ["z1", "z1", "z2", "z2"],
        "centroid_x": [0.0, 1.0, 9000.0, 9001.0], "centroid_y": [0.0, 0.0, 0.0, 0.0],
    })
    groups = spatial_zone_groups(np.arange(4), labels, k=2)
    zsets = [set(labels.iloc[g].zone_id) for g in groups]
    assert {frozenset(z) for z in zsets} == {frozenset({"z1"}), frozenset({"z2"})}

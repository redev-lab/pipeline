"""spatial_cv.py — R3 공간 검증 (LODO fold + 버퍼 + 구역단위 inner + 격전지 채점).

설계: docs/design/spatial_cv.md. 무작위 CV는 옆필지 커닝으로 성능을 뻥튀기한다(유효표본
42구역). 이 모듈이 split의 유일한 책임자 — 모델은 split-agnostic(baseline.md §7).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from redev.config import load_graph_config
from redev.eval.metrics import battleground_recall, best_f1, neg_split_report, pr_auc

# 자치구 코드 → 이름(per-fold 표 가독성, spatial_cv §3).
DISTRICT_NAMES = {"11290": "성북", "11590": "동작", "11380": "은평", "11530": "구로"}


@dataclass
class Fold:
    name: str
    train_idx: np.ndarray            # labels에 대한 행 인덱스
    test_idx: np.ndarray
    buffer_report: dict = field(default_factory=dict)


def lodo_folds(labels) -> list[Fold]:
    """자치구 단위 leave-one-district-out (4 fold). test=1구, train=나머지(버퍼 전).

    GNN의 inductive(학습 안 한 구 추론, CLAUDE §8)를 직접 시험. build_graph가 구 배치라
    구간 엣지=0 → 메시지패싱 누수 구조적 불가, 피처누수만 버퍼로 막으면 됨(§3).
    """
    sig = labels["sigungu"].to_numpy()
    folds = []
    for code in sorted(set(sig)):
        folds.append(Fold(
            name=DISTRICT_NAMES.get(code, str(code)),
            train_idx=np.where(sig != code)[0],
            test_idx=np.where(sig == code)[0],
        ))
    return folds


def _hop_neighbors(seed_global: np.ndarray, edge_index: np.ndarray, hops: int) -> set:
    """seed 전역노드에서 hops 홉 이내 전역노드 집합(그래프 버퍼용). edge_index 대칭 가정."""
    src, dst = edge_index[0], edge_index[1]
    order = np.argsort(src, kind="stable")
    src_s, dst_s = src[order], dst[order]
    closed = set(seed_global.tolist())
    frontier = seed_global
    for _ in range(hops):
        m = np.isin(src_s, frontier)
        nb = np.unique(dst_s[m])
        seen = np.fromiter(closed, dtype=np.int64) if closed else np.empty(0, np.int64)
        new = nb[~np.isin(nb, seen)]
        if new.size == 0:
            break
        closed.update(new.tolist())
        frontier = new
    return closed


def apply_buffer(fold: Fold, labels, edge_index, pnu_to_idx, *, hops: int, buffer_m: float) -> Fold:
    """train에서 test의 (a)그래프 hops 이웃 (b)기하 buffer_m 이내 필지 제외(R3).

    LODO에선 구간 엣지가 없어 (a)는 거의 무효(수검 출력 ~0 예상), (b)가 구 경계 안전벨트.
    (a)는 같은 구 안 inner zone-holdout에서 실효. 버퍼 폭=수용영역(hops, §4).
    """
    test_global = np.fromiter((pnu_to_idx[p] for p in labels["pnu"].to_numpy()[fold.test_idx]),
                              dtype=np.int64, count=len(fold.test_idx))
    # (a) 그래프 홉 이웃
    nbr = _hop_neighbors(test_global, edge_index, hops) - set(test_global.tolist())
    train_global = np.array([pnu_to_idx.get(p, -1) for p in labels["pnu"].to_numpy()[fold.train_idx]])
    nbr_arr = np.fromiter(nbr, np.int64) if nbr else np.empty(0, np.int64)
    in_hop = np.isin(train_global, nbr_arr)
    # (b) 기하 버퍼: test centroid에서 buffer_m 이내 train 필지
    cx, cy = labels["centroid_x"].to_numpy(), labels["centroid_y"].to_numpy()
    tree = cKDTree(np.c_[cx[fold.test_idx], cy[fold.test_idx]])
    dist, _ = tree.query(np.c_[cx[fold.train_idx], cy[fold.train_idx]], k=1)
    in_geom = dist < buffer_m
    keep = ~in_hop & ~in_geom
    report = {
        "removed_graph_hop": int(in_hop.sum()),
        "removed_geom": int(in_geom.sum()),
        "removed_total": int((~keep).sum()),
        "train_before": int(len(fold.train_idx)),
        "train_after": int(keep.sum()),
    }
    return Fold(fold.name, fold.train_idx[keep], fold.test_idx, report)


def build_lodo_folds(labels, edge_index, pnu_to_idx, *, cfg=None) -> list[Fold]:
    """LODO 4 fold + 버퍼 적용(config). 모델 평가의 표준 진입점."""
    cv = (cfg or load_graph_config())["cv"]
    return [apply_buffer(f, labels, edge_index, pnu_to_idx,
                         hops=cv["buffer_hops"], buffer_m=cv["buffer_m"])
            for f in lodo_folds(labels)]


def spatial_zone_groups(idx: np.ndarray, labels, *, k: int, seed: int = 0) -> list[np.ndarray]:
    """행 부분집합 idx를 k개 공간그룹으로 — inner zone-holdout용(§5).

    ★양성=zone_id 원자(절대 안 쪼갬). zone 중심을 KMeans(k)로 군집, 음성은 가장 가까운
    zone 그룹에 배정. 무작위 행 분할 금지(튜닝 뒷문 누수 차단). 반환: 길이 k 행idx 배열.
    """
    sub = labels.iloc[idx]
    yv = sub["y"].to_numpy()
    zid = sub["zone_id"].to_numpy()
    cx, cy = sub["centroid_x"].to_numpy(), sub["centroid_y"].to_numpy()
    pos = sub[sub["y"] == 1]
    zc = pos.groupby("zone_id")[["centroid_x", "centroid_y"]].mean()
    k = min(k, len(zc)) if len(zc) else 1
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(zc.values)
    zone_group = dict(zip(zc.index, km.labels_))
    tree = cKDTree(zc.values)                       # 음성→최근접 zone 그룹

    groups: list[list[int]] = [[] for _ in range(k)]
    for j, row in enumerate(idx):
        if yv[j] == 1:
            g = int(zone_group[zid[j]])
        else:
            _, zi = tree.query([cx[j], cy[j]])
            g = int(km.labels_[zi])
        groups[g].append(int(row))
    return [np.array(g, dtype=np.int64) for g in groups]


def evaluate(predict_fn, folds: list[Fold], labels, *, model: str = "model") -> dict:
    """각 fold predict_fn(train_idx,test_idx)->p_test. pooled(헤드라인)+per-fold 지표.

    지표(§6): PR-AUC(R8)·best-F1·격전지 recall(aging=0 positive)·hard/easy neg 분리.
    작동점=best-F1 임계(격전지·neg분리 공통). per-fold는 헤드라인과 항상 병기(§3).
    """
    y = labels["y"].to_numpy()
    aging = labels["aging"].to_numpy()
    nr = labels["neg_reason"].to_numpy()
    all_p = np.full(len(labels), np.nan)

    per_fold = []
    for f in folds:
        p = np.asarray(predict_fn(f.train_idx, f.test_idx), dtype=float)
        all_p[f.test_idx] = p
        yt = y[f.test_idx]
        bf, bt = best_f1(yt, p)
        bg, bgn = battleground_recall(yt, p, aging[f.test_idx], bt)
        per_fold.append({
            "fold": f.name, "n_test": int(len(f.test_idx)), "n_pos": int(yt.sum()),
            "pr_auc": pr_auc(yt, p), "best_f1": bf,
            "battleground_recall": bg, "battleground_n": bgn,
            "neg_split": neg_split_report(yt, p, nr[f.test_idx], bt),
            "buffer": f.buffer_report,
        })

    m = ~np.isnan(all_p)
    yt, p = y[m], all_p[m]
    bf, bt = best_f1(yt, p)
    bg, bgn = battleground_recall(yt, p, aging[m], bt)
    pooled = {
        "pr_auc": pr_auc(yt, p), "best_f1": bf, "best_thr": bt,
        "battleground_recall": bg, "battleground_n": bgn,
        "neg_split": neg_split_report(yt, p, nr[m], bt),
    }
    return {"model": model, "pooled": pooled, "per_fold": per_fold}

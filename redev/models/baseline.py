"""baseline.py — 심장1(GNN)의 정직한 대조군 (Phase 3, R9).

이 모듈은 GNN과 **같은 입력·라벨·공간CV** 위에서 비-그래프 모델을 돌려 "메시지
패싱이 실제로 값을 더하는가"를 측정한다. 설계: docs/design/baseline.md.

대조군:
  B0  region-growing      — 무학습 공간확장(바닥선, R13 좌표축)
  B1  XGBoost + 1홉집계    — 이웃 1홉 손피처 (R9 강한 대조군)
  B1+ XGBoost + 2홉집계    — GNN 2층 수용영역과 도달범위 맞춘 ablation

이 파일의 첫 책임은 **학습행렬 조립·캐시**(load_training_matrix): 무거운 ingest를
한 번 돌려 parquet로 굳히고, 이후 하이퍼파라미터 실행은 캐시만 읽는다(R10 연장).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from redev.config import load_graph_config, training_sigungu_codes
from redev.data.ingest.building_gis import load_buildings
from redev.data.ingest.cancelled import load_cancelled
from redev.data.ingest.parcels import build_jibun_index, load_parcels
from redev.data.ingest.zone_boundary import load_zones
from redev.data.labels import build_label_table
from redev.graph.build import build_graph, reconcile_labels_to_graph
from redev.graph.features import FEATURE_COLUMNS, node_features

# ── v1 로컬 원천 경로. 상업 전환 시 소스 교체(§11), config 승격은 v1.1. ──
_RAW = Path("_data/raw")
_PROCESSED = Path("_data/processed")


def _vsizip(zip_name: str, inner: str) -> str:
    """GDAL /vsizip 가상경로. zip 내부 SHP를 압축 해제 없이 직접 읽는다."""
    # os.path.abspath: 작동 확인된 기존 스크립트와 동일(Windows 백슬래시도 GDAL이 수용).
    return f"/vsizip/{os.path.abspath(_RAW / zip_name)}/{inner}"


# 원천 파일 매핑 — 한 곳에 모아 교체 지점을 명시(§11 소스 추상화).
_SRC = {
    "buildings": ("서울시GIS건물통합정보2026.zip", "AL_D010_11_20260609.shp"),
    "parcels":   ("서울시연속지적도2026.zip", "AL_D002_11_20260608.shp"),
    "uq":        ("UQ181_의제처리구역_202602.zip", "shp파일/UPIS_C_UQ181.shp"),
    "gosi":          "서울시 도시계획 결정고시 정보.csv",
    "cancelled":     "해제구역_negative.csv",
    "jeonbisaeop":   "서울특별시_서울시 정비사업 데이터_20211227.csv",
    "shintong":      "신통_선정구역_positive.csv",
    "public_redev":  "공공재개발_선정목록_4구.csv",
}

_CACHE_MATRIX = _PROCESSED / "train_matrix.parquet"      # 라벨 + self 피처
_CACHE_EDGES = _PROCESSED / "graph_edge_index.npy"       # [2,E] 전역 그래프
_CACHE_PNUIDX = _PROCESSED / "graph_pnu_idx.parquet"     # pnu→idx (전역 노드 매핑)


@dataclass
class TrainingMatrix:
    """Phase 3 학습 입력 묶음.

    labels   : 한 행 = 학습 예시 (pnu, t, y, certainty, zone_id, sigungu, +self피처).
               contaminated·uncertain·비노드는 이미 제외됨.
    edge_index: [2,E] int64 — ★전역 그래프(141K 노드 전체). 이웃집계/GNN이 라벨노드의
               이웃(라벨 없는 노드 포함)을 찾으려면 전역 매핑이 필요해 라벨에 한정 안 함.
    pnu_to_idx: pnu → 전역 노드 인덱스. edge_index가 가리키는 좌표계.
    report   : drop 단계별 행수·불균형비 등 수검 리포트.
    """

    labels: pd.DataFrame
    edge_index: np.ndarray
    pnu_to_idx: dict
    report: dict


def _assemble() -> TrainingMatrix:
    """원천 → 학습행렬 (무거운 경로). load_training_matrix가 캐시 부재 시 호출.

    순서: ingest 6종 → build_label_table → build_graph → reconcile → drop(완공·미확정)
    → node_features(self). 각 호출은 Phase 0~2 모듈에 위임(여기선 조립만).
    """
    codes = training_sigungu_codes()

    # ① ingest. 건물은 노후도용이라 geometry 불필요(R10), 필지는 그래프용이라 필요.
    buildings, _ = load_buildings(_vsizip(*_SRC["buildings"]), with_geometry=False)
    parcels, _ = load_parcels(_vsizip(*_SRC["parcels"]), codes, with_geometry=True)
    zone_table, _ = load_zones(
        _vsizip(*_SRC["uq"]), str(_RAW / _SRC["gosi"]), parcels, codes,
        jeonbisaeop_csv=str(_RAW / _SRC["jeonbisaeop"]),
        shintong_csv=str(_RAW / _SRC["shintong"]),
        public_redev_csv=str(_RAW / _SRC["public_redev"]),
    )
    cancelled_df, _ = load_cancelled(str(_RAW / _SRC["cancelled"]), build_jibun_index(parcels))

    # ② 라벨 테이블(키스톤) — (필지,시점) 행.
    table, _ = build_label_table(zone_table, parcels, buildings, cancelled_df)

    # ③ 전역 인접 그래프. building_overrides_jimok:false(측정, graph.md)라 override 안 씀.
    graph, pnu_to_idx, _ = build_graph(parcels)

    # ④ reconcile: 비노드(도로 등) 라벨 drop — 학습은 그래프 노드 위에서만(R3 브리지).
    table, rec_report = reconcile_labels_to_graph(table, set(pnu_to_idx))

    # ⑤ drop: 완공(contaminated)·미확정(uncertain). 행은 라벨테이블에 남아있고 여기서만 제외
    #    — v1.1 말소대장 부활 시 이 필터만 풀면 복원(설계 §4, R4·R2).
    n_reconciled = len(table)
    n_contam = int(table["contaminated"].sum())
    table = table[~table["contaminated"]]
    n_uncertain = int((table["certainty"] == "uncertain").sum())
    table = table[table["certainty"] != "uncertain"].copy()

    # ⑥ self 피처(각 행을 자기 t로 — features.py가 보강1/R1 시점정합 보장).
    feats = node_features(table[["pnu", "t"]], parcels, buildings)
    labels = table.merge(feats, on=["pnu", "t"], how="left")
    labels["y"] = labels["label"].astype(int)
    labels["sigungu"] = labels["pnu"].str[:5]                 # PNU 앞5 = 시군구코드
    # 필지 중심좌표(EPSG:5186 m) — spatial_cv 기하버퍼(200m)용. 한 번 계산해 캐시.
    cg = parcels[["pnu"]].copy()
    cent = parcels.geometry.centroid
    cg["centroid_x"], cg["centroid_y"] = cent.x.values, cent.y.values
    labels = labels.merge(cg, on="pnu", how="left")
    # neg_reason: hard(해제=cancelled) vs easy(신축=new_construction) 분리 리포트용(spatial_cv §6).
    keep = ["pnu", "t", "y", "certainty", "neg_reason", "zone_id", "sigungu",
            "centroid_x", "centroid_y", *FEATURE_COLUMNS]
    labels = labels[keep].reset_index(drop=True)

    n_pos = int((labels["y"] == 1).sum())
    n_neg = int((labels["y"] == 0).sum())
    report = {
        "reconciled_rows": n_reconciled,
        "dropped_contaminated": n_contam,
        "dropped_uncertain": n_uncertain,
        "final_rows": len(labels),
        "positives": n_pos,
        "negatives": n_neg,
        "imbalance_neg_per_pos": round(n_neg / n_pos, 1) if n_pos else None,
        "reconcile_dropped_non_node": rec_report["dropped_non_node"],
        "graph_nodes": len(pnu_to_idx),
        "graph_edges": int(graph.edge_index.shape[1]),
    }
    return TrainingMatrix(labels, graph.edge_index.numpy(), pnu_to_idx, report)


def load_training_matrix(*, force_rebuild: bool = False) -> TrainingMatrix:
    """학습행렬을 캐시에서 읽거나(빠름) 원천에서 조립(느림) 후 캐시.

    왜 캐시인가: Phase 3는 B0/B1/B1+/GNN × 하이퍼파라미터를 여러 번 돈다. 매번 GIS
    건물·필지(수십만행)를 재적재하면 분 단위 낭비 → 한 번 굳혀 초 단위로(R10).
    캐시 무결성은 수검 ⑤(재로드==즉시조립)에서 확인.

    force_rebuild=True: 원천이 바뀌었거나 피처를 추가했을 때 캐시 무시·재조립.
    """
    if not force_rebuild and _CACHE_MATRIX.exists() and _CACHE_EDGES.exists() and _CACHE_PNUIDX.exists():
        labels = pd.read_parquet(_CACHE_MATRIX)
        edge_index = np.load(_CACHE_EDGES)
        pi = pd.read_parquet(_CACHE_PNUIDX)
        pnu_to_idx = dict(zip(pi["pnu"], pi["idx"]))
        report = json.loads((_PROCESSED / "train_report.json").read_text(encoding="utf-8")) \
            if (_PROCESSED / "train_report.json").exists() else {"cached": True}
        return TrainingMatrix(labels, edge_index, pnu_to_idx, report)

    tm = _assemble()
    _PROCESSED.mkdir(parents=True, exist_ok=True)
    tm.labels.to_parquet(_CACHE_MATRIX, index=False)
    np.save(_CACHE_EDGES, tm.edge_index)
    pd.DataFrame({"pnu": list(tm.pnu_to_idx), "idx": list(tm.pnu_to_idx.values())}) \
        .to_parquet(_CACHE_PNUIDX, index=False)
    (_PROCESSED / "train_report.json").write_text(
        json.dumps(tm.report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return tm


# ──────────────────────────────────────────────────────────────────────────
# 이웃집계 (R9 핵심) — GNN의 메시지 패싱을 손피처로 흉내내 '구조만의 값'을 드러냄.
# ──────────────────────────────────────────────────────────────────────────
_CACHE_NB = _PROCESSED / "train_matrix_nb.parquet"   # self + 이웃집계 증강행렬


def aggregate_once(feat: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """각 노드에 이웃 피처의 mean·max를 부착 (단일 시점 t, 순수함수).

    역할: GNN 한 층이 하는 일 — "이웃의 상태를 모아 자기 옆에 쌓는다" — 을 학습 없이
    한 번 수행. 엣지로 벡터화(노드 루프 없음): src→dst 방향마다 dst의 피처를 src에 모음.

    입력: feat [N,F] (이 t 기준 노드 피처), src·dst [E] (방향 엣지; 그래프는 대칭이라
          양방향 모두 들어옴). 출력: [N, 2F] = [이웃 mean | 이웃 max]. 이웃 없으면 0
          (features.py의 무건물 fillna(0.0)와 같은 규약 — '신호 없음'을 0으로).
    """
    n, f = feat.shape
    sums = np.zeros((n, f), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    np.add.at(sums, src, feat[dst])          # 이웃 피처 합(분산 누적; 중복 src 안전)
    np.add.at(counts, src, 1.0)
    has = counts > 0
    mean = np.zeros((n, f), dtype=np.float64)
    mean[has] = sums[has] / counts[has, None]
    maxs = np.full((n, f), -np.inf, dtype=np.float64)
    np.maximum.at(maxs, src, feat[dst])      # 이웃 피처 최댓값
    maxs[~has] = 0.0
    return np.concatenate([mean, maxs], axis=1)


def _csr_neighbors(edge_index: np.ndarray, n_nodes: int):
    """edge_index [2,E] → src로 정렬된 (src_sorted, dst_sorted). 폐포 확장·로컬엣지용."""
    src, dst = edge_index[0], edge_index[1]
    order = np.argsort(src, kind="stable")
    return src[order], dst[order]


def _closure(seed: np.ndarray, src_s: np.ndarray, dst_s: np.ndarray, hops: int) -> np.ndarray:
    """seed 노드에서 hops 홉 이내 전역 노드 집합(폐포). 2홉 집계가 닿는 모든 노드.

    왜 폐포가 hops여야 하나: 라벨노드 v의 2홉 집계는 v의 이웃 u의 1홉집계를 모으고,
    u의 1홉집계는 u의 이웃 w(=v의 2홉)를 모은다 → w까지 스냅샷에 있어야 정확.
    """
    closed = set(seed.tolist())
    frontier = seed
    for _ in range(hops):
        m = np.isin(src_s, frontier)                 # frontier가 src인 엣지
        nb = np.unique(dst_s[m])
        new = nb[~np.isin(nb, np.fromiter(closed, dtype=np.int64) if closed else np.empty(0, np.int64))]
        if new.size == 0:
            break
        closed.update(new.tolist())
        frontier = new
    return np.array(sorted(closed), dtype=np.int64)


# 증강 컬럼 이름: nb{홉}_{mean|max}_{피처}
def _nb_columns(hops: int) -> list[str]:
    cols = []
    for h in range(1, hops + 1):
        for stat in ("mean", "max"):
            cols += [f"nb{h}_{stat}_{c}" for c in FEATURE_COLUMNS]
    return cols


def build_neighbor_features(
    labels: pd.DataFrame,
    edge_index: np.ndarray,
    pnu_to_idx: dict,
    parcels,
    buildings,
    *,
    hops: int = 2,
) -> pd.DataFrame:
    """★(A) per-t 시점정합 이웃집계 — 라벨행렬에 nb1·nb2 컬럼을 붙여 반환.

    각 라벨노드 v(시점 t_v)의 이웃 피처를 **t_v 스냅샷에서** 계산해 흡수한다(R1 누수
    차단: 2015 positive가 이웃의 2026 동결상태를 보지 않게). t별로 묶어 처리 — distinct
    t가 ~14뿐이라 스냅샷 비용 감당(수검 §6.5). hops=2면 B1(1홉)+B1+(2홉) 컬럼 동시 생성.

    입력: load_training_matrix의 labels·edge_index·pnu_to_idx + 원천 parcels·buildings
          (이웃엔 라벨 없는 노드도 포함되므로 features를 그때그때 계산해야 함).
    출력: labels + _nb_columns(hops). 같은 행 순서.
    """
    n_nodes = len(pnu_to_idx)
    idx_to_pnu = np.empty(n_nodes, dtype=object)
    for p, i in pnu_to_idx.items():
        idx_to_pnu[i] = p
    src_s, dst_s = _csr_neighbors(edge_index, n_nodes)
    f = len(FEATURE_COLUMNS)

    nb_cols = _nb_columns(hops)
    out = np.zeros((len(labels), len(nb_cols)), dtype=np.float64)
    lab_global = labels["pnu"].map(pnu_to_idx).to_numpy()    # 라벨행 → 전역 idx

    for t, grp in labels.groupby("t"):
        V = np.unique(grp["pnu"].map(pnu_to_idx).to_numpy())
        clo = _closure(V, src_s, dst_s, hops)                # t 스냅샷 노드 집합
        g2l = -np.ones(n_nodes, dtype=np.int64)
        g2l[clo] = np.arange(clo.size)
        # 폐포 내부 로컬 엣지(양 끝 모두 폐포)
        m = (g2l[src_s] >= 0) & (g2l[dst_s] >= 0)
        l_src, l_dst = g2l[src_s[m]], g2l[dst_s[m]]
        # 폐포 노드 피처 as-of t (라벨 없는 이웃도 — 그래서 node_features 재호출)
        rows = pd.DataFrame({"pnu": idx_to_pnu[clo], "t": int(t)})
        ft = node_features(rows, parcels, buildings)[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        agg1 = aggregate_once(ft, l_src, l_dst)              # [Nc, 2F] 1홉
        layers = [agg1]
        if hops >= 2:
            agg2 = aggregate_once(agg1[:, :f], l_src, l_dst)  # 집계의 집계(2홉)
            layers.append(agg2)
        stacked = np.concatenate(layers, axis=1)             # [Nc, 2F*hops]
        # 이 t의 라벨행을 폐포 로컬 idx로 찾아 기록
        rid = np.where(labels["t"].to_numpy() == t)[0]
        out[rid] = stacked[g2l[lab_global[rid]]]

    return pd.concat([labels.reset_index(drop=True), pd.DataFrame(out, columns=nb_cols)], axis=1)


def _load_parcels_buildings():
    """이웃집계가 필요로 하는 원천 둘만 적재(라벨 없는 이웃의 피처 계산용)."""
    buildings, _ = load_buildings(_vsizip(*_SRC["buildings"]), with_geometry=False)
    parcels, _ = load_parcels(_vsizip(*_SRC["parcels"]), training_sigungu_codes(), with_geometry=True)
    return parcels, buildings


def prepare_baseline_matrix(*, force_rebuild: bool = False, hops: int = 2) -> pd.DataFrame:
    """self + 이웃집계(B1·B1+) 증강행렬 — 캐시 우선. 모델 학습의 입력 진입점.

    캐시 부재 시: load_training_matrix(라벨·그래프) + parcels·buildings 적재 후
    build_neighbor_features(per-t). 한 번 굳히면 B0/B-2/B1/B1+/GNN 실행은 캐시만 읽음.
    """
    if not force_rebuild and _CACHE_NB.exists():
        return pd.read_parquet(_CACHE_NB)
    tm = load_training_matrix()
    parcels, buildings = _load_parcels_buildings()
    aug = build_neighbor_features(tm.labels, tm.edge_index, tm.pnu_to_idx, parcels, buildings, hops=hops)
    _PROCESSED.mkdir(parents=True, exist_ok=True)
    aug.to_parquet(_CACHE_NB, index=False)
    return aug


# ──────────────────────────────────────────────────────────────────────────
# 모델 — 대조군. 모두 같은 v1 피처·같은 fold(R9 공정). spatial_cv.evaluate가 채점.
# ──────────────────────────────────────────────────────────────────────────
def feature_sets(aug: pd.DataFrame) -> dict:
    """증강행렬에서 모델별 피처 컬럼 묶음. B1=self+1홉, B1+=self+1·2홉(R9 도달범위)."""
    nb1 = [c for c in aug.columns if c.startswith("nb1_")]
    nb2 = [c for c in aug.columns if c.startswith("nb2_")]
    return {"B1": list(FEATURE_COLUMNS) + nb1, "B1+": list(FEATURE_COLUMNS) + nb1 + nb2}


def aging_floor_predict(aug: pd.DataFrame):
    """B-2 동어반복 바닥선: 점수=aging 그대로(무학습). best-F1 임계가 컷을 정함.

    aging=0 positive는 어떤 양수 임계에서도 못 잡힌다 → 격전지 recall 정의상 0%
    (R9 0점선). PR-AUC는 'aging 단독 랭킹력' = 라벨링 규칙 되외우기의 점수.
    """
    aging = aug["aging"].to_numpy()
    return lambda train_idx, test_idx: aging[test_idx]


# XGBoost 하파 좁은 탐색(환경요청): depth{3,5}×lr{0.05,0.1} 4조합 수동.
_XGB_COMBOS = [(3, 0.05), (3, 0.1), (5, 0.05), (5, 0.1)]


def _make_xgb(depth, lr, *, spw):
    import xgboost as xgb
    # CPU hist(환경: GPU 없음). aucpr=불균형 직접 최적화. early stopping=과적합·CPU.
    return xgb.XGBClassifier(
        max_depth=depth, learning_rate=lr, n_estimators=600,
        tree_method="hist", n_jobs=-1, subsample=0.8, colsample_bytree=0.8,
        eval_metric="aucpr", early_stopping_rounds=30, scale_pos_weight=spw,
    )


def _spw(y) -> float:
    """scale_pos_weight=n_neg/n_pos. 실측 1.6 근처라 거의 1(R8 재프레이밍)."""
    pos = int((y == 1).sum())
    return (int((y == 0).sum()) / pos) if pos else 1.0


def _fit_predict_xgb(aug, feat_cols, tr_idx, va_idx, te_idx, depth, lr):
    """inner-train으로 학습(early stopping은 inner-val) → test 확률. 단일 적합."""
    X = aug[feat_cols].to_numpy(np.float32)
    y = aug["y"].to_numpy()
    m = _make_xgb(depth, lr, spw=_spw(y[tr_idx]))
    m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])], verbose=False)
    return m.predict_proba(X[te_idx])[:, 1], int(m.best_iteration or 0)


def run_xgb_cv(aug, feat_cols, edge_index, pnu_to_idx, *, model_name: str, cfg=None) -> dict:
    """B1/B1+ 전체 평가: inner k=3 하파선택(fold횡단 한세트 고정) → LODO 채점.

    ★하파는 outer fold마다 따로 고르지 않는다(과적합). 각 (fold,조합)의 inner k=3 평균
    PR-AUC를 구해 c*=argmax 한 조합 고정 → 그 c*로 각 outer fold를 per-fold early stopping
    으로 재학습(라운드만 fold별 조정, §5). 선정표를 리포트에 남김.
    """
    from redev.eval.spatial_cv import build_lodo_folds, evaluate, spatial_zone_groups
    cv = (cfg or load_graph_config())["cv"]
    folds = build_lodo_folds(aug, edge_index, pnu_to_idx, cfg=cfg)

    # ── 하파선택: 각 outer fold의 train 안 inner k=3 ──
    combo_scores = {c: [] for c in _XGB_COMBOS}
    for f in folds:
        groups = spatial_zone_groups(f.train_idx, aug, k=cv["inner_k_xgb"])
        for depth, lr in _XGB_COMBOS:
            aucs = []
            for i in range(len(groups)):
                va = groups[i]
                tr = np.concatenate([groups[j] for j in range(len(groups)) if j != i])
                p, _ = _fit_predict_xgb(aug, feat_cols, tr, va, va, depth, lr)
                aucs.append(_inner_prauc(aug["y"].to_numpy()[va], p))
            combo_scores[(depth, lr)].append(float(np.nanmean(aucs)))
    sel = {f"d{d}_lr{lr}": round(float(np.nanmean(s)), 4) for (d, lr), s in combo_scores.items()}
    best = max(_XGB_COMBOS, key=lambda c: np.nanmean(combo_scores[c]))

    # ── c* 고정 → outer fold별 per-fold early stopping(inner k=2의 1그룹=val) ──
    def predict_fn(train_idx, test_idx):
        g = spatial_zone_groups(train_idx, aug, k=2)
        va, tr = g[0], np.concatenate(g[1:]) if len(g) > 1 else (g[0], g[0])
        p, _ = _fit_predict_xgb(aug, feat_cols, tr, va, test_idx, best[0], best[1])
        return p

    rep = evaluate(predict_fn, folds, aug, model=model_name)
    rep["selection"] = {"combos_inner_prauc": sel, "chosen": f"depth={best[0]},lr={best[1]}"}
    return rep


def _inner_prauc(y, p) -> float:
    from redev.eval.metrics import pr_auc
    return pr_auc(y, p)


def region_grow(aging: np.ndarray, edge_index, pnu_to_idx, *, seed_cut, grow_cut, min_nodes=5) -> list:
    """B0 — 무학습 공간 바닥선(연기됐던 것, baseline.md). 노후 seed→인접 노후 필지 확장.

    노후도≥grow_cut 노드의 그래프 연결요소 중 seed(노후도≥seed_cut)를 품은 것 = 후보(BFS 확장과
    동치). R13 IoU 비교의 좌표축 — 학습 없이 공간 인접성만으로 어디까지 가나. 반환: PNU 집합 목록.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(aging)
    idx_to_pnu = np.empty(n, dtype=object)
    for p, i in pnu_to_idx.items():
        idx_to_pnu[i] = p
    grow = aging >= grow_cut
    src, dst = np.asarray(edge_index[0]), np.asarray(edge_index[1])
    keep = grow[src] & grow[dst]
    g = coo_matrix((np.ones(keep.sum()), (src[keep], dst[keep])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    seed = aging >= seed_cut
    clusters = []
    for c in np.unique(comp[grow]):
        members = np.where((comp == c) & grow)[0]
        if len(members) >= min_nodes and seed[members].any():
            clusters.append(set(idx_to_pnu[members].tolist()))
    return clusters

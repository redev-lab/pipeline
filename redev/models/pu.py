"""pu.py — v1.2 PU 학습 (uncertain 저가중 음성으로 '지정 vs 미지정' 대조 학습). 설계: pu_learning.md.

★함정(§0): uncertain은 음성이 아니라 '미래 양성 섞인 미지' — 점수 0으로 누르면 제품(미래 후보)을
죽인다. 목표는 순위 변별. ★타임박스 1세션: P1(저가중) 측정으로 채택/기각.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UNDESIGNATED_CUT = 0.5      # labels §4 노후미지정 컷(현재 노후도≥ → uncertain)


def load_pu_matrix():
    """uncertain 포함 PU 행렬 — positive(as-of-t)=기존 aug, uncertain(2026)=infer 캐시 슬라이스.

    ★재빌드 회피(concat). uncertain = 비라벨 노드 & 현재 노후도≥컷(labels §4와 동일 정의).
    반환: (pu[라벨+uncertain], aug[라벨만]).
    """
    from redev.models.baseline import _load_parcels_buildings, prepare_baseline_matrix
    aug = prepare_baseline_matrix()
    infer = pd.read_parquet("_data/processed/infer_features.parquet")
    unc = infer[~infer["pnu"].isin(set(aug["pnu"])) & (infer["aging"] >= UNDESIGNATED_CUT)].copy()
    parcels, _ = _load_parcels_buildings()
    cent = parcels.set_index("pnu").geometry.centroid
    unc["t"] = 2026
    unc["y"] = 0
    unc["certainty"] = "uncertain"
    unc["neg_reason"] = "not_yet"
    unc["zone_id"] = pd.NA
    unc["sigungu"] = unc["pnu"].str[:5]
    unc["centroid_x"] = unc["pnu"].map(cent.x).values
    unc["centroid_y"] = unc["pnu"].map(cent.y).values
    pu = pd.concat([aug, unc[aug.columns]], ignore_index=True)
    return pu, aug


def pu_weights(certainty: pd.Series, *, w: float) -> np.ndarray:
    """sample_weight — positive·reliable_neg=1.0, uncertain=w(저가중, §0 함정 대비 낮게)."""
    return np.where(certainty.to_numpy() == "uncertain", w, 1.0)


def _fit_pu(X, y, sw, *, depth=5, lr=0.1):
    """가중 XGBoost(early stopping 없이 고정 n_estimators — 가중 eval_set 복잡 회피)."""
    import xgboost as xgb
    m = xgb.XGBClassifier(max_depth=depth, learning_rate=lr, n_estimators=300,
                          tree_method="hist", n_jobs=-1, subsample=0.8, colsample_bytree=0.8,
                          eval_metric="aucpr")
    m.fit(X, y, sample_weight=sw)
    return m


def spy_promote(pu: pd.DataFrame, *, spy_frac=0.1, percentile=10, fc=None, seed=0) -> pd.DataFrame:
    """★P2 spy — positive 일부를 uncertain에 숨겨 학습 → spy 점수분포 하위 percentile을 임계로,
    그 아래(=진짜 음성스러운) uncertain만 reliable_negative 승격. 나머지 uncertain은 학습 제외(미지 존중).

    함정(§0) 완화: 모든 uncertain을 음성 가중(P1)이 아니라 *음성 확실한 것만* 음성으로.
    반환: 학습용 행렬(positive[spy 복원]+reliable_neg+승격 uncertain), uncertain 나머지는 제거.
    """
    from redev.models.baseline import production_feature_set
    fc = fc or production_feature_set(pu)
    rng = np.random.default_rng(seed)
    pos = pu.index[(pu["certainty"] == "positive")].to_numpy()
    unc = pu.index[(pu["certainty"] == "uncertain")].to_numpy()
    spy = rng.choice(pos, size=int(len(pos) * spy_frac), replace=False)
    # spy를 uncertain처럼(라벨 0) 두고 학습 → spy 점수로 임계
    tr = pu.drop(index=spy).copy()                          # spy 제외하고 1차 학습
    trU = pd.concat([tr[tr["certainty"] != "uncertain"], tr.loc[[i for i in unc]]])
    m = _fit_pu(trU[fc].to_numpy(np.float32),
                np.where(trU["certainty"].to_numpy() == "uncertain", 0, trU["y"].to_numpy()),
                np.ones(len(trU)))
    spy_scores = m.predict_proba(pu.loc[spy, fc].to_numpy(np.float32))[:, 1]
    thr = np.percentile(spy_scores, percentile)            # spy 하위 percentile = '음성스러움' 경계
    unc_scores = m.predict_proba(pu.loc[unc, fc].to_numpy(np.float32))[:, 1]
    promote = unc[unc_scores < thr]                        # 경계 아래 uncertain만 음성 승격
    keep = np.concatenate([pos, pu.index[pu["certainty"] == "reliable_negative"].to_numpy(), promote])
    out = pu.loc[keep].copy()
    out.loc[promote, "certainty"] = "reliable_negative"    # 승격
    return out, {"spy_thr": float(thr), "promoted": int(len(promote)), "unc_total": int(len(unc))}


def run_pu_lodo(pu: pd.DataFrame, edge_index, pnu_to_idx, *, w: float, fc=None) -> dict:
    """LODO — train(라벨+uncertain 저가중)/test(라벨만). ① recall·격전지·hard-FPR. 같은 장비."""
    from redev.eval.metrics import battleground_recall, best_f1, neg_split_report, pr_auc
    from redev.eval.spatial_cv import lodo_folds
    from redev.models.baseline import production_feature_set
    fc = fc or production_feature_set(pu)
    X = pu[fc].to_numpy(np.float32)
    y = pu["y"].to_numpy()
    cert = pu["certainty"]
    aging = pu["aging"].to_numpy()
    nr = pu["neg_reason"].to_numpy()
    all_p = np.full(len(pu), np.nan)
    for f in lodo_folds(pu):
        te = f.test_idx[cert.iloc[f.test_idx].to_numpy() != "uncertain"]   # 라벨 test만
        sw = pu_weights(cert.iloc[f.train_idx], w=w)
        m = _fit_pu(X[f.train_idx], y[f.train_idx], sw)
        all_p[te] = m.predict_proba(X[te])[:, 1]
    msk = ~np.isnan(all_p)
    yt, p = y[msk], all_p[msk]
    bf, bt = best_f1(yt, p)
    rec = float(((p >= bt) & (yt == 1)).sum() / max(1, (yt == 1).sum()))   # ① positive recall@best-F1
    bg, _ = battleground_recall(yt, p, aging[msk], bt)
    return {"w": w, "pr_auc": pr_auc(yt, p), "pos_recall": rec, "battleground": bg,
            "hard_fpr": neg_split_report(yt, p, nr[msk], bt)["hard_해제"]["fpr"], "thr": bt}


def pu_production_eval(pu: pd.DataFrame, infer_feats: pd.DataFrame, edge_index, pnu_to_idx, *, w: float, fc=None) -> dict:
    """전 라벨+uncertain 학습 → 전 노드 스코어링: ② 과대예측률·③ IoU·uncertain 분포(함정 가드)."""
    from redev.config import load_infer_config
    from redev.eval.iou import compare_methods
    from redev.models.baseline import production_feature_set
    from redev.models.infer import candidate_clusters, percentile_threshold
    fc = fc or production_feature_set(pu)
    m = _fit_pu(pu[fc].to_numpy(np.float32), pu["y"].to_numpy(), pu_weights(pu["certainty"], w=w))
    scores = m.predict_proba(infer_feats[fc].to_numpy(np.float32))[:, 1]
    # 운영 임계 = LODO best-F1(여기선 0.5 근사 — 방향 측정이라 고정컷도 가능). 과대예측=≥0.5
    over = float((scores >= 0.5).mean())
    cfg = load_infer_config()
    wide = candidate_clusters(scores, pnu_to_idx, edge_index, thr=0.5, min_nodes=cfg["cluster"]["min_nodes"])
    aug = pu[pu["certainty"] != "uncertain"]
    zones = {z: set(g["pnu"]) for z, g in aug[aug["y"] == 1].groupby("zone_id")}
    iou = compare_methods({"B1넓은": wide}, zones)["B1넓은"]
    # ★함정 가드: uncertain 점수 분포(전부 0 근처면 제품 죽인 것)
    unc_idx = pu["certainty"].to_numpy() == "uncertain"
    unc_pnu = pu.loc[unc_idx, "pnu"].to_numpy()
    if len(unc_pnu):                                   # baseline(uncertain無)이면 빈 배열 가드
        usub = infer_feats[infer_feats["pnu"].isin(set(unc_pnu))]
        unc_scores = m.predict_proba(usub[fc].to_numpy(np.float32))[:, 1]
        unc_pnu = usub["pnu"].to_numpy()
        unc_med, unc_top = float(np.median(unc_scores)), float(np.percentile(unc_scores, 90))
    else:
        unc_scores, unc_med, unc_top = np.array([]), float("nan"), float("nan")
    return {"w": w, "over_pred": over, "iou": iou["mean_iou"], "core": iou["mean_core_capture"],
            "unc_median": unc_med, "unc_top10_cut": unc_top,
            "unc_scores": unc_scores, "unc_pnu": unc_pnu, "model": m, "scores": scores}

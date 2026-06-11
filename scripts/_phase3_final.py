"""Phase 3 결승 — GNN+pretrain 리프트 + ★양성·음성 모두 zone-block resample bootstrap.
분석 전용. 예측을 .npy로 저장(이후 bootstrap 수정은 재학습 불요). → _data/processed/_final_*
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.cluster import KMeans

from redev.eval.metrics import pr_auc
from redev.models.baseline import (_load_parcels_buildings, feature_sets,
                                   load_training_matrix, prepare_baseline_matrix, run_xgb_cv)
from redev.models.gnn.train import build_district_tgroups, run_gnn_cv

PROC = "_data/processed"
tm = load_training_matrix()
aug = prepare_baseline_matrix()
fs = feature_sets(aug)
parcels, buildings = _load_parcels_buildings()
y = aug["y"].to_numpy()

# 예측 캐시 있으면 재사용(재학습 불요), 없으면 생성
pB1p_f, pGp_f = f"{PROC}/_pred_b1plus.npy", f"{PROC}/_pred_gnnpretrain.npy"
if os.path.exists(pB1p_f) and os.path.exists(pGp_f):
    pB1p, pGp = np.load(pB1p_f), np.load(pGp_f)
    print("예측 캐시 재사용")
else:
    rB1p = run_xgb_cv(aug, fs["B1+"], tm.edge_index, tm.pnu_to_idx, model_name="B1+")
    G = build_district_tgroups(aug, tm.edge_index, tm.pnu_to_idx, parcels, buildings, hops=2)
    t0 = time.time()
    rGp = run_gnn_cv(aug, tm.edge_index, tm.pnu_to_idx, parcels, buildings, tgroups=G,
                     pretrain=True, fixed_params={"hidden": 64, "dropout": 0.3, "lr": 0.01, "wd": 5e-4},
                     model_name="GNN+pretrain")
    print(f"GNN+pretrain(선택생략) {time.time()-t0:.1f}s, pooled PR-AUC {rGp['pooled']['pr_auc']:.3f}")
    pB1p, pGp = rB1p["all_p"], rGp["all_p"]
    np.save(pB1p_f, pB1p)
    np.save(pGp_f, pGp)

# ── ★양성·음성 모두 zone-block resample (음성=centroid KMeans 블록, zone 크기 맞춤) ──
zone = aug["zone_id"].to_numpy()
pos_mask = y == 1
zones = np.unique(zone[pos_mask])
zrows = {z: np.where(pos_mask & (zone == z))[0] for z in zones}
neg_rows = np.where(y == 0)[0]
avg_zone = pos_mask.sum() / len(zones)                       # 구역당 평균 필지(~367)
K_neg = int(round(len(neg_rows) / avg_zone))                # 음성 블록 수(~66)
xy = aug[["centroid_x", "centroid_y"]].to_numpy()
neg_lab = KMeans(n_clusters=K_neg, n_init=10, random_state=0).fit(xy[neg_rows]).labels_
nblocks = {b: neg_rows[neg_lab == b] for b in range(K_neg)}

rng = np.random.default_rng(0)
diffs = []
for _ in range(1000):
    pz = rng.choice(zones, size=len(zones), replace=True)
    nb = rng.choice(K_neg, size=K_neg, replace=True)
    rows = np.concatenate([zrows[z] for z in pz] + [nblocks[b] for b in nb])
    diffs.append(pr_auc(y[rows], pB1p[rows]) - pr_auc(y[rows], pGp[rows]))
ci = np.percentile(diffs, [2.5, 97.5])
inc0 = ci[0] <= 0 <= ci[1]

L = [
    f"양성·음성 모두 zone-block resample (양성 zone {len(zones)}개 + 음성 KMeans 블록 {K_neg}개)",
    f"Δ(B1+ − GNN+pretrain) PR-AUC: 중앙 {np.median(diffs):+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]",
    f"→ {'0 포함: 천장 미달이나 통계적으로 구분 불가(동급)' if inc0 else '0 불포함: 미세 열세, 유의(실질 무시가능)'}",
    "",
    f"(참고) 이전 음성-고정 CI [+0.0016, +0.0139] → 음성 resample 추가 후: [{ci[0]:+.4f}, {ci[1]:+.4f}]",
]
open(f"{PROC}/_final_ci.txt", "w", encoding="utf-8").write("\n".join(L))
json.dump({"ci": ci.tolist(), "median": float(np.median(diffs)), "includes_0": bool(inc0), "K_neg": K_neg},
          open(f"{PROC}/_final_ci.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n".join(L))
print("DONE")

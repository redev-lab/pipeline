import os, sys
sys.path.insert(0, os.getcwd()); sys.stdout.reconfigure(encoding="utf-8")
from redev.models.baseline import load_training_matrix, prepare_baseline_matrix, feature_sets, run_xgb_cv
tm = load_training_matrix(); aug = prepare_baseline_matrix(); fs = feature_sets(aug)
ei, pi = tm.edge_index, tm.pnu_to_idx
def drop(cols, *keys): return [c for c in cols if not any(k in c for k in keys)]
runs = {}
# 신구: v1(5)만 vs v1.1(10) — 같은 rig
runs["B1 v1(5피처)"] = run_xgb_cv(aug, drop(fs["B1"],"land","zoning","rail"), ei, pi, model_name="B1v1")
runs["B1 v1.1(10)"] = run_xgb_cv(aug, fs["B1"], ei, pi, model_name="B1v11")
runs["B1+ v1(5)"]   = run_xgb_cv(aug, drop(fs["B1+"],"land","zoning","rail"), ei, pi, model_name="B1pv1")
runs["B1+ v1.1(10)"]= run_xgb_cv(aug, fs["B1+"], ei, pi, model_name="B1pv11")
# leakage_ablation: v1.1에서 zoning·rail 각각 빼고
runs["B1 -용도지역"] = run_xgb_cv(aug, drop(fs["B1"],"zoning"), ei, pi, model_name="noz")
runs["B1 -역세권"]   = run_xgb_cv(aug, drop(fs["B1"],"rail"), ei, pi, model_name="nor")
runs["B1 -공시지가"] = run_xgb_cv(aug, drop(fs["B1"],"land"), ei, pi, model_name="nol")
L=[f'{"모델":<16}{"PR-AUC":>8}{"격전지":>8}  per-fold(성북/은평/구로/동작)']
import numpy as np
for name,r in runs.items():
    p=r["pooled"]; pf={x["fold"]:x["pr_auc"] for x in r["per_fold"]}
    L.append(f'{name:<16}{p["pr_auc"]:>8.3f}{p["battleground_recall"]:>8.3f}  '
             +"/".join(f'{pf.get(g,float("nan")):.3f}' for g in ["성북","은평","구로","동작"]))
open("_data/processed/_v11_rematch.txt","w",encoding="utf-8").write("\n".join(L))
print("\n".join(L)); print("DONE")

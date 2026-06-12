import os, sys, time
sys.path.insert(0, os.getcwd()); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from redev.models.baseline import load_training_matrix, prepare_baseline_matrix, _load_parcels_buildings
from redev.models.gnn.train import build_district_tgroups, run_gnn_cv
tm=load_training_matrix(); aug=prepare_baseline_matrix()
parcels,buildings=_load_parcels_buildings()
G=build_district_tgroups(aug, tm.edge_index, tm.pnu_to_idx, parcels, buildings, hops=2)
t0=time.time()
rep=run_gnn_cv(aug, tm.edge_index, tm.pnu_to_idx, parcels, buildings, tgroups=G,
               fixed_params={"hidden":64,"dropout":0.3,"lr":0.01,"wd":5e-4}, model_name="GNN v1.1")
p=rep["pooled"]; pf={x["fold"]:x["pr_auc"] for x in rep["per_fold"]}
L=[f'GNN v1.1(10피처) {time.time()-t0:.0f}s',
   f'  PR-AUC {p["pr_auc"]:.3f} (v1 GNN 0.929) | 격전지 {p["battleground_recall"]:.3f} (v1 0.606)',
   f'  per-fold 성북/은평/구로/동작: '+"/".join(f'{pf.get(g,0):.3f}' for g in ["성북","은평","구로","동작"]),
   f'  (천장 B1+ v1.1 0.937/격전지0.683 — GNN이 넘나?)']
open("_data/processed/_v11_gnn.txt","w",encoding="utf-8").write("\n".join(L))
print("\n".join(L)); print("DONE")

import os, sys
sys.path.insert(0, os.getcwd()); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from redev.config import load_infer_config
from redev.models.baseline import load_training_matrix, prepare_baseline_matrix, _load_parcels_buildings, region_grow
from redev.models.infer import build_all_node_features, train_production_b1, operating_threshold, score_all, candidate_clusters, percentile_threshold
from redev.eval.iou import compare_methods
tm=load_training_matrix(); aug=prepare_baseline_matrix()
parcels,buildings=_load_parcels_buildings()
allf=build_all_node_features(parcels,buildings,tm.pnu_to_idx,tm.edge_index)  # 10피처 재빌드
allf.to_parquet("_data/processed/infer_features.parquet")
thr=operating_threshold(aug,tm.edge_index,tm.pnu_to_idx)
model,fc=train_production_b1(aug); scores=score_all(model,allf,fc)
L=[f'★infer 과대예측: ≥thr({thr:.3f}) {(scores>=thr).sum()} / {len(scores)} = {100*(scores>=thr).mean():.0f}% (v1 81%)',
   f'  중앙 확률 {np.median(scores):.3f} (v1 0.977)']
cfg=load_infer_config()
wide=candidate_clusters(scores,tm.pnu_to_idx,tm.edge_index,thr=thr,min_nodes=cfg["cluster"]["min_nodes"])
tcut=percentile_threshold(scores,top_pct=cfg["cluster"]["tight_top_pct"])
tight=candidate_clusters(scores,tm.pnu_to_idx,tm.edge_index,thr=tcut,min_nodes=cfg["cluster"]["min_nodes"])
rg=cfg["region_grow"]
b0=region_grow(allf["aging"].to_numpy(),tm.edge_index,tm.pnu_to_idx,seed_cut=rg["seed_cut"],grow_cut=rg["grow_cut"],min_nodes=cfg["cluster"]["min_nodes"])
zones={z:set(g["pnu"]) for z,g in aug[aug.y==1].groupby("zone_id")}
res=compare_methods({"B1넓은":wide,"B1타이트":tight,"B0":b0},zones)
L.append("IoU 4종(v1: 넓은0.294/타이트0.019/B0 0.129):")
for m,r in res.items(): L.append(f'  {m:<10} IoU {r["mean_iou"]:.3f} 핵심부 {r["mean_core_capture"]:.3f}')
open("_data/processed/_v11_infer.txt","w",encoding="utf-8").write("\n".join(L))
print("\n".join(L)); print("DONE")

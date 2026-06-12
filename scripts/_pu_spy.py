import os, sys
sys.path.insert(0, os.getcwd()); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from redev.models.baseline import load_training_matrix, production_feature_set
from redev.models.pu import load_pu_matrix, run_pu_lodo, pu_production_eval, spy_promote, UNDESIGNATED_CUT
tm=load_training_matrix(); pu,aug=load_pu_matrix()
infer=pd.read_parquet("_data/processed/infer_features.parquet")
ei,pi=tm.edge_index,tm.pnu_to_idx; fc=production_feature_set(pu)
out,info=spy_promote(pu, spy_frac=0.1, percentile=10, fc=fc)
print(f"spy 임계 {info['spy_thr']:.3f} | uncertain {info['unc_total']} 중 음성 승격 {info['promoted']} ({100*info['promoted']/info['unc_total']:.0f}%), 나머지 학습제외")
lo=run_pu_lodo(out, ei, pi, w=1.0, fc=fc)
pr=pu_production_eval(out, infer, ei, pi, w=1.0, fc=fc)
# 원래 uncertain 전체를 spy 모델로 점수(함정 가드)
unc_pnu=pu.loc[pu.certainty=="uncertain","pnu"].to_numpy()
usub=infer[infer.pnu.isin(set(unc_pnu))]
us=pr["model"].predict_proba(usub[fc].to_numpy(np.float32))[:,1]
L=[f"{'구성':<16}{'pos_rec':>8}{'격전지':>7}{'과대예측':>8}{'IoU':>6}{'unc중앙':>8}{'unc상위10%':>9}",
   f"{'v1.1 baseline':<16}{0.830:>8.3f}{0.712:>7.3f}{'81%':>8}{0.300:>6.3f}{'—':>8}{'—':>9}",
   f"{'P2 spy승격':<16}{lo['pos_recall']:>8.3f}{lo['battleground']:>7.3f}{100*pr['over_pred']:>7.0f}%{pr['iou']:>6.3f}{np.median(us):>8.3f}{np.percentile(us,90):>9.3f}"]
ok1=(0.830-lo['pos_recall'])<=0.03; ok3=pr['iou']>0.300
L.append(f"\n채택: ①recall 하락 {100*(0.830-lo['pos_recall']):.1f}%p(≤3) ③IoU {pr['iou']:.3f} vs 0.300")
L.append(f"  → ①{'합격' if ok1 else '불합격'} ③{'개선' if ok3 else '미개선'} → {'production 교체' if ok1 and ok3 else 'PU(P1·P2) 불충분 → 라벨확장(v2)이 답, Phase 8로'}")
open("_data/processed/_pu_spy.txt","w",encoding="utf-8").write("\n".join(L))
print("\n".join(L)); print("DONE")

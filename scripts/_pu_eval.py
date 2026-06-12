import os, sys
sys.path.insert(0, os.getcwd()); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from redev.models.baseline import load_training_matrix, _load_parcels_buildings
from redev.models.pu import load_pu_matrix, run_pu_lodo, pu_production_eval
tm = load_training_matrix()
pu, aug = load_pu_matrix()
infer = pd.read_parquet("_data/processed/infer_features.parquet")
parcels, _ = _load_parcels_buildings()
ei, pi = tm.edge_index, tm.pnu_to_idx
print(f"PU 행렬 {len(pu)} (라벨 {len(aug)} + uncertain {len(pu)-len(aug)})")
L=[f"{'구성':<16}{'PR-AUC':>7}{'pos_rec':>8}{'격전지':>7}{'과대예측':>8}{'IoU':>6}{'unc중앙':>8}{'unc상위10%':>9}"]
def row(name, lodo, prod):
    L.append(f"{name:<16}{lodo['pr_auc']:>7.3f}{lodo['pos_recall']:>8.3f}{lodo['battleground']:>7.3f}"
             f"{100*prod['over_pred']:>7.0f}%{prod['iou']:>6.3f}{prod['unc_median']:>8.3f}{prod['unc_top10_cut']:>9.3f}")
# baseline = v1.1(uncertain 없음): aug만
bl_lodo = run_pu_lodo(aug, ei, pi, w=1.0)
bl_prod = pu_production_eval(aug, infer, ei, pi, w=1.0)
row("v1.1(uncertain無)", bl_lodo, bl_prod)
best=None
for w in (0.1, 0.3, 0.5):
    lo = run_pu_lodo(pu, ei, pi, w=w)
    pr = pu_production_eval(pu, infer, ei, pi, w=w)
    row(f"PU w={w}", lo, pr)
    if best is None or pr["over_pred"] < best[1]["over_pred"]: best=(w, pr, lo)
# ★함정 가드: best w의 uncertain 상위권 표본 5개 육안(미래후보군)
w,pr,lo = best
us=pr["unc_scores"]; up=pr["unc_pnu"]; order=np.argsort(-us)[:5]
dong=parcels.set_index("pnu")["dong_addr"]
L.append(f"\n★uncertain 상위권 표본(w={w}, 미래후보군 — 전부 0근처면 제품 죽인 것):")
for i in order:
    L.append(f"  {up[i][-10:]} 점수{us[i]:.3f} {str(dong.get(up[i],'?')).split()[-1] if pd.notna(dong.get(up[i])) else '?'}")
# 채택 규칙 판정
L.append(f"\n채택규칙: ①recall 하락 {100*(bl_lodo['pos_recall']-lo['pos_recall']):.1f}%p(≤3합격) + ③IoU {pr['iou']:.3f} vs 0.300")
ok1 = (bl_lodo['pos_recall']-lo['pos_recall'])<=0.03
ok3 = pr['iou']>0.300
L.append(f"  → ①{'합격' if ok1 else '불합격'} ③{'개선' if ok3 else '미개선'} → {'production 교체 검토' if (ok1 and ok3) else 'PU v1 불충분, 라벨확장(v2)이 답 — Phase 8로'}")
open("_data/processed/_pu_eval.txt","w",encoding="utf-8").write("\n".join(L))
print("\n".join(L)); print("DONE")

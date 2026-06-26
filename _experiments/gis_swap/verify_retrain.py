"""#3-c 검증: 닮은동네 유지(재학습 무관) + 데모 점수 + 동작 skew 해소(national-train vs seoul-train)."""
import sys, io, pickle, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd, numpy as np
from redev.retrieval.case_search import search_cases

new = pickle.load(open("_data/processed/serve_ctx.pkl", "rb"))   # national-served, national-trained
# [1] 닮은동네 타당성 유지 (재학습은 zone_vectors 불변 — 모델만 바뀜)
DEMOS = {"정릉동(성북)": "11290108", "노량진(동작)": "11590101", "응암(은평)": "11380107"}
print("[1] 닮은동네 타당성(재학습 후) — top5 코사인·1위 sim")
for name, pre in DEMOS.items():
    pn = [p for p in new.parcels["pnu"] if str(p).startswith(pre)][:35]
    if len(pn) < 5: continue
    r = search_cases(pn, new.parcels, new.buildings, new.zone_vectors, k=5)["matches"]
    print(f"   {name}: top5 코사인 평균 {np.mean([m['similarity'] for m in r]):.3f} · 1위 sim {r[0]['similarity']:.3f} · 1위 {r[0]['display_name'][:14]}")

# [2] 데모 환경점수 + candidate (new infer_scores)
sc = pd.read_parquet("_data/processed/infer_scores.parquet")
pcol = "score_pct" if "score_pct" in sc else [c for c in sc.columns if "pct" in c][0]
sc = sc.set_index(sc["pnu"].astype(str))
print(f"\n[2] 데모 환경점수(상위%) — 전노드 {len(sc):,}")
for name, pre in DEMOS.items():
    sub = sc[sc.index.str.startswith(pre)][pcol]
    if len(sub): print(f"   {name}: 상위 {(1-sub.median())*100:.0f}% (중앙) · 상위10% 필지 {(sub>0.9).sum()}/{len(sub)}")

# [3] 동작 skew 해소: national-trained vs seoul-trained(backup) score_pct 비교
od = pd.read_parquet("_data/processed/infer_scores.parquet.nat_seoultrain"); od = od.set_index(od["pnu"].astype(str))[pcol]
dj_new = sc[sc.index.str.startswith("11590")][pcol]; dj_old = od[od.index.str.startswith("11590")]
common = dj_new.index.intersection(dj_old.index)
from scipy.stats import spearmanr
print(f"\n[3] 동작 재학습 영향(national-train vs seoul-train): Spearman {spearmanr(dj_new.reindex(common), dj_old.reindex(common)).correlation:.3f}")
print(f"   동작 상위10% candidate: seoul-train {int((dj_old>0.9).sum())} → national-train {int((dj_new>0.9).sum())} (재학습이 candidate 재배치)")
print("VERIFY_DONE")

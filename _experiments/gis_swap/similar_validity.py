"""national 닮은동네 절대 타당성: 질의-결과 코사인 유사도 + 1위 안정 + 다름 vs 틀림. 현 산출물 읽기만."""
import sys, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd, numpy as np
from redev.retrieval.case_search import search_cases

new = pickle.load(open("_data/processed/serve_ctx.pkl", "rb"))            # national+backfill
old = pickle.load(open("_data/processed/serve_ctx.pkl.seoul_bak", "rb"))  # 서울
GU = {"11170": "용산", "11290": "성북", "11380": "은평", "11440": "마포", "11530": "구로", "11590": "동작", "11680": "강남"}
pz = new.pnu_zone; z2p = {}
for p, z in (pz.items() if isinstance(pz, dict) else zip(pz.index, pz)):
    z2p.setdefault(z, []).append(p)

rows = []
for zid, pnus in z2p.items():
    if len(pnus) < 5:
        continue
    rn = search_cases(pnus[:40], new.parcels, new.buildings, new.zone_vectors, k=5)["matches"]
    ro = search_cases(pnus[:40], old.parcels, old.buildings, old.zone_vectors, k=5)["matches"]
    ni = [m["zone_id"] for m in rn]; oi = [m["zone_id"] for m in ro]
    nsim = [m["similarity"] for m in rn]; osim = [m["similarity"] for m in ro]
    # 첫 불일치 순위(1..5), 없으면 6
    div = next((i + 1 for i in range(5) if i >= len(ni) or i >= len(oi) or ni[i] != oi[i]), 6)
    rows.append({"gu": GU.get(zid[:5], zid[:5]),
                 "nat_top1_sim": nsim[0], "seo_top1_sim": osim[0],
                 "nat_top5_mean": np.mean(nsim), "seo_top5_mean": np.mean(osim),
                 "nat_top5_min": min(nsim), "top1_same": ni[0] == oi[0], "div_rank": div})
d = pd.DataFrame(rows)
print(f"=== national 닮은동네 절대 타당성 (지정 {len(d)}구역) ===")
print(f"[1] 질의-결과 코사인 — national top5 평균 {d['nat_top5_mean'].mean():.3f} (min {d['nat_top5_min'].mean():.3f}) vs 서울 {d['seo_top5_mean'].mean():.3f}")
print(f"    → national top5가 질의와 {'높게 유사(타당)' if d['nat_top5_mean'].mean()>0.9 else '유사도 낮음(품질의심)'}")
print(f"[2] 다름 vs 틀림 — national 1위 sim {d['nat_top1_sim'].mean():.3f} vs 서울 1위 sim {d['seo_top1_sim'].mean():.3f}")
worse = (d["nat_top1_sim"] < d["seo_top1_sim"] - 0.02).mean()
print(f"    national 1위가 서울보다 유의히 먼(>0.02 낮은) 구역 비율 {worse*100:.0f}% (낮을수록 '다름이지 틀림 아님')")
print(f"[3] 1위 안정 {d['top1_same'].mean()*100:.0f}% · 첫 불일치 순위 중앙 {d['div_rank'].median():.0f} (분포: {dict(d['div_rank'].value_counts().sort_index())})")
print(f"    → 불일치가 주로 {'4-5위(꼬리, 무해)' if d['div_rank'].median()>=4 else '1-2위(상위, 유의)'}")
print(f"\n[동작 12구역] national top5 평균 {d[d.gu=='동작']['nat_top5_mean'].mean():.3f} · 1위sim nat {d[d.gu=='동작']['nat_top1_sim'].mean():.3f}/서울 {d[d.gu=='동작']['seo_top1_sim'].mean():.3f} · 1위안정 {d[d.gu=='동작']['top1_same'].mean()*100:.0f}% · 불일치순위중앙 {d[d.gu=='동작']['div_rank'].median():.0f}")

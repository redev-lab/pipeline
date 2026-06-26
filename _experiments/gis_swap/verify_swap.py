"""#4 재현 검증: 새 national serve_ctx vs 서울 백업 — 닮은동네 robust + 노후도 일치. _experiments 전용."""
import sys, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd
from redev.retrieval.case_search import search_cases
from redev.rules.stage1 import cluster_metrics
from redev.config import load_thresholds

th = load_thresholds()
new = pickle.load(open("_data/processed/serve_ctx.pkl", "rb"))            # national(신규)
old = pickle.load(open("_data/processed/serve_ctx.pkl.seoul_bak", "rb"))  # 서울(백업)
print(f"new(national) buildings {len(new.buildings):,} · old(서울) buildings {len(old.buildings):,}")

DEMOS = {"정릉동(성북)": "11290108", "노량진(동작)": "11590101", "응암(은평)": "11380107", "역삼(강남)": "11680101"}
ov_sum = top1_ok = n = 0
for name, pre in DEMOS.items():
    pn = [p for p in new.parcels["pnu"] if str(p).startswith(pre)][:35]
    if len(pn) < 5:
        print(f"{name}: 필지 부족 건너뜀"); continue
    rn = search_cases(pn, new.parcels, new.buildings, new.zone_vectors, k=5)
    ro = search_cases(pn, old.parcels, old.buildings, old.zone_vectors, k=5)
    ni = [m["zone_id"] for m in rn["matches"]]; oi = [m["zone_id"] for m in ro["matches"]]
    inter = len(set(ni) & set(oi)); t1 = ni[0] == oi[0]; n += 1; ov_sum += inter; top1_ok += t1
    mn = cluster_metrics(pn, new.parcels, new.buildings, cfg=th); mo = cluster_metrics(pn, old.parcels, old.buildings, cfg=th)
    print(f"\n=== {name} ({len(pn)}필지) ===")
    print(f"  닮은동네 top5 교집합 {inter}/5 · 1위 {'동일' if t1 else '★뒤바뀜'}")
    print(f"  노후도 서울 {mo['old_area_ratio']:.3f} → national {mn['old_area_ratio']:.3f} (차 {abs((mn['old_area_ratio'] or 0)-(mo['old_area_ratio'] or 0))*100:.2f}%p)")
    print(f"  national top5: {[m['display_name'][:13] for m in rn['matches']]}")
print(f"\n=== 종합: {n}개 데모 · 1위 유지 {top1_ok}/{n} · 평균 교집합 {ov_sum/max(n,1):.1f}/5 ===")

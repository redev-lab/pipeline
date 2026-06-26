"""zone_vectors national 재구축 영향: 타깃(재개발구역) vs 비타깃(강남) 분리. 현 national 산출물 유지(읽기만)."""
import sys, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
from redev.retrieval.case_search import search_cases
from redev.rules.stage1 import cluster_metrics
from redev.config import load_thresholds

th = load_thresholds()
new = pickle.load(open("_data/processed/serve_ctx.pkl", "rb"))            # national
old = pickle.load(open("_data/processed/serve_ctx.pkl.seoul_bak", "rb"))  # 서울
GU = {"11170": "용산", "11290": "성북", "11380": "은평", "11440": "마포", "11530": "구로", "11590": "동작", "11680": "강남"}

# zone→PNU (new ctx pnu_zone)
pz = new.pnu_zone
z2p = {}
for p, z in (pz.items() if isinstance(pz, dict) else zip(pz.index, pz)):
    z2p.setdefault(z, []).append(p)

# ── #1 지정구역 닮은동네 robustness (national zv vs 서울 zv), 구별 ──
print("[#1] 지정 재개발구역 닮은동네 robustness (national vs 서울)")
rows = []
for zid, pnus in z2p.items():
    if len(pnus) < 5:
        continue
    gu = GU.get(zid[:5], zid[:5])
    rn = search_cases(pnus[:40], new.parcels, new.buildings, new.zone_vectors, k=5)
    ro = search_cases(pnus[:40], old.parcels, old.buildings, old.zone_vectors, k=5)
    ni = [m["zone_id"] for m in rn["matches"]]; oi = [m["zone_id"] for m in ro["matches"]]
    rows.append({"gu": gu, "inter": len(set(ni) & set(oi)), "top1": ni[0] == oi[0]})
d = pd.DataFrame(rows)
print(f"  전체 지정구역 {len(d)}개: 평균 교집합 {d['inter'].mean():.1f}/5 · 1위 유지 {d['top1'].mean()*100:.0f}%")
for gu, g in d.groupby("gu"):
    tgt = "비타깃" if gu == "강남" else "타깃"
    print(f"   [{gu}/{tgt}] {len(g)}구역: 교집합 {g['inter'].mean():.1f}/5 · 1위 유지 {g['top1'].mean()*100:.0f}%")

# ── #2 zone_vectors 변동: 어느 기준구역 벡터가 크게 변했나 ──
print("\n[#2] zone_vectors 벡터 변동 (national vs 서울, 표준화벡터 L2)")
om = {m["zone_id"]: i for i, m in enumerate(old.zone_vectors.meta)}
shifts = []
for i, m in enumerate(new.zone_vectors.meta):
    if m["zone_id"] in om:
        diff = float(np.linalg.norm(new.zone_vectors.Z[i] - old.zone_vectors.Z[om[m["zone_id"]]]))
        shifts.append({"zone": m["display_name"][:18], "gu": GU.get(m["zone_id"][:5], m["zone_id"][:5]), "shift": diff})
sd = pd.DataFrame(shifts).sort_values("shift", ascending=False)
print("  변동 상위 6 기준구역:")
for _, r in sd.head(6).iterrows():
    print(f"   {r['shift']:.2f}  {r['gu']:5} {r['zone']}")
print("  구별 평균 벡터변동:", {k: round(v, 2) for k, v in sd.groupby("gu")["shift"].mean().sort_values(ascending=False).items()})

# ── #3 강남(역삼) 노후도 + backfill 손실 ──
print("\n[#3] 강남 노후도·커버리지")
bf = pd.read_csv("_experiments/gis_swap/backfill_pnus.csv", dtype={"pnu": str})
bf = bf[bf["is_res"] == True] if bf["is_res"].dtype != bool else bf[bf["is_res"]]
print("  강남(11680) 실주거 손실 PNU:", (bf["gu"].astype(str) == "11680").sum(), f"(전체 {len(bf)})")
yeoksam = [p for p in new.parcels["pnu"] if str(p).startswith("11680101")][:35]
mn = cluster_metrics(yeoksam, new.parcels, new.buildings, cfg=th); mo = cluster_metrics(yeoksam, old.parcels, old.buildings, cfg=th)
print(f"  역삼 노후도 서울 {mo['old_area_ratio']:.3f} → national {mn['old_area_ratio']:.3f} (둘 다 낮음=신축 → 작은 절대차도 코사인 순위 흔듦)")

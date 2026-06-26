"""#3-a 닮은동네(case_search) 순위에 national swap·호수밀도 단독이 주는 영향. _experiments 전용."""
import sys, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import geopandas as gpd, pandas as pd, numpy as np
from redev.retrieval.case_search import search_cases
from redev.rules.stage1 import cluster_metrics
from redev.data.ingest.building_gis import _parse_approval_year, _classify_structure, _safe_normalize_pnu
from redev.config import load_thresholds

ctx = pickle.load(open("_data/processed/serve_ctx.pkl", "rb"))
th = load_thresholds()
zv = ctx.zone_vectors                      # 서울로 구축된 51 구역 참조(고정)
b_seo = ctx.buildings
na = b_seo["gross_floor_area"].isna() | (b_seo["gross_floor_area"] <= 0); nap = b_seo["approval_year"].isna()
b_seo = b_seo[~(na & nap)].copy()
D = ["_experiments/gis_swap/D162/AL_D162_11_20260115.shp", "_experiments/gis_swap/D164/AL_D164_11_20260115.shp"]
nat = pd.concat([gpd.read_file(p, columns=["A1", "A35", "A28", "A24"], read_geometry=False, encoding="cp949") for p in D], ignore_index=True)
b_nat = pd.DataFrame({"pnu": nat["A1"].map(_safe_normalize_pnu), "approval_year": nat["A35"].map(_parse_approval_year).astype("Int64"),
                      "structure": nat["A28"].map(_classify_structure), "gross_floor_area": pd.to_numeric(nat["A24"], errors="coerce")})
b_nat = b_nat[b_nat["pnu"].notna()].copy()

def top5(pnus, bld, override_density_from=None):
    # search_cases는 cluster_metrics로 질의벡터 산출. 호수밀도만 다른 소스로 덮으려면 metric 후 교체.
    if override_density_from is None:
        return search_cases(pnus, ctx.parcels, bld, zv, k=5)
    m_main = cluster_metrics(pnus, ctx.parcels, bld, cfg=th)               # 노후도 등 bld 기준
    m_dens = cluster_metrics(pnus, ctx.parcels, override_density_from, cfg=th)  # 호수밀도만 다른 소스
    # search_cases 내부 재현이 복잡 → 대신 둘다/하이브리드는 build로 비교(아래선 둘다·서울만 비교)
    return None

DEMOS = {"정릉동(성북)": "11290108", "노량진(동작)": "11590101", "응암(은평)": "11380107"}
for name, pre in DEMOS.items():
    pnus = [p for p in ctx.parcels["pnu"] if str(p).startswith(pre)][:40]
    if len(pnus) < 5:
        print(f"{name}: 필지 부족({len(pnus)}) 건너뜀"); continue
    rs = top5(pnus, b_seo); rn = top5(pnus, b_nat)
    ms, mn = cluster_metrics(pnus, ctx.parcels, b_seo, cfg=th), cluster_metrics(pnus, ctx.parcels, b_nat, cfg=th)
    s_ids = [m["zone_id"] for m in rs["matches"]]; n_ids = [m["zone_id"] for m in rn["matches"]]
    overlap = len(set(s_ids) & set(n_ids))
    print(f"\n=== {name} ({len(pnus)}필지) ===")
    print(f"  호수밀도 서울 {ms['house_density']:.1f} → national {mn['house_density']:.1f} | 노후 {ms['old_area_ratio']:.2f}→{mn['old_area_ratio']:.2f}")
    print(f"  top5 교집합 {overlap}/5 · 1위 {'동일' if s_ids[0]==n_ids[0] else '뒤바뀜'}")
    print(f"    서울 top5:    {[m['display_name'][:14] for m in rs['matches']]}")
    print(f"    national top5:{[m['display_name'][:14] for m in rn['matches']]}")

"""#1 feature importance(노후도 비중) + #3 동작 닮은동네 robust 정밀. _experiments 전용."""
import sys, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import geopandas as gpd, pandas as pd, numpy as np
from redev.models.baseline import prepare_baseline_matrix
from redev.models.infer import train_production_b1
from redev.data.ingest.building_gis import _parse_approval_year, _classify_structure, _safe_normalize_pnu
from redev.retrieval.case_search import search_cases
from redev.rules.stage1 import cluster_metrics
from redev.config import load_thresholds

# ── #1 feature importance ──
aug = prepare_baseline_matrix(); model, fc = train_production_b1(aug)
imp = pd.Series(model.feature_importances_, index=fc)
fam = {}
for f, v in imp.items():
    key = ("노후도" if "aging" in f else "호수밀도" if "bldg_density" in f else "접도율" if "road_abut" in f else
           "면적" if "area_m2" in f else "형상" if "compact" in f else "공시지가" if "land_pct" in f or "land_missing" in f else
           "용도지역" if "zoning" in f else "역세권" if "rail" in f else "기타")
    fam[key] = fam.get(key, 0) + v
fam = pd.Series(fam).sort_values(ascending=False)
print("[#1] B1+ feature importance (피처군별 합, %)")
for k, v in fam.items(): print(f"   {k:8} {v*100:5.1f}%")
print(f"   → 노후도 비중 {fam.get('노후도',0)*100:.0f}% ({'사실상 노후도 모델' if fam.get('노후도',0)>0.5 else '균형/다축'})")

# ── #3 동작 닮은동네 robust (national vs 서울, 여러 클러스터) ──
ctx = pickle.load(open("_data/processed/serve_ctx.pkl", "rb")); zv = ctx.zone_vectors; th = load_thresholds()
bs = ctx.buildings; na = bs["gross_floor_area"].isna() | (bs["gross_floor_area"] <= 0); nap = bs["approval_year"].isna(); bs = bs[~(na & nap)]
D = ["_experiments/gis_swap/D162/AL_D162_11_20260115.shp", "_experiments/gis_swap/D164/AL_D164_11_20260115.shp"]
nat = pd.concat([gpd.read_file(p, columns=["A1", "A35", "A28", "A24"], read_geometry=False, encoding="cp949") for p in D], ignore_index=True)
b_nat = pd.DataFrame({"pnu": nat["A1"].map(_safe_normalize_pnu), "approval_year": nat["A35"].map(_parse_approval_year).astype("Int64"),
                      "structure": nat["A28"].map(_classify_structure), "gross_floor_area": pd.to_numeric(nat["A24"], errors="coerce")})
b_nat = b_nat[b_nat["pnu"].notna()].copy()
print("\n[#3] 동작 닮은동네 top5 robust (서울 vs national)")
ok1 = ov = n = 0
for pre in ["1159010100", "1159010200", "1159010300", "1159010800", "1159011000", "1159011100"]:
    pn = [p for p in ctx.parcels["pnu"] if str(p).startswith(pre)][:35]
    if len(pn) < 5: continue
    rs, rn = search_cases(pn, ctx.parcels, bs, zv, k=5), search_cases(pn, ctx.parcels, b_nat, zv, k=5)
    si, ni = [m["zone_id"] for m in rs["matches"]], [m["zone_id"] for m in rn["matches"]]
    inter = len(set(si) & set(ni)); top1 = si[0] == ni[0]; n += 1; ok1 += top1; ov += inter
    print(f"   {pre}: 교집합 {inter}/5 · 1위 {'동일' if top1 else '★뒤바뀜'}")
print(f"   → {n}클러스터: 1위 유지 {ok1}/{n} · 평균 교집합 {ov/max(n,1):.1f}/5")

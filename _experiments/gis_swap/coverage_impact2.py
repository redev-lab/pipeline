"""#2 동작 손실필지 환경점수 추락(깨지는 주소율) + 닮은동네 robust 재확인. _experiments 전용."""
import sys, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import geopandas as gpd, pandas as pd, numpy as np
from redev.models.baseline import _SRC, _vsizip, prepare_baseline_matrix, build_neighbor_features
from redev.data.ingest.parcels import load_parcels
from redev.data.ingest.building_gis import load_buildings, _parse_approval_year, _classify_structure, _safe_normalize_pnu
from redev.graph.build import build_graph
from redev.graph.features import node_features
from redev.models.infer import train_production_b1, score_all
from redev.retrieval.case_search import search_cases
from redev.rules.stage1 import cluster_metrics
from redev.config import load_thresholds

th = load_thresholds()
D = ["_experiments/gis_swap/D162/AL_D162_11_20260115.shp", "_experiments/gis_swap/D164/AL_D164_11_20260115.shp"]

def nat_b():
    fr = []
    for p in D:
        g = gpd.read_file(p, columns=["A1", "A35", "A28", "A24"], read_geometry=False, encoding="cp949")
        fr.append(pd.DataFrame({"pnu": g["A1"].map(_safe_normalize_pnu), "approval_year": g["A35"].map(_parse_approval_year).astype("Int64"),
                                "structure": g["A28"].map(_classify_structure), "gross_floor_area": pd.to_numeric(g["A24"], errors="coerce")}))
    df = pd.concat(fr, ignore_index=True); df = df[df["pnu"].notna()].copy()
    na = df["gross_floor_area"].isna() | (df["gross_floor_area"] <= 0); nap = df["approval_year"].isna()
    df = df[~(na & nap)].copy(); df["land_div"] = None; df["sigungu"] = df["pnu"].str[:5]; return df

aug = prepare_baseline_matrix(); model, fc = train_production_b1(aug)
AG = [c for c in fc if "aging" in c]
parcels, _ = load_parcels(_vsizip(*_SRC["parcels"]), ["11590"], with_geometry=True)   # 동작(최악 커버리지)
b_seo, _ = load_buildings(_vsizip(*_SRC["buildings"]), with_geometry=False)
b_nat = nat_b()
gp = parcels[parcels["sigungu"] == "11590"].copy()
graph, p2i, _ = build_graph(gp); idx = np.empty(len(p2i), dtype=object)
for p, i in p2i.items(): idx[i] = p
base = pd.DataFrame({"pnu": idx, "t": 2026})
fs = build_neighbor_features(node_features(base, gp, b_seo), graph.edge_index, p2i, gp, b_seo, hops=2)
fn = build_neighbor_features(node_features(base, gp, b_nat), graph.edge_index, p2i, gp, b_nat, hops=2)
ss = pd.Series(score_all(model, fs, fc), index=fs["pnu"]).rank(pct=True)
sn = pd.Series(score_all(model, fn, fc), index=fn["pnu"]).rank(pct=True)

# 손실 필지 = 서울 aging>0인데 national 건물0
seo_ag = set(b_seo[b_seo["approval_year"].notna()]["pnu"]); nat_p = set(b_nat["pnu"])
lost = [p for p in gp["pnu"] if p in seo_ag and p not in nat_p]
print(f"동작 필지 {len(gp):,} · 손실(aging→0) {len(lost):,} ({len(lost)/len(gp)*100:.1f}%)")
d = (sn.reindex(lost) - ss.reindex(lost)).dropna()
print(f"[손실 필지 환경점수 백분위 변화] 중앙 {d.median()*100:+.1f}%p · 하락>20%p {((d<-0.2).mean())*100:.0f}% · 상위10%→밖 {((ss.reindex(lost)>0.9)&(sn.reindex(lost)<0.9)).sum()}/{int((ss.reindex(lost)>0.9).sum())}")
dall = (sn - ss).reindex(gp['pnu']).dropna()
print(f"[비손실 포함 전체] 백분위 |Δ| 중앙 {dall.abs().median()*100:.1f}%p")

# #2 동작 여러 클러스터 닮은동네
ctx = pickle.load(open("_data/processed/serve_ctx.pkl", "rb")); zv = ctx.zone_vectors
bs = ctx.buildings; na = bs["gross_floor_area"].isna() | (bs["gross_floor_area"] <= 0); nap = bs["approval_year"].isna(); bs = bs[~(na & nap)]
print("\n[#2 동작 클러스터 닮은동네 top5 robust]")
for pre in ["1159010100", "1159010200", "1159010800"]:
    pn = [p for p in ctx.parcels["pnu"] if str(p).startswith(pre)][:35]
    if len(pn) < 5: continue
    rs, rn = search_cases(pn, ctx.parcels, bs, zv, k=5), search_cases(pn, ctx.parcels, b_nat, zv, k=5)
    si, ni = [m["zone_id"] for m in rs["matches"]], [m["zone_id"] for m in rn["matches"]]
    print(f"  {pre}: top5 교집합 {len(set(si)&set(ni))}/5 · 1위 {'동일' if si[0]==ni[0] else '뒤바뀜'} · 노후 {cluster_metrics(pn,ctx.parcels,bs,cfg=th)['old_area_ratio']:.2f}→{cluster_metrics(pn,ctx.parcels,b_nat,cfg=th)['old_area_ratio']:.2f}")

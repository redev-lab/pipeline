"""동작 hybrid aging(national + backfill) → candidate 회복률 측정. _experiments 전용."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import geopandas as gpd, pandas as pd, numpy as np
from redev.models.baseline import _SRC, _vsizip, prepare_baseline_matrix, build_neighbor_features
from redev.data.ingest.parcels import load_parcels
from redev.data.ingest.building_gis import load_buildings, _parse_approval_year, _classify_structure, _safe_normalize_pnu
from redev.graph.build import build_graph
from redev.graph.features import node_features
from redev.models.infer import train_production_b1, score_all

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

def backfill_b():
    c = pd.read_parquet("_experiments/gis_swap/backfill_dongjak.parquet")
    c["pnu"] = c["pnu"].astype(str).str.zfill(19)
    c = c[c["approval_year"].astype(str).str.len() == 4].copy()
    return pd.DataFrame({"pnu": c["pnu"], "approval_year": c["approval_year"].astype(int).astype("Int64"),
                         "structure": c["structure"].map(_classify_structure), "gross_floor_area": pd.to_numeric(c["gross_floor_area"], errors="coerce"),
                         "land_div": None, "sigungu": c["pnu"].str[:5]})

aug = prepare_baseline_matrix(); model, fc = train_production_b1(aug)
parcels, _ = load_parcels(_vsizip(*_SRC["parcels"]), ["11590"], with_geometry=True)
gp = parcels[parcels["sigungu"] == "11590"].copy()
b_seo, _ = load_buildings(_vsizip(*_SRC["buildings"]), with_geometry=False)
b_nat = nat_b(); bf = backfill_b()
b_hyb = pd.concat([b_nat, bf], ignore_index=True)
print(f"backfill 건물행 {len(bf)} (동작 PNU {bf['pnu'].nunique()})")

graph, p2i, _ = build_graph(gp); idx = np.empty(len(p2i), dtype=object)
for p, i in p2i.items(): idx[i] = p
base = pd.DataFrame({"pnu": idx, "t": 2026})
def sc(bld):
    f = build_neighbor_features(node_features(base, gp, bld), graph.edge_index, p2i, gp, bld, hops=2)
    return pd.Series(score_all(model, f, fc), index=f["pnu"]).rank(pct=True)
ss, sn, sh = sc(b_seo), sc(b_nat), sc(b_hyb)

# candidate 회복: 서울 top10% & national 밖 = 손실 candidate
lost = ss[(ss > 0.9) & (sn.reindex(ss.index) < 0.9)].index
rec_h = sh.reindex(lost) > 0.9
print(f"\n=== 동작 candidate 회복률 ===")
print(f"서울 기준 top10% candidate: {(ss>0.9).sum()}")
print(f"national에서 top10% 밖으로 추락(손실): {len(lost)} ({len(lost)/max((ss>0.9).sum(),1)*100:.0f}%)")
print(f"★hybrid로 top10% 복귀: {rec_h.sum()}/{len(lost)} ({rec_h.mean()*100:.0f}% 회복)")
# 전반 순위 회복(Spearman vs 서울)
from scipy.stats import spearmanr
print(f"\nSpearman(서울 기준): national {spearmanr(ss,sn.reindex(ss.index)).correlation:.3f} → hybrid {spearmanr(ss,sh.reindex(ss.index)).correlation:.3f}")
print(f"백분위 |Δ| 중앙: national {(sn.reindex(ss.index)-ss).abs().median()*100:.1f}%p → hybrid {(sh.reindex(ss.index)-ss).abs().median()*100:.1f}%p")

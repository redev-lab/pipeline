"""#3-a national swap이 ML 유사도(환경점수·닮은동네)에 주는 영향 + 호수밀도 단독영향 분해. _experiments 전용."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import geopandas as gpd, pandas as pd, numpy as np
from scipy.stats import spearmanr
from redev.models.baseline import _SRC, _vsizip, prepare_baseline_matrix, build_neighbor_features
from redev.data.ingest.parcels import load_parcels
from redev.data.ingest.building_gis import load_buildings, _parse_approval_year, _classify_structure, _safe_normalize_pnu
from redev.graph.build import build_graph
from redev.graph.features import node_features
from redev.models.infer import train_production_b1, score_all

DEMO = {"11290": "성북(정릉동)", "11590": "동작", "11380": "은평"}
D162, D164 = "_experiments/gis_swap/D162/AL_D162_11_20260115.shp", "_experiments/gis_swap/D164/AL_D164_11_20260115.shp"

def national_buildings():
    fr = []
    for p, (a1, a35, a28, a24) in [(D162, ("A1", "A35", "A28", "A24")), (D164, ("A1", "A35", "A28", "A24"))]:
        g = gpd.read_file(p, columns=[a1, a35, a28, a24], read_geometry=False, encoding="cp949")
        df = pd.DataFrame({"pnu": g[a1].map(_safe_normalize_pnu),
                           "approval_year": g[a35].map(_parse_approval_year).astype("Int64"),
                           "structure": g[a28].map(_classify_structure),
                           "gross_floor_area": pd.to_numeric(g[a24], errors="coerce")})
        fr.append(df)
    df = pd.concat(fr, ignore_index=True)
    df = df[df["pnu"].notna()].copy()
    na = df["gross_floor_area"].isna() | (df["gross_floor_area"] <= 0); nap = df["approval_year"].isna()
    df = df[~(na & nap)].copy(); df["land_div"] = None; df["sigungu"] = df["pnu"].str[:5]
    return df

print("모델 학습(동결 production B1+)·데이터 로드...")
aug = prepare_baseline_matrix()
model, fc = train_production_b1(aug)
AG = [c for c in fc if "aging" in c]; DN = [c for c in fc if "bldg_density" in c]
print(f"  fc {len(fc)}개 | 노후도계열 {AG} | 호수밀도계열 {DN}")
codes = list(DEMO)
parcels, _ = load_parcels(_vsizip(*_SRC["parcels"]), codes, with_geometry=True)
b_seo, _ = load_buildings(_vsizip(*_SRC["buildings"]), with_geometry=False)   # 서울(빈폴리곤 필터 적용됨)
b_nat = national_buildings()
print(f"  buildings 서울 {len(b_seo):,} · national {len(b_nat):,}")

def feats(gp, bld):
    graph, p2i, _ = build_graph(gp)
    base = pd.DataFrame({"pnu": [None]*len(p2i), "t": 2026})
    idx = np.empty(len(p2i), dtype=object)
    for p, i in p2i.items(): idx[i] = p
    base["pnu"] = idx
    sf = node_features(base, gp, bld)
    return build_neighbor_features(sf, graph.edge_index, p2i, gp, bld, hops=2)

for code, name in DEMO.items():
    gp = parcels[parcels["sigungu"] == code].copy()
    fs = feats(gp, b_seo)                       # 서울(B 기준)
    fn = feats(gp, b_nat)                        # national 둘 다
    # 하이브리드: 노후도 national, 호수밀도 서울 (호수밀도만 서울로 유지)
    fh = fs.copy(); fh[AG] = fn[AG].values       # 노후도←national, 호수밀도=서울 그대로
    out = {}
    for tag, f in [("B_서울", fs), ("노후national_밀도서울", fh), ("A_둘다national", fn)]:
        s = score_all(model, f, fc); out[tag] = pd.Series(s, index=f["pnu"])
    base = out["B_서울"]
    print(f"\n=== {name}({code}) {len(gp):,}필지 ===")
    for tag in ["노후national_밀도서울", "A_둘다national"]:
        v = out[tag].reindex(base.index)
        rho = spearmanr(base.values, v.values).correlation
        pct_b = base.rank(pct=True); pct_v = v.rank(pct=True)
        dpct = (pct_v - pct_b).abs()
        print(f"  [{tag}] 점수 Spearman {rho:.3f} | 백분위 |Δ| 중앙 {dpct.median()*100:.1f}%p 90분위 {dpct.quantile(.9)*100:.1f}%p | top10%유지 {((pct_b>0.9)&(pct_v>0.9)).sum()}/{int((pct_b>0.9).sum())}")
    # 호수밀도 단독영향 = (둘다national) vs (노후national·밀도서울)
    a, hyb = out["A_둘다national"].reindex(base.index), out["노후national_밀도서울"].reindex(base.index)
    rho_d = spearmanr(a.values, hyb.values).correlation
    print(f"  ★호수밀도 단독 효과(A vs 하이브리드): Spearman {rho_d:.3f} | 백분위 |Δ| 중앙 {(a.rank(pct=True)-hyb.rank(pct=True)).abs().median()*100:.1f}%p")
    # train-serve skew: 점수 분포
    print(f"  분포 중앙: 서울 {base.median():.3f} / 노후nat {hyb.median():.3f} / 둘다nat {a.median():.3f} (학습 score 중앙 ~{score_all(model, aug, fc).mean():.3f})")

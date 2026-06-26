"""노후도 커버리지 손실이 제품을 깨는가: #1 누락 쓰레기/실주거 분해 #2 동작 순위 robust #3 정직처리 손실."""
import sys, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import geopandas as gpd, pandas as pd, numpy as np
from redev.data.ingest.building_gis import _parse_approval_year, _safe_normalize_pnu
from redev.retrieval.case_search import search_cases
from redev.rules.stage1 import cluster_metrics
from redev.config import load_thresholds

th = load_thresholds()
SEO = "_experiments/gis_swap/seoul_al_d010/AL_D010_11_20260609.shp"
D = ["_experiments/gis_swap/D162/AL_D162_11_20260115.shp", "_experiments/gis_swap/D164/AL_D164_11_20260115.shp"]
RES = ("단독주택", "다가구주택", "다세대주택", "연립주택", "아파트", "공동주택", "도시형생활주택")

# 서울 건물(용도 A9 포함, 빈폴리곤 필터)
g = gpd.read_file(SEO, columns=["A2", "A9", "A13", "A14"], read_geometry=False, encoding="cp949")
g["pnu"] = g["A2"].map(_safe_normalize_pnu); g["apr"] = g["A13"].map(_parse_approval_year)
area = pd.to_numeric(g["A14"], errors="coerce")
g = g[g["pnu"].notna() & ((area > 0) | g["apr"].notna())].copy()
# national 건물 PNU(빈폴리곤 필터)
nat = pd.concat([gpd.read_file(p, columns=["A1", "A24", "A35"], read_geometry=False, encoding="cp949") for p in D], ignore_index=True)
nat["pnu"] = nat["A1"].map(_safe_normalize_pnu); nat["apr"] = nat["A35"].map(_parse_approval_year)
na = pd.to_numeric(nat["A24"], errors="coerce")
nat = nat[nat["pnu"].notna() & ((na > 0) | nat["apr"].notna())]
nat_pnus = set(nat["pnu"])

# aging 산출되는 서울 필지(승인일 건물 ≥1) vs national에 건물 0 = aging=0 손실 필지
seo_aging_pnus = set(g[g["apr"].notna()]["pnu"])
loss = seo_aging_pnus - nat_pnus
print(f"서울 aging 산출 필지 {len(seo_aging_pnus):,} · national 건물0(aging=0 손실) {len(loss):,} ({len(loss)/len(seo_aging_pnus)*100:.1f}%)")

# #1 손실 필지의 서울 건물 용도(실주거 vs 쓰레기/비주거)
lb = g[g["pnu"].isin(loss)]
res_share = (lb["A9"].astype(str).isin(RES)).mean()
print(f"\n[#1] 손실 필지 건물 용도 — 실주거 {res_share*100:.1f}% · 비주거/불명 {(1-res_share)*100:.1f}%")
print("   상위:", dict(lb["A9"].astype(str).value_counts().head(6)))
# 동작 집중
g["gu"] = g["pnu"].str[:5]; lossg = g[g["pnu"].isin(loss)].drop_duplicates("pnu")["gu"].value_counts()
allg = g.drop_duplicates("pnu")["gu"].value_counts()
sk = (lossg / allg).dropna().sort_values()
print(f"   손실 집중(손실율/구): 최고 {dict(sk.tail(3).round(2))} (11590=동작) 최저 {dict(sk.head(2).round(2))}")

# #3 손실 필지 노후도(서울) — 재개발 관심(노후)인가 + candidate 겹침
from redev.data.aging import old_ratio_by_parcel
seo_b = g.rename(columns={"apr": "approval_year"}); seo_b["structure"] = "other"; seo_b["gross_floor_area"] = pd.to_numeric(g["A14"], errors="coerce")
so = old_ratio_by_parcel(seo_b[["pnu", "approval_year", "structure", "gross_floor_area"]], 2026)
print(f"\n[#3] 손실 필지 서울 노후도: 중앙 {so.reindex(list(loss)).dropna().median()*100:.0f}% · 노후≥60% 비율 {(so.reindex(list(loss)).dropna()>=0.6).mean()*100:.0f}% (전체 {(so.dropna()>=0.6).mean()*100:.0f}%)")
try:
    sc = pd.read_parquet("_data/processed/infer_scores.parquet")
    pcol = "score_pct" if "score_pct" in sc else [c for c in sc.columns if "pct" in c][0]
    sc = sc.set_index("pnu")[pcol]
    print(f"   손실 필지 score_pct 중앙 {sc.reindex(list(loss)).dropna().median()*100:.0f}% · 상위10% 비율 {(sc.reindex(list(loss)).dropna()>0.9).mean()*100:.0f}% (전체 10%) → 관심구역 겹침")
except Exception as e:
    print("   infer_scores 조인 실패:", str(e)[:60])
print(f"   ★'데이터 없음' 정직처리 시 빠지는 필지 = {len(loss)/len(seo_aging_pnus)*100:.1f}% (이 중 노후≥60%가 재개발 관심 손실)")

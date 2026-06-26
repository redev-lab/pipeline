"""#1 보완 대상 필지 명세 확정: national aging=0 & 서울 건물有 & 실주거 PNU 목록. _experiments 전용."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import geopandas as gpd, pandas as pd
from redev.data.ingest.building_gis import _parse_approval_year, _safe_normalize_pnu

SEO = "_experiments/gis_swap/seoul_al_d010/AL_D010_11_20260609.shp"
D = ["_experiments/gis_swap/D162/AL_D162_11_20260115.shp", "_experiments/gis_swap/D164/AL_D164_11_20260115.shp"]
RES = ("단독주택", "다가구주택", "다세대주택", "연립주택", "아파트", "공동주택", "도시형생활주택")

g = gpd.read_file(SEO, columns=["A2", "A9", "A13", "A14"], read_geometry=False, encoding="cp949")
g["pnu"] = g["A2"].map(_safe_normalize_pnu); g["apr"] = g["A13"].map(_parse_approval_year)
g = g[g["pnu"].notna() & g["apr"].notna()].copy()           # aging 산출에 쓰이는 건물(사용승인일 有)
nat = pd.concat([gpd.read_file(p, columns=["A1"], read_geometry=False, encoding="cp949") for p in D], ignore_index=True)
nat_pnus = set(nat["A1"].map(_safe_normalize_pnu).dropna())

# 필지 단위: 서울 건물 용도(실주거 1개라도 있으면 실주거 필지) + national 유무
pg = g.groupby("pnu").agg(is_res=("A9", lambda s: s.astype(str).isin(RES).any()),
                          n_bld=("A9", "size"), gu=("pnu", lambda s: s.iloc[0][:5]))
pg["in_national"] = pg.index.isin(nat_pnus)
loss = pg[~pg["in_national"]].copy()                         # national 건물0 = aging=0 손실
loss_res = loss[loss["is_res"]]
print(f"서울 aging 필지 {len(pg):,}")
print(f"손실(national 건물0) {len(loss):,} ({len(loss)/len(pg)*100:.1f}%)")
print(f"  ★실주거 손실(보완 핵심) {len(loss_res):,} ({len(loss_res)/len(loss)*100:.0f}% of 손실)")
print(f"  비주거/불명 손실 {len(loss)-len(loss_res):,}")
loss.reset_index().to_csv("_experiments/gis_swap/backfill_pnus.csv", index=False)
print("저장: backfill_pnus.csv (pnu, is_res, n_bld, gu, in_national)")
print("구별 실주거 손실 상위:", dict(loss_res["gu"].value_counts().head(6)))
# PNU 형식 확인(건축물대장 조인 키)
print("PNU 예시(19자리):", list(loss.index[:3]))

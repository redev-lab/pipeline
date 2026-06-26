"""#2 집합 세대수/호수 컬럼, #3 누락 단독 공간대조, #4 용도불명 값. _experiments 전용."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import geopandas as gpd, pandas as pd, numpy as np
from redev.data.ingest.building_gis import _safe_normalize_pnu

D162 = "_experiments/gis_swap/D162/AL_D162_11_20260115.shp"
D164 = "_experiments/gis_swap/D164/AL_D164_11_20260115.shp"
SEO = "_experiments/gis_swap/seoul_al_d010/AL_D010_11_20260609.shp"

# ── #2. 집합(D164) 세대수/호수 컬럼 스캔 — 공동주택 행에서 정수형 후보 ──
print("[#2] 집합 D164 세대수/호수 컬럼 스캔")
d164 = gpd.read_file(D164, read_geometry=False, encoding="cp949")
apt = d164[d164["A30"].astype(str).str.contains("공동주택|아파트|연립|다세대", na=False)]
print(f"  집합 총 {len(d164):,} · 공동주택류 {len(apt):,}")
for c in d164.columns:
    if c == "geometry":
        continue
    s = pd.to_numeric(apt[c], errors="coerce").dropna()
    if len(s) and s.between(2, 5000).mean() > 0.7 and (s == s.round()).mean() > 0.9:  # 세대수다운 정수
        print(f"  후보 {c}: 중앙 {s.median():.0f} 범위 {s.min():.0f}~{s.max():.0f} 예 {list(apt[c].dropna().head(3))}")

# ── #3. 누락 단독주택 20샘플 — national footprint 공간 인접 존재? ──
print("\n[#3] 누락 단독주택 공간대조 (national footprint 인접 여부)")
seo_g = gpd.read_file(SEO, columns=["A2", "A9"], encoding="cp949").to_crs(5186)
seo_g["pnu"] = seo_g["A2"].map(_safe_normalize_pnu)
nat = pd.concat([gpd.read_file(p, columns=["A1"], read_geometry=False, encoding="cp949") for p in [D162, D164]], ignore_index=True)
nat["pnu"] = nat["A1"].map(_safe_normalize_pnu)
missing = set(seo_g["pnu"].dropna()) - set(nat["pnu"].dropna())
md = seo_g[seo_g["pnu"].isin(missing) & seo_g["A9"].astype(str).str.contains("단독", na=False)].dropna(subset=["pnu"])
samp = md.drop_duplicates("pnu").head(20).copy()
samp["geometry"] = samp.geometry.representative_point()
nat_pt = gpd.read_file(D162, columns=[], encoding="cp949").to_crs(5186)
nat_pt = gpd.GeoDataFrame(geometry=pd.concat([nat_pt.geometry, gpd.read_file(D164, columns=[], encoding="cp949").to_crs(5186).geometry], ignore_index=True), crs=5186)
nat_pt["geometry"] = nat_pt.geometry.representative_point()
near = gpd.sjoin_nearest(samp[["pnu", "geometry"]], nat_pt, how="left", distance_col="dist_m")
near = near.groupby("pnu")["dist_m"].min()
print(f"  단독 누락 샘플 {len(near)} · national 최근접 건물 거리(m): 중앙 {near.median():.1f} · <10m {((near<10).mean())*100:.0f}% · <30m {((near<30).mean())*100:.0f}% · >100m {((near>100).mean())*100:.0f}%")
print("  → <10m 다수면 footprint 존재(귀속 어긋남, 진짜 손실 아님) / >100m 다수면 진짜 누락")

# ── #4. 용도불명(None) 20샘플 — 서울에서 사용승인일·연면적 값 있나(진짜) vs 텅(쓰레기) ──
print("\n[#4] 용도불명(None) 건물 값 충실도")
sf = gpd.read_file(SEO, columns=["A9", "A13", "A14"], read_geometry=False, encoding="cp949")
none_b = sf[sf["A9"].isna() | (sf["A9"].astype(str).isin(["None", "nan", ""]))]
has_apr = none_b["A13"].notna() & (none_b["A13"].astype(str).str.len() >= 4)
area = pd.to_numeric(none_b["A14"], errors="coerce")
print(f"  용도불명 {len(none_b):,} · 사용승인일 있음 {has_apr.mean()*100:.0f}% · 연면적>0 {((area>0).mean())*100:.0f}% · 연면적 중앙 {area.median():.0f}㎡")
print(f"  → 사용승인일·연면적 다 있으면 실건물(신호), 다 결측이면 쓰레기 폴리곤")

"""호수밀도 재검증: #1 250m 격자 비지정(밀집편향 제거) AUC, #2 세대환산 가중 AUC. _experiments 전용."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import geopandas as gpd, pandas as pd, numpy as np
from shapely.geometry import box
from redev.data.ingest.zone_boundary import REDEV_SCLAS
from redev.config import load_thresholds
from sklearn.metrics import roc_auc_score

THR = load_thresholds()["housing_redevelopment"]["house_density_min"]
STD_SEDAE = 70.0     # 표준 세대 연면적(㎡) 프록시 — 집합 세대수 ≈ 연면적/70 (방향 테스트용, 관대)
SEO = "_experiments/gis_swap/seoul_al_d010/AL_D010_11_20260609.shp"
D162, D164 = "_experiments/gis_swap/D162/AL_D162_11_20260115.shp", "_experiments/gis_swap/D164/AL_D164_11_20260115.shp"
JIP = "공동주택|다세대|연립|아파트|도시형"   # 집합(세대 다수) 용도

# 실건물(연면적>0) + 세대환산 가중치 w. 일반=1, 집합=max(1, 연면적/표준세대면적).
def seo_pts():
    g = gpd.read_file(SEO, columns=["A9", "A14"], encoding="cp949").to_crs(5186)
    a = pd.to_numeric(g["A14"], errors="coerce").fillna(0); g = g[a > 0].copy(); a = a[a > 0]
    jip = g["A9"].astype(str).str.contains(JIP, na=False)
    g["w"] = np.where(jip, np.maximum(1.0, a / STD_SEDAE), 1.0)
    g["geometry"] = g.geometry.representative_point(); return g[["w", "geometry"]]
def nat_pts():
    fr = []
    for p, jipfile in [(D162, False), (D164, True)]:
        g = gpd.read_file(p, columns=["A24"], encoding="cp949").to_crs(5186)
        a = pd.to_numeric(g["A24"], errors="coerce").fillna(0); g = g[a > 0].copy(); a = a[a > 0]
        g["w"] = np.maximum(1.0, a / STD_SEDAE) if jipfile else 1.0   # D164=집합 전체 세대환산
        g["geometry"] = g.geometry.representative_point(); fr.append(g[["w", "geometry"]])
    return gpd.GeoDataFrame(pd.concat(fr, ignore_index=True), crs=5186)

print("점 로드...")
seo, nat = seo_pts(), nat_pts()
print(f"  서울 실건물 {len(seo):,} (가중합 {seo['w'].sum():,.0f}) · national {len(nat):,} (가중합 {nat['w'].sum():,.0f})")

# 지정구역
z = gpd.read_file("_experiments/gis_swap/uq181/shp파일/UPIS_C_UQ181.shp", columns=["SIGNGU_SE", "SCLAS_CL"], encoding="cp949").to_crs(5186)
z = z[z["SIGNGU_SE"].astype(str).str.startswith("11") & z["SCLAS_CL"].astype(str).isin(REDEV_SCLAS)]
z = z[z.geometry.notna() & (z.geometry.area > 0)].reset_index(drop=True)
z["gid"] = range(len(z)); z["ha"] = z.geometry.area / 1e4

# #1 250m 격자 비지정: 건물 있는 셀만, zone 교차 제외 (밀집 편향 없는 진짜 비교)
minx, miny, maxx, maxy = seo.total_bounds
cells = [box(x, y, x + 250, y + 250) for x in np.arange(minx, maxx, 250) for y in np.arange(miny, maxy, 250)]
grid = gpd.GeoDataFrame(geometry=cells, crs=5186); grid["gid"] = range(len(grid)); grid["ha"] = 6.25
has = gpd.sjoin(grid, gpd.GeoDataFrame(geometry=seo.geometry, crs=5186), predicate="intersects", how="inner")["gid"].unique()
grid = grid[grid["gid"].isin(has)]
inz = gpd.sjoin(grid, z[["geometry"]], predicate="intersects", how="inner")["gid"].unique()
grid = grid[~grid["gid"].isin(inz)].reset_index(drop=True)
print(f"격자 비지정 셀: {len(grid)} (건물有·zone밖, 6.25ha)")

def density(polys, pts, weighted):
    j = gpd.sjoin(pts, polys[["gid", "geometry"]], predicate="within", how="inner")
    agg = j.groupby("index_right")["w"].sum() if weighted else j.groupby("index_right").size()
    return (polys.index.to_series().map(agg).fillna(0) / polys["ha"]).values

for src, pts in [("서울", seo), ("national", nat)]:
    for wt, tag in [(False, "동수(원)"), (True, "세대환산")]:
        dz, dc = density(z, pts, wt), density(grid, pts, wt)
        y, s = np.r_[np.ones(len(dz)), np.zeros(len(dc))], np.r_[dz, dc]
        print(f"  [{src} · {tag}] 지정 중앙 {np.median(dz):.0f} vs 비지정 {np.median(dc):.0f} · AUC {roc_auc_score(y, s):.3f} · 지정통과율(≥{THR}) {(dz>=THR).mean()*100:.0f}%")

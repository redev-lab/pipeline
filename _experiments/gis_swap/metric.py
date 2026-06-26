"""#2 basis 정합(빈폴리곤 제거 후 격차), #3 지정/비지정 분리력(AUC), #4 통과율 수렴. _experiments 전용."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import geopandas as gpd, pandas as pd, numpy as np
from shapely.geometry import Point
from redev.data.ingest.zone_boundary import REDEV_SCLAS
from redev.config import load_thresholds
from sklearn.metrics import roc_auc_score

THR = load_thresholds()["housing_redevelopment"]["house_density_min"]
SEO = "_experiments/gis_swap/seoul_al_d010/AL_D010_11_20260609.shp"
D = ["_experiments/gis_swap/D162/AL_D162_11_20260115.shp", "_experiments/gis_swap/D164/AL_D164_11_20260115.shp"]

# 실건물 대표점(연면적>0): 빈/부속 폴리곤 제거 = 신발생무허가 제외 취지에 근접
def real_pts(paths, area_cols):
    fr = []
    for p, ac in zip(paths, area_cols):
        g = gpd.read_file(p, columns=[ac], encoding="cp949").to_crs(5186)
        g = g[pd.to_numeric(g[ac], errors="coerce").fillna(0) > 0]
        fr.append(gpd.GeoDataFrame(geometry=g.geometry.representative_point(), crs=5186))
    return gpd.GeoDataFrame(pd.concat(fr, ignore_index=True), crs=5186)

print("실건물 점 로드...")
seo = real_pts([SEO], ["A14"])
nat = real_pts(D, ["A24", "A24"])
print(f"  서울 실건물 {len(seo):,} · national 실건물 {len(nat):,}")

# 지정구역
z = gpd.read_file("_experiments/gis_swap/uq181/shp파일/UPIS_C_UQ181.shp", columns=["SIGNGU_SE", "SCLAS_CL"], encoding="cp949").to_crs(5186)
z = z[z["SIGNGU_SE"].astype(str).str.startswith("11") & z["SCLAS_CL"].astype(str).isin(REDEV_SCLAS)]
z = z[z.geometry.notna() & (z.geometry.area > 0)].reset_index(drop=True)
z["zid"] = range(len(z)); z["ha"] = z.geometry.area / 1e4
med_ha = z["ha"].median(); R = (med_ha * 1e4 / np.pi) ** 0.5  # 비지정 원 반지름(동일 면적)

# 비지정 표본: zone 밖 무작위 건물점 중심 동일면적 원
rng_idx = np.unique(np.linspace(0, len(seo) - 1, 4000).astype(int))
cand = seo.iloc[rng_idx].reset_index(drop=True)
inz = gpd.sjoin(cand, z[["geometry"]], predicate="within", how="inner")  # zone 안 후보
cand = cand[~cand.index.isin(inz.index)]                                  # zone 밖만
ctrl = gpd.GeoDataFrame(geometry=cand.geometry.buffer(R).values[:len(z)], crs=5186).reset_index(drop=True)
ctrl["zid"] = range(len(ctrl)); ctrl["ha"] = ctrl.geometry.area / 1e4
print(f"비지정 원 {len(ctrl)} (반지름 {R:.0f}m, 면적 {med_ha:.2f}ha)")

def dens(polys, pts):
    j = gpd.sjoin(pts, polys[["zid", "geometry"]], predicate="within", how="inner")
    return (polys["zid"].map(j.groupby("zid").size()).fillna(0) / polys["ha"]).values

for src, pts in [("서울(실건물)", seo), ("national(실건물)", nat)]:
    dz = dens(z, pts); dc = dens(ctrl, pts)
    y = np.r_[np.ones(len(dz)), np.zeros(len(dc))]; s = np.r_[dz, dc]
    auc = roc_auc_score(y, s)
    print(f"\n[{src}] 지정 밀도 중앙 {np.median(dz):.0f} · 비지정 중앙 {np.median(dc):.0f} · 분리 AUC {auc:.3f} · 지정 통과율(≥{THR}) {(dz>=THR).mean()*100:.0f}%")

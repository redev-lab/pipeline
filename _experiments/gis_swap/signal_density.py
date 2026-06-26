"""#1 핵심: 지정 재개발구역(의제처리) 호수밀도를 서울(4유형) vs national(1유형)로 계산,
60동/ha 임계 통과/미달 비교. swap이 실제 판정을 깨는가. _experiments 전용, 파이프라인 미변경."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import geopandas as gpd, pandas as pd
from redev.data.ingest.zone_boundary import REDEV_SCLAS
from redev.config import load_thresholds

THR = load_thresholds()["housing_redevelopment"]["house_density_min"]  # 60 동/ha
UQ = "_experiments/gis_swap/uq181/shp파일/UPIS_C_UQ181.shp"
D = ["_experiments/gis_swap/D162/AL_D162_11_20260115.shp", "_experiments/gis_swap/D164/AL_D164_11_20260115.shp"]
SEO = "_experiments/gis_swap/seoul_al_d010/AL_D010_11_20260609.shp"


def centroids(paths):  # 건물 → 대표점(5186)
    gs = [gpd.read_file(p, columns=[], encoding="cp949") for p in paths]
    g = pd.concat(gs, ignore_index=True) if len(gs) > 1 else gs[0]
    g = g.to_crs(5186)
    return gpd.GeoDataFrame(geometry=g.geometry.representative_point(), crs=5186)


print("의제처리 zone 로드...")
z = gpd.read_file(UQ, columns=["SIGNGU_SE", "SCLAS_CL", "NTFC_SN"], encoding="cp949").to_crs(5186)
z = z[z["SIGNGU_SE"].astype(str).str.startswith("11") & z["SCLAS_CL"].astype(str).isin(REDEV_SCLAS)].copy()
z = z[z.geometry.notna() & (z.geometry.area > 0)].reset_index(drop=True)
z["zid"] = range(len(z))
z["area_ha"] = z.geometry.area / 10000.0
print(f"서울 재개발 의제처리 구역: {len(z)} | 중앙 면적 {z['area_ha'].median():.2f}ha")

print("건물 centroid 로드 (서울/national)...")
seo_pt = centroids([SEO])
nat_pt = centroids(D)
print(f"  서울 {len(seo_pt):,} · national {len(nat_pt):,}")


def cnt_in_zones(pts):  # zone별 건물 동수
    j = gpd.sjoin(pts, z[["zid", "geometry"]], predicate="within", how="inner")
    return j.groupby("zid").size()


z["n_seo"] = z["zid"].map(cnt_in_zones(seo_pt)).fillna(0)
z["n_nat"] = z["zid"].map(cnt_in_zones(nat_pt)).fillna(0)
z["d_seo"] = z["n_seo"] / z["area_ha"]
z["d_nat"] = z["n_nat"] / z["area_ha"]

ps, pn = z["d_seo"] >= THR, z["d_nat"] >= THR
flip = ps & ~pn                     # 서울 통과 → national 미달 (판정 깨짐)
print(f"\n=== 호수밀도 {THR}동/ha 판정 (구역 {len(z)}) ===")
print(f"  서울 통과 {ps.sum()} ({ps.mean()*100:.0f}%) · national 통과 {pn.sum()} ({pn.mean()*100:.0f}%)")
print(f"  ★서울 통과인데 national 미달(판정 깨짐): {flip.sum()} ({flip.sum()/max(ps.sum(),1)*100:.1f}% of 서울통과)")
print(f"  national→서울 역전(national만 통과): {(pn & ~ps).sum()}")
print(f"  호수밀도 national/서울 비 중앙 {(z['d_nat']/z['d_seo']).replace([float('inf')],float('nan')).median():.3f}")
# flip 구역 특성
if flip.sum():
    f = z[flip]
    print(f"  flip 구역 서울밀도 중앙 {f['d_seo'].median():.0f} → national {f['d_nat'].median():.0f} (임계 {THR})")
z[["NTFC_SN", "area_ha", "n_seo", "n_nat", "d_seo", "d_nat"]].to_csv("_experiments/gis_swap/zone_density.csv", index=False)
print("저장: zone_density.csv")

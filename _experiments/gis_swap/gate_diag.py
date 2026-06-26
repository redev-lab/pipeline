"""#1 PNU 매칭 누락 452구역 원인 진단. redev/ 미변경(조사)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import geopandas as gpd, pandas as pd, numpy as np
from redev.models.baseline import _SRC, _vsizip
from redev.data.ingest.parcels import load_parcels
from redev.data.ingest.zone_boundary import REDEV_SCLAS

z = gpd.read_file("_experiments/gis_swap/uq181/shp파일/UPIS_C_UQ181.shp", columns=["SIGNGU_SE", "SCLAS_CL", "NTFC_SN"], encoding="cp949").to_crs(5186)
z = z[z["SIGNGU_SE"].astype(str).str.startswith("11") & z["SCLAS_CL"].astype(str).isin(REDEV_SCLAS)]
z = z[z.geometry.notna() & (z.geometry.area > 0)].reset_index(drop=True)
z["ha"] = z.geometry.area / 1e4
codes = sorted(z["SIGNGU_SE"].astype(str).unique())
print(f"지정구역 {len(z)} · 구 {len(codes)}")

parcels, _ = load_parcels(_vsizip(*_SRC["parcels"]), codes, with_geometry=True)
print(f"parcels {len(parcels):,} · 구 분포 {parcels['sigungu'].astype(str).str[:5].value_counts().to_dict()}" if "sigungu" in parcels else f"parcels {len(parcels):,}")

# 매칭 1: within(대표점), 매칭 2: intersects(폴리곤이 필지에 닿기만 해도)
pc = parcels[["pnu", "geometry"]].copy(); pc["rep"] = pc.geometry.representative_point()
pts = gpd.GeoDataFrame(pc[["pnu"]], geometry=pc["rep"], crs=5186)
within = set(gpd.sjoin(pts, z[["NTFC_SN", "geometry"]], predicate="within", how="inner")["NTFC_SN"])
inter = set(gpd.sjoin(gpd.GeoDataFrame(pc[["pnu"]], geometry=pc.geometry, crs=5186), z[["NTFC_SN", "geometry"]], predicate="intersects", how="inner")["NTFC_SN"])
allz = set(z["NTFC_SN"])
print(f"\n매칭된 지정구역: within(대표점) {len(within)} · intersects(필지겹침) {len(inter)} · 전체 {len(allz)}")
miss_w = allz - within
print(f"within 누락 {len(miss_w)} 중 intersects로 회복 {len(miss_w & inter)} · 끝까지 누락 {len(miss_w - inter)}")

# 누락 구역 특성: 면적·구 분포
zmw = z[z["NTFC_SN"].isin(miss_w)]
zfull = z[z["NTFC_SN"].isin(miss_w - inter)]
print(f"\nwithin-누락 구역 면적: 중앙 {zmw['ha'].median():.2f}ha vs 전체 중앙 {z['ha'].median():.2f}ha")
print(f"끝까지(intersects도)-누락 {len(zfull)} 면적 중앙 {zfull['ha'].median():.2f}ha")
print("within-누락 구 분포(상위):", dict(zmw["SIGNGU_SE"].astype(str).value_counts().head(6)))
# parcels에 그 구가 아예 없나
zgus = set(z["SIGNGU_SE"].astype(str)); pgus = set(parcels["sigungu"].astype(str).str[:5]) if "sigungu" in parcels else set()
print("zone 있는 구 중 parcels에 없는 구:", zgus - pgus if pgus else "(sigungu컬럼 확인필요)")
print("끝까지-누락 구 분포:", dict(zfull["SIGNGU_SE"].astype(str).value_counts().head(8)))

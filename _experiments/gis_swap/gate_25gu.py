"""1단계 게이트 확인: 25구 전 지정구역에서 호수밀도 OR-항 제거(A vs B) flip 카운트. redev/ 미변경(조사)."""
import sys, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import geopandas as gpd, pandas as pd, numpy as np
from redev.rules.stage1 import cluster_metrics, _urban_eligible
from redev.models.baseline import _SRC, _vsizip
from redev.data.ingest.parcels import load_parcels
from redev.data.ingest.zone_boundary import REDEV_SCLAS
from redev.config import load_thresholds

th = load_thresholds(); h = th["housing_redevelopment"]

# 지정구역 폴리곤(25구 재개발 의제처리)
z = gpd.read_file("_experiments/gis_swap/uq181/shp파일/UPIS_C_UQ181.shp", columns=["SIGNGU_SE", "SCLAS_CL", "NTFC_SN"], encoding="cp949").to_crs(5186)
z = z[z["SIGNGU_SE"].astype(str).str.startswith("11") & z["SCLAS_CL"].astype(str).isin(REDEV_SCLAS)]
z = z[z.geometry.notna() & (z.geometry.area > 0)].reset_index(drop=True)
codes = sorted(z["SIGNGU_SE"].astype(str).unique())
print(f"지정구역 {len(z)} · 해당 구 {len(codes)}개")

# 25구(해당구) parcels + buildings(빈폴리곤 필터)
print("parcels 로드(해당 구 전체, geometry)...")
parcels, _ = load_parcels(_vsizip(*_SRC["parcels"]), codes, with_geometry=True)
print(f"  parcels {len(parcels):,}")
ctx = pickle.load(open("_data/processed/serve_ctx.pkl", "rb"))
b = ctx.buildings
na = b["gross_floor_area"].isna() | (b["gross_floor_area"] <= 0); nap = b["approval_year"].isna()
b = b[~(na & nap)].copy()
print(f"  buildings(빈폴리곤 제거 후) {len(b):,}")

# PNU→zone: 필지 대표점이 지정 폴리곤 내부
pc = parcels[["pnu", "geometry"]].copy(); pc["geometry"] = pc.geometry.representative_point()
j = gpd.sjoin(gpd.GeoDataFrame(pc, crs=5186), z[["NTFC_SN", "geometry"]], predicate="within", how="inner")
z2p = j.groupby("NTFC_SN")["pnu"].apply(list).to_dict()
print(f"PNU 귀속된 지정구역 {len(z2p)}개 (멤버 필지 있는 구역)")

now = nod = flip = 0; rescued = []
for zid, pnus in z2p.items():
    m = cluster_metrics(pnus, parcels, b, cfg=th)
    oa, ab, hd = m["old_area_ratio"], m["abut_ratio"], m["house_density"]
    old_ok = pd.notna(oa) and oa >= h["old_building_area_ratio"]
    abut_ok = pd.notna(ab) and ab <= h["abutting_road_ratio_max"]
    dens_ok = pd.notna(hd) and hd >= h["house_density_min"]
    u_elig = _urban_eligible(m, th)[0]
    p_now = (old_ok and (abut_ok or dens_ok)) or u_elig    # A 현재
    p_nod = (old_ok and abut_ok) or u_elig                 # B 제거
    now += p_now; nod += p_nod
    if p_now and not p_nod:
        flip += 1; rescued.append((zid, round(hd or 0, 0), round(ab or 0, 2), u_elig))
print(f"\n=== 25구 지정 {len(z2p)}구역 요건 통과 (빈폴리폴 필터 반영) ===")
print(f"  A 현재(호수밀도 포함): {now}")
print(f"  B 제거(old_ok and abut_ok): {nod}")
print(f"  ★flip(A통과·B탈락 = 호수밀도로만 구제): {flip}")
for r in rescued:
    print("   구제구역:", r)

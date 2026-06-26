"""#2 flip 정확수(644 전수) + #3 구제구역 national 호수밀도 위험. redev/ 미변경(조사)."""
import sys, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import geopandas as gpd, pandas as pd, numpy as np
from redev.rules.stage1 import cluster_metrics, _urban_eligible
from redev.models.baseline import _SRC, _vsizip
from redev.data.ingest.parcels import load_parcels
from redev.data.ingest.building_gis import _safe_normalize_pnu
from redev.data.ingest.zone_boundary import REDEV_SCLAS
from redev.config import load_thresholds

th = load_thresholds(); h = th["housing_redevelopment"]; THR = h["house_density_min"]
GU = {"11000": "서울시", "11140": "중구", "11200": "성동구", "11560": "영등포구", "11110": "종로구"}

z = gpd.read_file("_experiments/gis_swap/uq181/shp파일/UPIS_C_UQ181.shp", columns=["SIGNGU_SE", "SCLAS_CL", "NTFC_SN"], encoding="cp949").to_crs(5186)
z = z[z["SIGNGU_SE"].astype(str).str.startswith("11") & z["SCLAS_CL"].astype(str).isin(REDEV_SCLAS)]
z = z[z.geometry.notna() & (z.geometry.area > 0)]
zdis = z.dissolve("NTFC_SN").reset_index()                 # 다획지 → 고시 단위로 병합
zdis["ha"] = zdis.geometry.area / 1e4
codes = sorted(z["SIGNGU_SE"].astype(str).unique())

parcels, _ = load_parcels(_vsizip(*_SRC["parcels"]), codes, with_geometry=True)
ctx = pickle.load(open("_data/processed/serve_ctx.pkl", "rb"))
b = ctx.buildings; na = b["gross_floor_area"].isna() | (b["gross_floor_area"] <= 0); nap = b["approval_year"].isna()
b = b[~(na & nap)].copy()

# national 건물 PNU 테이블(실건물 연면적>0)
nat = pd.concat([gpd.read_file(p, columns=["A1", "A24"], read_geometry=False, encoding="cp949") for p in
                 ["_experiments/gis_swap/D162/AL_D162_11_20260115.shp", "_experiments/gis_swap/D164/AL_D164_11_20260115.shp"]], ignore_index=True)
nat = nat[pd.to_numeric(nat["A24"], errors="coerce").fillna(0) > 0]
nat["pnu"] = nat["A1"].map(_safe_normalize_pnu)
nat_cnt = nat.groupby("pnu").size()

pc = parcels[["pnu", "geometry"]].copy(); pc["geometry"] = pc.geometry.representative_point()
j = gpd.sjoin(gpd.GeoDataFrame(pc, crs=5186), zdis[["NTFC_SN", "geometry"]], predicate="within", how="inner")
z2p = j.groupby("NTFC_SN")["pnu"].apply(list).to_dict()
ha = dict(zip(zdis["NTFC_SN"], zdis["ha"]))
print(f"고유 지정구역 {len(zdis)} · 매칭 {len(z2p)}")

flip = []
for zid, pnus in z2p.items():
    m = cluster_metrics(pnus, parcels, b, cfg=th)
    oa, ab, hd = m["old_area_ratio"], m["abut_ratio"], m["house_density"]
    old_ok = pd.notna(oa) and oa >= h["old_building_area_ratio"]
    abut_ok = pd.notna(ab) and ab <= h["abutting_road_ratio_max"]
    dens_ok = pd.notna(hd) and hd >= THR
    u_elig = _urban_eligible(m, th)[0]
    if (old_ok and (abut_ok or dens_ok)) or u_elig:
        if not ((old_ok and abut_ok) or u_elig):           # 호수밀도로만 구제
            nd = sum(nat_cnt.get(p, 0) for p in pnus) / ha[zid]   # national 호수밀도
            flip.append((zid, GU.get(zid[:5], zid[:5]), round(hd, 1), round(ab, 2), round(nd, 1)))
print(f"\n★flip(호수밀도로만 구제) 정확수: {len(flip)}/{len(z2p)} 구역")
print(f"{'고시':24} {'구':8} {'서울hd':>6} {'접도율':>5} {'nat_hd':>6} {'nat<60?':>7}")
for zid, gu, hd, ab, nd in sorted(flip, key=lambda x: x[2]):
    print(f"{zid:24} {gu:8} {hd:6.1f} {ab:5.2f} {nd:6.1f} {'★탈락' if nd < THR else '유지':>7}")

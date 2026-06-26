"""일회용 측정: national(1유형 AL_D162/D164) vs 서울(4유형 AL_D010) 노후도·호수밀도 diff.
building_gis 헬퍼·aging.old_ratio_by_parcel를 수정 없이 재사용. 어댑터로 컬럼만 리네임.
매핑(실증 확정): PNU A1→A2 / 사용승인일 A35→A13 / 연면적 A24→A14 / 구조 A28→A11."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import geopandas as gpd, pandas as pd
from redev.data.ingest.building_gis import _parse_approval_year, _classify_structure, _safe_normalize_pnu
from redev.data.aging import old_ratio_by_parcel

T = 2026
D162 = "_experiments/gis_swap/D162/AL_D162_11_20260115.shp"
D164 = "_experiments/gis_swap/D164/AL_D164_11_20260115.shp"
SEO = "_experiments/gis_swap/seoul_al_d010/AL_D010_11_20260609.shp"


def build_table(gdf, pnu_c, apr_c, str_c, area_c):  # 어댑터 + 기존 정제 로직 재사용
    df = pd.DataFrame({
        "pnu": gdf[pnu_c].map(_safe_normalize_pnu),
        "approval_year": gdf[apr_c].map(_parse_approval_year),
        "structure": gdf[str_c].map(_classify_structure),
        "gross_floor_area": pd.to_numeric(gdf[area_c], errors="coerce"),
    })
    return df[df["pnu"].notna()].copy()


print("로드·정제 중...")
seo = build_table(gpd.read_file(SEO, columns=["A2", "A13", "A11", "A14"], read_geometry=False, encoding="cp949"), "A2", "A13", "A11", "A14")
nat_raw = pd.concat([gpd.read_file(p, columns=["A1", "A35", "A28", "A24"], read_geometry=False, encoding="cp949") for p in [D162, D164]], ignore_index=True)
nat = build_table(nat_raw, "A1", "A35", "A28", "A24")
print(f"정제 건물수: 서울(4유형) {len(seo):,} | national(1유형) {len(nat):,}")

# ── 노후도 (면적가중, 파이프라인과 동일) ──
seo_old, nat_old = old_ratio_by_parcel(seo, T), old_ratio_by_parcel(nat, T)
j = pd.DataFrame({"seo": seo_old, "nat": nat_old}).dropna()
d = j["nat"] - j["seo"]
print(f"\n=== 노후도 (공통 {len(j):,} 필지, 면적가중, t={T}) ===")
print(f"  |차이| 평균 {d.abs().mean()*100:.2f}%p · 중앙값 {d.abs().median()*100:.2f}%p · 최대 {d.abs().max()*100:.1f}%p")
print(f"  ±5%p 초과 {((d.abs()>0.05).mean())*100:.2f}% · ±10%p 초과 {((d.abs()>0.10).mean())*100:.2f}% · 완전일치(±0.1%p) {((d.abs()<0.001).mean())*100:.1f}%")

# ── 호수밀도: 필지당 건물 동수 (필지면적 동일 → 동수 차이 = 밀도 차이) ──
sc, nc = seo.groupby("pnu").size(), nat.groupby("pnu").size()
jc = pd.DataFrame({"seo": sc, "nat": nc}).dropna()
dc = jc["nat"] - jc["seo"]
print(f"\n=== 호수밀도(필지당 건물 동수) (공통 {len(jc):,} 필지) ===")
print(f"  동수차 평균 {dc.mean():.3f} · 중앙값 {dc.median():.1f} · |차|평균 {dc.abs().mean():.3f} · 동일필지 {((dc==0).mean())*100:.1f}% · national 적음 {((dc<0).mean())*100:.1f}%")
print(f"  PNU 커버리지: 서울만 {len(set(sc.index)-set(nc.index)):,} · national만 {len(set(nc.index)-set(sc.index)):,} · 공통 {len(jc):,}")

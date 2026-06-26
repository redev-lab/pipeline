"""커버리지 갭 진단: 국가표준 23% 필지 누락이 신호냐 노이즈냐. _experiments 전용, 본 파이프라인 미변경."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import geopandas as gpd, pandas as pd, numpy as np
from redev.data.ingest.building_gis import _parse_approval_year, _classify_structure, _safe_normalize_pnu
from redev.data.aging import old_ratio_by_parcel

T = 2026
D = ["_experiments/gis_swap/D162/AL_D162_11_20260115.shp", "_experiments/gis_swap/D164/AL_D164_11_20260115.shp"]
SEO = "_experiments/gis_swap/seoul_al_d010/AL_D010_11_20260609.shp"

seo_raw = gpd.read_file(SEO, columns=["A2", "A9", "A13", "A11", "A14", "A23"], read_geometry=False, encoding="cp949")
seo_raw["pnu"] = seo_raw["A2"].map(_safe_normalize_pnu)
seo = seo_raw.dropna(subset=["pnu"]).copy()
d162 = gpd.read_file(D[0], columns=["A1", "A30"], read_geometry=False, encoding="cp949"); d162["src"] = "일반"
d164 = gpd.read_file(D[1], columns=["A1", "A30"], read_geometry=False, encoding="cp949"); d164["src"] = "집합"
nat = pd.concat([d162, d164], ignore_index=True)
nat["pnu"] = nat["A1"].map(_safe_normalize_pnu)
nat = nat.dropna(subset=["pnu"]).copy()

seo_pnus, nat_pnus = set(seo["pnu"]), set(nat["pnu"])
missing = seo_pnus - nat_pnus                      # 서울에만 (national 누락)
common = seo_pnus & nat_pnus
print(f"필지: 서울 {len(seo_pnus):,} · national {len(nat_pnus):,} · 누락(서울만) {len(missing):,} · 공통 {len(common):,}")

# ── Q1. 진짜 누락 vs 조인 허상 (raw 문자열 레벨로도 누락인가) ──
seo_raw_set, nat_raw_set = set(seo_raw["A2"].astype(str)), set(nat["A1"].astype(str))
sample = list(missing)[:20]
# 정규화 전 raw에서도 national에 없나 (앞 18자리 느슨 매칭까지)
nat_raw18 = set(s[:18] for s in nat_raw_set)
raw_absent = sum(1 for p in sample if p not in nat_raw_set and p[:18] not in nat_raw18)
print(f"\n[Q1] 누락 20샘플 중 raw(정규화 전)·앞18자리로도 national에 없음: {raw_absent}/20  → {raw_absent>=18 and '진짜 누락' or '조인 허상 의심'}")
print(f"   raw 레벨 전체 누락 필지수: {len(seo_raw_set - nat_raw_set):,} (정규화 후 {len(missing):,} 와 유사하면 허상 아님)")

# ── Q2. 누락 필지 건물 용도 분포 (서울 기준) ──
RES = ("단독주택", "다가구주택", "다세대주택", "연립주택", "아파트", "주택", "도시형생활주택")
def res_share(df):
    u = df["A9"].astype(str)
    return (u.isin(RES) | u.str.contains("주택", na=False)).mean()
miss_b = seo[seo["pnu"].isin(missing)]
comm_b = seo[seo["pnu"].isin(common)]
print(f"\n[Q2] 누락 필지 건물 용도 — 주거비율 {res_share(miss_b)*100:.1f}% (공통 필지 {res_share(comm_b)*100:.1f}%)")
print("   누락 용도 상위:", dict(miss_b["A9"].astype(str).value_counts().head(8)))

# ── Q3. 누락 필지가 노후+밀집 신호를 가졌나 (서울 데이터로) ──
so = old_ratio_by_parcel(seo.rename(columns={}).assign(approval_year=seo["A13"].map(_parse_approval_year), structure=seo["A11"].map(_classify_structure), gross_floor_area=pd.to_numeric(seo["A14"], errors="coerce")), T)
cnt = seo.groupby("pnu").size()
def dist(pnus, name):
    o = so.reindex(list(pnus)).dropna(); c = cnt.reindex(list(pnus)).dropna()
    print(f"   {name}: 노후도 중앙 {o.median()*100:.0f}% 평균 {o.mean()*100:.0f}% | 노후≥60% 비율 {(o>=0.6).mean()*100:.0f}% | 동수 중앙 {c.median():.0f}")
print("\n[Q3] 노후+밀집 신호 (서울 기준)")
dist(missing, "누락 필지")
dist(common, "공통 필지")

# ── Q4. 위치 — 누락 필지 구 분포 + 노후밀집 집중도 ──
gu_miss = seo[seo["pnu"].isin(missing)].drop_duplicates("pnu")["A23"].value_counts(normalize=True)
gu_all = seo.drop_duplicates("pnu")["A23"].value_counts(normalize=True)
skew = (gu_miss / gu_all).dropna().sort_values()
print(f"\n[Q4] 누락 필지 구 편중 (누락비율/전체비율, 1=균등): 최저 {dict(skew.head(2).round(2))} 최고 {dict(skew.tail(2).round(2))}")

# ── Q5. 갭이 일반/집합 어디서? (누락 건물 용도가 일반(단독) 위주면 부속·무허가 가설) ──
print(f"\n[Q5] national 건물수 일반 {len(d162):,} 집합 {len(d164):,} | 서울 건물수 {len(seo):,}")
print("   서울 전체 용도 상위:", dict(seo["A9"].astype(str).value_counts(normalize=True).head(5).round(3)))

# ── Q6. 호수밀도 7.9% 불일치 필지: 방향·크기·노후밀집 겹침 ──
sc, nc = seo.groupby("pnu").size(), nat.groupby("pnu").size()
jc = pd.DataFrame({"seo": sc, "nat": nc}).dropna(); dc = jc["nat"] - jc["seo"]
mis = jc[dc != 0]
print(f"\n[Q6] 호수밀도 불일치 필지 {len(mis):,} ({len(mis)/len(jc)*100:.1f}%) | national 더 낮음 {(dc[dc!=0]<0).mean()*100:.0f}% | 동수차 중앙 {dc[dc!=0].median():.0f}")
mis_old = so.reindex(mis.index).dropna()
print(f"   불일치 필지 노후도 중앙 {mis_old.median()*100:.0f}% (전체 공통 {so.reindex(jc.index).dropna().median()*100:.0f}%) → 노후밀집 겹침 {'높음' if mis_old.median()>so.reindex(jc.index).dropna().median() else '비슷/낮음'}")

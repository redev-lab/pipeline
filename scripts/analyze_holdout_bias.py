"""보류(no_clean_t) 편향 점검 — 분석 전용 (본 파이프라인 미수정).

73개 의제처리 재개발 폴리곤(4구) 중, 신통/정비사업에 t가 안 붙어 '보류'되는
구역이 무작위로 빠지는지 / 구·연대에 체계적으로 쏠리는지 집계한다.

매칭은 이름키(결정고시 제목 ↔ 정비사업·신통 구역명) 근사 — 신통은 좌표가 없어
공간매칭 불가라 근사임. 따라서 절대 멤버십이 아니라 '분포 쏠림'을 본다.
"""
import os, re
import pandas as pd
from pyogrio import read_dataframe
from redev.config import training_districts

RAW = "_data/raw"
name2code = {d["name"]: d["sigungu_code"] for d in training_districts()}
code2name = {v: k for k, v in name2code.items()}
codes = set(name2code.values())

BOIL = ["주택재개발","도시정비형","도시환경정비","재정비촉진","재개발","재건축","정비사업",
        "정비구역","존치관리","장기전세주택","역세권","구역","사업","조합","일대","일원",
        "(촉)","정비","단독","주거환경관리","주거환경개선","관리형","가로주택","소규모"]
def key(s, code):
    s = str(s)
    for t in BOIL: s = s.replace(t, "")
    s = s.replace("제", "").replace(" ", "")
    m = re.search(r"([가-힣]{2,}?)(\d+)", s)
    return (code, m.group(1), m.group(2)) if m else None

# UQ181 4구 재개발 + 결정고시 제목/연도
uq = read_dataframe(f"/vsizip/{os.path.abspath(RAW)}/UQ181_의제처리구역_202602.zip/shp파일/UPIS_C_UQ181.shp",
                    read_geometry=False, encoding="cp949")
uq = uq[uq["SIGNGU_SE"].astype(str).isin(codes) &
        uq["SCLAS_CL"].astype(str).isin(["UQ1221","UQ1222"])].copy()
gosi = pd.read_csv(f"{RAW}/서울시 도시계획 결정고시 정보.csv", encoding="cp949", dtype=str)\
        .drop_duplicates("고시관리코드").set_index("고시관리코드")
uq["sig"] = uq["SIGNGU_SE"].astype(str)
uq["title"] = uq["NTFC_SN"].map(gosi["제목"])
uq["gy"] = uq["NTFC_SN"].map(gosi["고시일자"]).astype(str).str[:4]
uq["key"] = [key(t, c) for t, c in zip(uq["title"], uq["sig"])]

# 정비사업/신통 키
jb = pd.read_csv(f"{RAW}/서울특별시_서울시 정비사업 데이터_20211227.csv", encoding="cp949", dtype=str)
jb = jb[jb["시군구명"].isin(name2code) & jb["사업시행방식"].astype(str).str.contains("재개발", na=False)]
jbkeys = {key(n, name2code[g]) for n, g in zip(jb["정비 구역명"], jb["시군구명"])} - {None}
jb_names = jb["정비 구역명"].astype(str).tolist()
sht = pd.read_csv(f"{RAW}/신통_선정구역_positive.csv", encoding="utf-8-sig", dtype=str)
sht = sht[sht["자치구"].isin(name2code)]
shtkeys = {key(n, name2code[g]) for n, g in zip(sht["구역명"], sht["자치구"])} - {None}

uq["m_jb"] = uq["key"].map(lambda k: bool(k) and k in jbkeys)
uq["m_sht"] = uq["key"].map(lambda k: bool(k) and k in shtkeys)
uq["has_t"] = uq["m_jb"] | uq["m_sht"]
uq["holdout"] = ~uq["has_t"]

print("=== [전체] UQ181 4구 재개발 %d, 보류 %d (%.0f%%) ===" % (
    len(uq), int(uq["holdout"].sum()), 100*uq["holdout"].mean()))

print("\n=== 1) 자치구별 보류율 ===")
g = uq.groupby("sig").agg(전체=("holdout","size"), 보류=("holdout","sum"))
g["보류율%"] = (100*g["보류"]/g["전체"]).round(0)
g.index = [code2name[c] for c in g.index]
print(g.to_string())

print("\n=== 2) 소스 커버리지 ===")
print("  정비사업만:", int((uq.m_jb & ~uq.m_sht).sum()),
      " 신통만:", int((~uq.m_jb & uq.m_sht).sum()),
      " 둘다:", int((uq.m_jb & uq.m_sht).sum()),
      " 보류(neither):", int(uq.holdout.sum()))

print("\n=== 3) 보류 구역 연대 분포 (UQ181 최신고시 연도) ===")
hd = uq[uq.holdout]
print("  ", hd["gy"].value_counts().sort_index().to_dict())
print("  2022+ :", int((hd.gy>="2022").sum()), " / 2021이하 :", int((hd.gy<"2022").sum()))

print("\n=== 4) 보류 사유 breakdown ===")
keyless = int(hd["key"].isna().sum())
recent = int((hd["key"].notna() & (hd.gy>="2022")).sum())
old = int((hd["key"].notna() & (hd.gy<"2022")).sum())
print("  이름키 추출실패(매칭 기계적 불가):", keyless)
print("  최근(2022+, 정비사업 2021CSV에 애초 없음):", recent)
print("  옛날(<2022, CSV gap 또는 신통 매칭실패):", old)
# 옛날 보류가 정비사업 CSV에 '이름이라도' 있나 (substring)
old_hd = hd[hd["key"].notna() & (hd.gy<"2022")]
in_csv = sum(any(str(t).split()[0][:3] in n for n in jb_names) for t in old_hd["title"].dropna())
print("  └ 그 옛날 보류 중 정비사업CSV에 이름 흔적 있음(매칭만 실패):", in_csv, "/", len(old_hd))

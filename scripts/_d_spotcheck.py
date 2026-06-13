"""D 전역 일반화 수검 — 점수 캐시(25구) 기반, 메모리 경량(필요 구만 적재).

D-1 [비학습 구 spot check]: 학습 안 쓴 용산·동대문·중랑의 실제 의제처리 지정구역 위에서
     점수가 상위권인지(score_pct 높은지). 의제처리 SHP는 서울 전체분이라 라벨 없이 위치만 본다.
D-2 [유형 밖]: 북촌 한옥(종로 가회동·삼청동)·도심 상업지(종로1가·공평동) 점수 — 높으면
     '보존지구·상업지역은 점수 높아도 정비대상 아닐 수 있음' caveat 대상(오답 라벨링).
실행: python scripts/_d_spotcheck.py
"""
import os, sys
sys.path.insert(0, os.getcwd()); sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from redev.serve.api import load_scores

sc = load_scores()[["pnu", "sigungu", "score", "score_pct"]]
print(f"점수 캐시 {len(sc):,}필지 / {sc['sigungu'].nunique()}구\n")

# ── D-1: 비학습 구 지정구역 점수 상위권? ────────────────────────────────────
print("=== D-1 비학습 구 지정구역 spot check ===")
from redev.data.ingest.parcels import load_parcels
from redev.data.ingest.zone_boundary import load_zones
from redev.data.labels import _positives_from_zonetable
from redev.models.baseline import _RAW, _SRC, _vsizip

D1 = {"11170": "용산구", "11230": "동대문구", "11260": "중랑구"}
parcels, _ = load_parcels(_vsizip(*_SRC["parcels"]), list(D1), with_geometry=True)
zt, _ = load_zones(_vsizip(*_SRC["uq"]), str(_RAW / _SRC["gosi"]), parcels, list(D1),
                   jeonbisaeop_csv=str(_RAW / _SRC["jeonbisaeop"]), shintong_csv=str(_RAW / _SRC["shintong"]),
                   public_redev_csv=str(_RAW / _SRC["public_redev"]))
pos = _positives_from_zonetable(zt, parcels)
m = pos.merge(sc, on="pnu", how="inner")
print(f"지정구역 필지 {len(pos):,} (점수캐시 매칭 {len(m):,})")
print(f"{'구':<8}{'지정필지':>8}{'중앙score_pct':>14}{'상위10%비율':>12}{'상위30%비율':>12}")
for code, name in D1.items():
    g = m[m["sigungu"] == code]
    if len(g):
        print(f"{name:<8}{len(g):>8,}{g['score_pct'].median():>14.2f}"
              f"{(g['score_pct']>=0.9).mean():>12.2f}{(g['score_pct']>=0.7).mean():>12.2f}")
    else:
        print(f"{name:<8}{'(지정구역 매칭 0)':>20}")
print(f"→ 해석: 지정구역 중앙 score_pct가 0.5보다 충분히 높으면 비학습 구에도 일반화(상위권 포착).\n")
del parcels, zt, pos, m

# ── D-2: 유형 밖(보존·상업) 점수 ───────────────────────────────────────────
print("=== D-2 유형 밖(북촌 한옥·종로 상업지) 점검 ===")
jp, _ = load_parcels(_vsizip(*_SRC["parcels"]), ["11110"], with_geometry=False)  # 종로
jp = jp.merge(sc, on="pnu", how="inner")
jp["dong"] = jp["dong_addr"].fillna("").str.split().str[-1]
groups = {"북촌 한옥(가회동·삼청동)": ["가회동", "삼청동"],
          "도심 상업(종로1가·공평동·청진동)": ["종로1가", "공평동", "청진동"]}
flagged = False
for label, dongs in groups.items():
    g = jp[jp["dong"].isin(dongs)]
    if len(g):
        hi = (g["score"] >= 0.5).mean()
        print(f"{label}: {len(g):,}필지 중앙score {g['score'].median():.3f} / "
              f"중앙score_pct {g['score_pct'].median():.2f} / ≥0.5비율 {hi:.2f}")
        if g["score"].median() >= 0.5:
            flagged = True
    else:
        print(f"{label}: 매칭 0")
print(f"\n→ 유형 밖 점수 높음(flagged)={flagged} — 높으면 caveat '보존지구·상업지역 등은 점수가 "
      f"높아도 정비 대상이 아닐 수 있음' 대상.")
print("DONE")

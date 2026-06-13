"""전역(25구) 측정 — C④ 점수 분포 상식 정합 + D-2 유형 밖 점검.

실행: python scripts/_global_measure.py  (infer_scores.parquet 필요)
C④: 자치구별 점수 분포 — 강남·서초 신축지대 낮고 노후 자치구 높은지(상식 정합·붕괴 구 보고).
D-2: 북촌 한옥(종로 가회동)·도심 상업지(종로 1가) 점수 — 높으면 caveat/백로그 대상.
"""
import os, sys
sys.path.insert(0, os.getcwd()); sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from redev.config import inference_districts
from redev.serve.api import load_scores

code2name = {d["sigungu_code"]: d["name"] for d in inference_districts()}
df = load_scores()
print(f"전역 캐시: {len(df):,}필지 / {df['sigungu'].nunique()}구\n")

# C④ 자치구별 분포(중앙 점수 내림차순) — 노후↑/신축↓ 상식 정합
g = df.groupby("sigungu").agg(n=("score", "size"), med=("score", "median"),
                              mean=("score", "mean"), hi=("score", lambda s: (s >= 0.5).mean()))
g["name"] = g.index.map(code2name)
g = g.sort_values("med", ascending=False)
print("자치구별 점수 분포 (중앙 내림차순):")
print(f"{'구':<8}{'n':>8}{'중앙':>8}{'평균':>8}{'≥0.5비율':>10}")
for code, r in g.iterrows():
    print(f"{str(r['name']):<8}{int(r['n']):>8,}{r['med']:>8.3f}{r['mean']:>8.3f}{r['hi']:>10.2f}")

newish = ["강남구", "서초구", "송파구"]
old = g.head(5)["name"].tolist()
print(f"\n상위5(노후 추정): {old}")
print(f"신축 많은 구 순위: " + ", ".join(
    f"{n}={list(g['name']).index(n)+1}위" for n in newish if n in list(g['name'])))

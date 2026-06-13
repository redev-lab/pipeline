import os, sys, time
sys.path.insert(0, os.getcwd()); sys.stdout.reconfigure(encoding="utf-8")
for line in open(".env",encoding="utf-8"):
    if "=" in line: k,v=line.strip().split("=",1); os.environ[k]=v
import pandas as pd
from redev.serve.api import build_serve_context, load_scores, report, screen
t0=time.time(); ctx=build_serve_context(); print(f"build_serve_context {time.time()-t0:.0f}s")
scores=load_scores()
# 데모 3주소용 PNU: 성북 고점 / 마포 고점 / 강남 저점
def pick(gu, high=True):
    s=scores[scores.sigungu==gu]
    return (s.nlargest(1,"score") if high else s.nsmallest(1,"score")).pnu.iloc[0]
cases=[("①성북(학습구)", pick("11290")), ("②마포(추론전용)", pick("11440")), ("③강남 저점(비대상)", pick("11680", high=False))]
L=[]
for name,pnu in cases:
    t0=time.time(); r=report(pnu, ctx, property_type="다세대", stage="사업시행인가"); dt=time.time()-t0
    rep=r.get("report",{}); h=rep.get("hallucination",{})
    st=r.get("stages",{})
    L.append(f"=== {name} PNU…{pnu[-8:]} ({dt:.1f}s) ===")
    L.append(f"  후보 {r.get('candidate')} | 점수 {r.get('b1_score')} | 요건 {(st.get('진단_요건',{}).get('result') or {}).get('path')}")
    L.append(f"  리포트 source={rep.get('source')} 환각불일치={h.get('unmatched')} ({'합격' if h.get('ok') else '★불합격'})")
    L.append(f"  리포트 앞120: {rep.get('report_text','')[:120]}")
# 스크리너
t0=time.time(); sc=screen(scores, gu="11440", min_pct=0.9, top_k=5); dt=1000*(time.time()-t0)
L.append(f"\n=== 스크리너(마포 상위10%) {dt:.0f}ms, {len(sc)}건 ===")
for s in sc[:3]: L.append(f"  …{s['pnu'][-8:]} 점수{s['score']:.3f} 백분위{s['score_pct']:.2f}")
open("_data/processed/_serve_e2e.txt","w",encoding="utf-8").write("\n".join(L))
print("\n".join(L)); print("DONE")

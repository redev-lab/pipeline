import os, sys, time
sys.path.insert(0, os.getcwd()); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from redev.serve.infer_districts import build_inference_scores
t0=time.time()
df=build_inference_scores(force_rebuild=True)
NM={"11290":"성북","11590":"동작","11380":"은평","11530":"구로","11440":"★마포","11680":"★강남"}
L=[f"6구 빌드 {time.time()-t0:.0f}s | 전 노드 {len(df)}",
   f"{'구':<8}{'노드수':>8}{'점수중앙':>9}{'≥0.5':>7}{'≥0.9':>7}{'std':>7}  (★마포·강남=학습밖 추론)"]
for c in ["11290","11590","11380","11530","11440","11680"]:
    s=df[df.sigungu==c]["score"]
    if len(s)==0: continue
    L.append(f"{NM[c]:<8}{len(s):>8}{s.median():>9.3f}{100*(s>=0.5).mean():>6.0f}%{100*(s>=0.9).mean():>6.0f}%{s.std():>7.3f}")
L.append("\n★inductive 가드: 마포·강남 분포가 4구와 같은 결(중앙·std 비슷)이면 일반화 성공;")
L.append("  전부 0(중앙~0,≥0.5~0%) 또는 전부 1(중앙~1,≥0.9~100%)로 미쳐 날뛰면 실패.")
# 강남 신축지대 점수 낮은지 표본(역삼 인근 저점수)
gn=df[df.sigungu=="11680"].nsmallest(3,"score")
L.append(f"강남 최저점 표본(신축지대 기대): "+", ".join(f'{r.pnu[-8:]} 점수{r.score:.2f} 노후{r.aging:.2f}' for _,r in gn.iterrows()))
open("_data/processed/_6gu_accept.txt","w",encoding="utf-8").write("\n".join(L))
print("\n".join(L)); print("DONE")

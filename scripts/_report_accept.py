"""Phase 7 ⑨ 수검 — run(with_report) 3주소 환각 diff(불일치 0) + 출처태그 + caveat 누락 + CPU시간.
분석 전용. → _data/processed/_report_accept.txt
"""
import os
import re
import sys
import time

sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
for line in open(".env", encoding="utf-8"):
    if "=" in line:
        k, v = line.strip().split("=", 1)
        os.environ[k] = v

from redev.llm.report import _all_caveats, _nums
from redev.models.baseline import prepare_baseline_matrix
from redev.orchestration.pipeline import build_context, run

t0 = time.time()
ctx = build_context()
aug = prepare_baseline_matrix()
code2name = {v: k for k, v in ctx.name2code.items()}
L = [f"build_context {time.time()-t0:.0f}s"]

# 서로 다른 구역 positive 3개로 주소 구성
pnus = [aug[aug.y == 1].iloc[i]["pnu"] for i in (0, 5000, 12000)]
addrs = []
for p in pnus:
    pr = ctx.parcels[ctx.parcels["pnu"] == p].iloc[0]
    addrs.append(f"{code2name[p[:5]]} {str(pr.get('dong_addr','')).split()[-1]} {pr.get('jibun','')}")

all_ok = True
for addr in addrs:
    t0 = time.time()
    r = run(addr, ctx, property_type="다세대", stage="사업시행인가", with_report=True)
    dt = time.time() - t0
    rep = r.get("report", {})
    h = rep.get("hallucination", {})
    text = rep.get("report_text", "")
    tags = len(re.findall(r"\[[^\]]+=[^\]]+\]", text))           # 출처 태그 [키=값]
    cav_in = sum(1 for c in _all_caveats(r) if c[:15] in text)   # caveat 포함 수
    cav_total = len(_all_caveats(r))
    ok = h.get("ok", False)
    all_ok = all_ok and ok
    L.append(f"\n=== '{addr}' ({dt:.1f}s, source={rep.get('source')}) ===")
    L.append(f"  ★환각: 불일치 {h.get('unmatched')} → {'합격(0)' if ok else '★불합격'}")
    L.append(f"  출처태그 {tags}개 / caveat 포함 {cav_in}/{cav_total}")
    L.append(f"  리포트 앞 200자: {text[:200]}")

L.append(f"\n=== 종합: 환각 3주소 모두 불일치 0 → {'★합격' if all_ok else '불합격'} ===")
open("_data/processed/_report_accept.txt", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
print("DONE")

"""⑤ 통합 검증 — 피클 재빌드(pnu_zone·zone_attrs) + 장위15 구역 리포트 재생성.
'얼마' 칸에 계획정보(용적률·세대수·출처)가 뜨는지 + 환각 0 확인. 실행: python scripts/_gosi_integrate_verify.py
"""
import os, sys, time
sys.path.insert(0, os.getcwd()); sys.stdout.reconfigure(encoding="utf-8")
for line in open(".env", encoding="utf-8"):
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.strip().split("=", 1); os.environ[k] = v
from redev.serve.api import load_serve_context, report

t = time.time(); ctx = load_serve_context(rebuild=True); print(f"[재빌드] {time.time()-t:.0f}s, 계획정보 {len(ctx.zone_attrs or {})}구역")
zid = "11290NTC202409250002"   # 장위15구역(1순위 정답지)
pnus = [p for p, z in (ctx.pnu_zone or {}).items() if z == zid]
print(f"장위15 필지 {len(pnus)}개")
if not pnus:
    print("★장위15 필지 0 — 매핑 실패"); sys.exit()
r = report(pnus[0], ctx, property_type="다세대", stage="사업시행인가")
rep = r.get("report", {}); h = rep.get("hallucination", {}); txt = rep.get("report_text", "")
pi = (r["stages"].get("진단_계획정보", {}) or {}).get("result")
print(f"\nPNU…{pnus[0][-8:]} candidate={r.get('candidate')} | 환각 ok={h.get('ok')} 불일치={h.get('unmatched')}")
print("계획정보 stage:", "있음" if pi else "★없음")
print("\n--- '얼마' 칸 본문 발췌 ---")
import re
lines = txt.splitlines()
for i, ln in enumerate(lines):
    if "얼마" in ln or "용적률" in ln or "세대" in ln or "계획정보" in ln or "시세" in ln:
        print("  " + ln.strip()[:200])
print("\n계획정보 fact 노출(NTC 원시코드 없어야):", "NTC" not in txt)
print("DONE")

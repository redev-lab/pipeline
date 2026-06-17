"""⑤+#2 통합 검증 — 피클 재빌드(resolve 매칭) + 가리봉1(새 매칭)·장위15(단위) 리포트.
'얼마' 칸 계획정보 단위 표시 + 가리봉1 연결 + 환각 0 확인. 실행: python scripts/_gosi_integrate_verify.py
"""
import os, sys, time
sys.path.insert(0, os.getcwd()); sys.stdout.reconfigure(encoding="utf-8")
for line in open(".env", encoding="utf-8"):
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.strip().split("=", 1); os.environ[k] = v
from redev.serve.api import load_serve_context, report

t = time.time(); ctx = load_serve_context(rebuild=True); print(f"[재빌드] {time.time()-t:.0f}s, 계획정보 {len(ctx.zone_attrs or {})}구역")
TARGETS = [("장위15(단위표시)", "11290NTC202409250002"), ("가리봉1(구역명 매칭)", "11530NTC202506110005")]
for label, zid in TARGETS:
    pnus = [p for p, z in (ctx.pnu_zone or {}).items() if z == zid]
    if not pnus:
        print(f"\n{label}: ★필지 0 (매칭/pnu_zone 실패)"); continue
    r = report(pnus[0], ctx, property_type="다세대", stage="사업시행인가")
    rep = r.get("report", {}); h = rep.get("hallucination", {}); txt = rep.get("report_text", "")
    pi = (r["stages"].get("진단_계획정보", {}) or {}).get("result")
    print(f"\n=== {label} PNU…{pnus[0][-8:]} | 환각 ok={h.get('ok')} 불일치={h.get('unmatched')} ===")
    print(f"  계획정보 매칭: {pi['match'] if pi else '★없음'}")
    for ln in txt.splitlines():
        if any(k in ln for k in ("용적률", "세대", "면적", "건폐율", "계획정보")):
            print("  본문:", ln.strip()[:220]); break
print("\nDONE")

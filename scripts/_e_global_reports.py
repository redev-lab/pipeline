"""E 전역(서브셋) 컨텍스트 4주소 리포트 재생성 — B 새 표시(상위/하위·raw순위·표시명) +
계약 본문 육안 + 환각 0 유지. ★메모리 안전: pickle 없이 in-process 빌드 + ctx에서 PNU 픽.

대상: 성북(학습·high) / 마포(추론·high) / 강남(비대상·low) / 용산(비학습·high — D-1 일반화 쇼케이스).
실행: python scripts/_e_global_reports.py
"""
import os, sys, time
sys.path.insert(0, os.getcwd()); sys.stdout.reconfigure(encoding="utf-8")
for line in open(".env", encoding="utf-8"):
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.strip().split("=", 1); os.environ[k] = v
import numpy as np
from redev.serve.api import build_serve_context, report

t0 = time.time(); ctx = build_serve_context(); print(f"build_serve_context(서브셋) {time.time()-t0:.0f}s")
sig = ctx.parcels.set_index("pnu")["sigungu"]                  # pnu→구


def pick(code, high=True):
    pnus = [p for p in sig[sig == code].index if p in ctx.pnu_to_idx]
    arr = [(p, ctx.scores[ctx.pnu_to_idx[p]]) for p in pnus]
    arr.sort(key=lambda x: x[1], reverse=high)
    return arr[0][0]


cases = [("①성북(학습·high)", pick("11290")), ("②마포(추론·high)", pick("11440")),
         ("③강남(비대상·low)", pick("11680", high=False)), ("④용산(비학습·high)", pick("11170"))]
for name, pnu in cases:
    r = report(pnu, ctx, property_type="다세대", stage="사업시행인가")
    rep = r.get("report", {}); h = rep.get("hallucination", {}); txt = rep.get("report_text", "")
    fe = (r["stages"].get("예언_환경점수", {}) or {}).get("result") or {}
    phrase = fe.get("rank_phrase")
    cav_u = (r.get("report") or {}).get("caveats_user") or []
    panel_codes = [c for c in ("R15", "R13", "R4", "§", "★") if any(c in x for x in cav_u)]
    print("\n" + "=" * 78)
    print(f"{name}  PNU…{pnu[-8:]}  candidate={r.get('candidate')}  raw={r.get('b1_score')}  신뢰도={r.get('confidence')}")
    print(f"  헤더 환경점수(=rank_phrase)='{phrase}' / 본문 결론 phrase 일치={phrase in txt}")
    print(f"  verdict={r.get('verdict', {}).get('class')} | 환각 ok={h.get('ok')} 불일치={h.get('unmatched')}")
    print(f"  본문 내부코드={[c for c in ('R15','R13','§','★','NTC') if c in txt] or '없음'} | 패널 내부코드={panel_codes or '없음'}")
    print("-" * 78)
    print(txt)
print("\nDONE")

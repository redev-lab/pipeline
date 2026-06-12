"""계약 v1.1 수검 — 3주소 리포트 재생성 + 환각 0 + 6대 결함 육안 검사.

★강남(비대상)에 일부러 stage='사업시행인가'를 줘서 단계 누수(결함3)가 막혔는지 본다.
실행: python scripts/_report_contract_e2e.py
"""
import os, sys, time
sys.path.insert(0, os.getcwd()); sys.stdout.reconfigure(encoding="utf-8")
for line in open(".env", encoding="utf-8"):
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.strip().split("=", 1); os.environ[k] = v
from redev.serve.api import build_serve_context, load_scores, report

t0 = time.time(); ctx = build_serve_context(); print(f"build_serve_context {time.time()-t0:.0f}s")
scores = load_scores()


def pick(gu, high=True):
    s = scores[scores.sigungu == gu]
    return (s.nlargest(1, "score") if high else s.nsmallest(1, "score")).pnu.iloc[0]


cases = [
    ("①성북 후보(학습구)", pick("11290"), "다세대", "사업시행인가"),
    ("②마포 후보(추론전용)", pick("11440"), "다세대", "사업시행인가"),
    ("③강남 비대상(stage 일부러 줌)", pick("11680", high=False), "다세대", "사업시행인가"),
]
for name, pnu, pt, stage in cases:
    r = report(pnu, ctx, property_type=pt, stage=stage)
    rep = r.get("report", {}); h = rep.get("hallucination", {})
    txt = rep.get("report_text", "")
    print("\n" + "=" * 78)
    print(f"{name}  PNU…{pnu[-8:]}  candidate={r.get('candidate')}  b1={r.get('b1_score')}")
    print(f"verdict.class = {r.get('verdict', {}).get('class')}")
    print(f"환각 ok={h.get('ok')} 불일치={h.get('unmatched')} | source={rep.get('source')}")
    # 결함 자동 점검
    bad_codes = [c for c in ("R15", "R13", "R18", "R4", "§", "★", "39%", "면책") if c in txt]
    print(f"내부코드 잔존(결함4)={bad_codes or '없음'}")
    leak = ("잔여" in txt and not r.get("candidate"))
    print(f"비후보 단계 누수(결함3)={'★누수!' if leak else '없음'}")
    print("-" * 78)
    print(txt)
print("\nDONE")

"""v1_3-gosi 수검 — 표본 5구역 추출+검증 (라이브 LLM). 설계 §7.
① verbatim 통과율 ② 범위가드 ③ 손대조(장위15 1순위) ④ 커버리지 ⑤(리포트 반영은 통합 후).
실행: python scripts/_gosi_run.py
"""
import os, sys, time
sys.path.insert(0, os.getcwd()); sys.stdout.reconfigure(encoding="utf-8")
for line in open(".env", encoding="utf-8"):
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.strip().split("=", 1); os.environ[k] = v
from redev.data.ingest.gosi_body import is_scanned, read_gosi
from redev.nlp.gosi_extract import ATTRS, extract_attrs
from redev.nlp.gosi_verify import RANGES, verify_extraction

DIR = "_data/raw/고시정보/"
# ★손대조 우선순위: 장위15 1순위(후속 변경 우려 덜·절대값 표 확실), 흑석2 보조(최신 미반영 플래그)
SAMPLES = [
    {"n": "2024-448", "zone": "장위15구역", "date": "2024-09-19", "role": "★1순위 정답지",
     "file": "[서고시 제2024-448호] 장위재정비촉진지구 변경 지정, 재정비촉진계획(장위15구역) 변경결정 및 지형도면 고시(2024. 9. 19.)성북구.pdf", "flags": []},
    {"n": "2025-426", "zone": "흑석2구역", "date": "2025-07-31", "role": "보조 정답지",
     "file": "서울특별시_제2025-426호_고시.pdf", "flags": ["최신 미반영(서울시 2025-659 변경안 협의중·미입수)"]},
    {"n": "2024-519", "zone": "청파2구역", "date": "2024-10-31", "role": "신규지정", "file": "용산2024-519.pdf", "flags": []},
    {"n": "2026-18", "zone": "가리봉1구역", "date": "2026-02-12", "role": "경미한변경(전체표)", "file": "구로구_제2026-18호_고시.pdf", "flags": []},
    {"n": "2020-133", "zone": "응암제2구역", "date": "2020-07-30", "role": "옛 양식(디지털)", "file": "응암.pdf", "flags": []},
]

cov = {a: 0 for a in ATTRS}
print(f"범위 가드: {RANGES}\n")
for s in SAMPLES:
    t0 = time.time()
    g = read_gosi(DIR + s["file"]); text, rows, grids = g["text"], g["rows"], g["grids"]
    scan = is_scanned(text)
    ex = extract_attrs(text, zone_name=s["zone"], 고시번호=s["n"], 고시일자=s["date"])
    vr = verify_extraction(ex, text, table_rows=rows, grids=grids)
    print("=" * 78)
    print(f"{s['zone']} (고시 {s['n']}, {s['date']}) — {s['role']}  [{len(text):,}자, {'스캔' if scan else '디지털'}, {time.time()-t0:.0f}s]")
    if s["flags"]:
        print("  ⚠ 플래그:", " / ".join(s["flags"]))
    print(f"  요약: {vr['summary']}")
    for a in ATTRS:
        r = vr["results"][a]
        if r["grade"] == "missing":
            print(f"  · {a}: 미기재")
            continue
        if r["grade"] == "verified":
            cov[a] += 1
        snip = (r.get("sentence") or "")[:70].replace("\n", " ")
        print(f"  · {a}: [{r['grade']}] {r.get('raw')} (값 {r.get('value')}, {r.get('label')}, {r.get('변경구분')})")
        print(f"      손대조 근거: …{snip}…")
print("\n" + "=" * 78)
print("④ 커버리지(5구역 중 verified):", {a: f"{cov[a]}/5" for a in ATTRS})
print("DONE")

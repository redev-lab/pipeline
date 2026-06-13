"""재직렬화 + B-3 표시명 검증. load_serve_context(rebuild=True) → 새 pickle(zone_vectors에
display_name 포함) → 후보 1건 리포트 유사사례가 '○○동 일대 (연도)'로 뜨는지 확인.
실행: python scripts/_reserialize_verify.py
"""
import os, sys, time, pickle
sys.path.insert(0, os.getcwd()); sys.stdout.reconfigure(encoding="utf-8")
for line in open(".env", encoding="utf-8"):
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.strip().split("=", 1); os.environ[k] = v
from redev.serve.api import _CTX_CACHE, load_serve_context, report

# (정보용) 기존 pickle이 B-3 이전이었는지 — zone_vectors meta에 display_name 키 유무
if _CTX_CACHE.exists():
    with open(_CTX_CACHE, "rb") as f:
        old = pickle.load(f)
    had = "display_name" in (old.zone_vectors.meta[0] if old.zone_vectors.meta else {})
    print(f"[기존 pickle] display_name 포함={had} (False면 B-3 이전 — 원시코드 재출현 위험 실재)")
    del old

t = time.time(); ctx = load_serve_context(rebuild=True); print(f"[재직렬화] 빌드+저장 {time.time()-t:.0f}s")
has = "display_name" in (ctx.zone_vectors.meta[0] if ctx.zone_vectors.meta else {})
sample = [m.get("display_name") for m in ctx.zone_vectors.meta[:3]]
print(f"[새 pickle] display_name 포함={has} / 표본 {sample}")

# 후보 주소 1건(성북 high) 리포트 → 유사사례 본문 확인
sig = ctx.parcels.set_index("pnu")["sigungu"]
cand = [(p, ctx.scores[ctx.pnu_to_idx[p]]) for p in sig[sig == "11290"].index if p in ctx.pnu_to_idx]
pnu = max(cand, key=lambda x: x[1])[0]
r = report(pnu, ctx, property_type="다세대", stage="사업시행인가")
match = ((r.get("retrieval") or {}).get("matches") or [{}])[0]
txt = r.get("report", {}).get("report_text", "")
import re
sim_line = next((l for l in txt.splitlines() if "유사" in l), "(유사사례 줄 없음)")
print(f"\n성북 후보 PNU…{pnu[-8:]} candidate={r.get('candidate')}")
print(f"  retrieval match: display_name='{match.get('display_name')}' zone_id(메타)='{match.get('zone_id')}'")
print(f"  본문 유사사례 줄: {sim_line.strip()}")
print(f"  ★원시 zone_id(NTC) 본문 노출={'NTC' in txt} / 표시명 본문 포함={(match.get('display_name') or 'X') in txt}")
print("DONE")

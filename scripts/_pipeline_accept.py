"""Phase 6 엔드투엔드 수검 — run() 통과 + 릴레이 IoU(시스템 vs 부품) + CPU시간 + 부분실패.
분석 전용. → _data/processed/_pipeline_accept.txt
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
from redev.config import load_infer_config
from redev.eval.iou import compare_methods
from redev.models.baseline import load_training_matrix, prepare_baseline_matrix
from redev.models.infer import candidate_clusters
from redev.orchestration.pipeline import address_to_pnu, build_context, run
from redev.rules.stage1 import score_cluster

t0 = time.time()
ctx = build_context()
L = [f"build_context {time.time()-t0:.0f}s | 후보 클러스터 {len(set(id(c) for c in ctx.pnu_cluster.values()))}개"]

aug = prepare_baseline_matrix()
tm = load_training_matrix()
# 실제 4구 주소 구성(positive 필지에서)
pos_pnu = aug[aug.y == 1]["pnu"].iloc[0]
prow = ctx.parcels[ctx.parcels["pnu"] == pos_pnu].iloc[0]
code2name = {v: k for k, v in ctx.name2code.items()}
addr = f"{code2name[pos_pnu[:5]]} {str(prow.get('dong_addr','')).split()[-1]} {prow.get('jibun','')}"
L.append(f"\n=== ① run('{addr}') 엔드투엔드 ===")
r = run(addr, ctx, property_type="다세대", stage="사업시행인가")
L.append(f"PNU {r.get('pnu')} | B1점수 {r.get('b1_score')} | 후보 {r.get('candidate')}")
for k, v in r.get("stages", {}).items():
    st = v.get("status")
    if k == "진단_요건" and st == "ok":
        L.append(f"  {k}: {v['result']['path']} (housing={v['result']['housing_eligible']})")
    elif k == "진단_시세맥락":
        rr = v.get("result", {})
        L.append(f"  {k}[{st}]: 대지지분 {rr.get('land_share_pyung_man')} / 신축 {rr.get('newbuild_exclu_pyung_man')}")
    elif k == "예언_환경점수" and st == "ok":
        L.append(f"  {k}: {v['result']['label']} 상위 {v['result']['rank_top_pct']}%")
    elif k == "진입_eligibility" and st == "ok":
        t = v["result"]["진단_토허"]
        L.append(f"  {k}: 토허 {t['toheo_applies']}·갭 {t['gap_investment_possible']}")

# ② 주소파싱: 도로명 친절에러 + PNU 직접
L.append("\n=== ② 주소 파싱 ===")
try:
    address_to_pnu("성북구 보문로 100", ctx)
    L.append("  도로명: ★에러 안 남(문제)")
except ValueError as e:
    L.append(f"  도로명 친절에러 ✓: {str(e)[:40]}")
L.append(f"  PNU 직접입력 ✓: {address_to_pnu(pos_pnu, ctx) == pos_pnu}")

# ③ CPU 추론시간(주소당)
sample = aug["pnu"].sample(30, random_state=0).tolist()
t0 = time.time()
for p in sample:
    run(p, ctx, property_type="아파트", stage="관리처분인가")
L.append(f"\n=== ③ CPU 추론시간: {1000*(time.time()-t0)/len(sample):.1f} ms/주소 (캐시 후 조회) ===")

# ④ 부분 실패: 거래 없는 PNU → 시세맥락 skipped, 안 죽음
no_trade = [p for p in tm.pnu_to_idx if p not in ctx.target.index][:1]
if no_trade:
    rr = run(no_trade[0], ctx)
    L.append(f"=== ④ 부분실패 견고성: 거래없는 PNU → 시세맥락 status={rr['stages']['진단_시세맥락']['status']} (run 생존 ✓) ===")

# ⑤ ★릴레이 IoU: B1 넓은 vs B1→stage1 통과 후 (시스템 vs 부품)
L.append("\n=== ⑤ ★릴레이 IoU (시스템 vs 부품) ===")
cfg = load_infer_config()
wide = candidate_clusters(ctx.scores, ctx.pnu_to_idx, ctx.edge_index, thr=ctx.thr, min_nodes=cfg["cluster"]["min_nodes"])
relay = []
for cl in wide:
    try:
        if score_cluster(cl, ctx.parcels, ctx.buildings)["path"] in ("재개발", "모아타운·소규모정비"):
            relay.append(cl)
    except Exception:
        pass
zones = {z: set(g["pnu"]) for z, g in aug[aug.y == 1].groupby("zone_id")}
res = compare_methods({"부품 B1넓은": wide, "시스템 B1→stage1": relay}, zones)
for m, rr in res.items():
    L.append(f"  {m:<18} IoU {rr['mean_iou']:.3f} 핵심부 {rr['mean_core_capture']:.3f} 클러스터 {rr['n_clusters']} 평균크기 {rr['avg_cluster_size']:.0f}")

open("_data/processed/_pipeline_accept.txt", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
print("DONE")

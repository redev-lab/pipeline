"""전역 컨텍스트 빌드 + 직렬화 측정 (C③). 실행: python scripts/_build_ctx_measure.py

빌드 시간·피클 용량·재로드 시간·RSS 메모리를 측정한다(6구 228s 대비). .env 로드(거래 키).
"""
import os, sys, time
sys.path.insert(0, os.getcwd()); sys.stdout.reconfigure(encoding="utf-8")
for line in open(".env", encoding="utf-8"):
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.strip().split("=", 1); os.environ[k] = v
import psutil
from redev.serve.api import _CTX_CACHE, load_serve_context

proc = psutil.Process()
print(f"[mem] 시작 RSS {proc.memory_info().rss/1e6:.0f}MB")
t0 = time.time()
ctx = load_serve_context(rebuild=True)                      # 빌드 + 직렬화 저장
print(f"[build] 전역 컨텍스트 빌드+저장 {time.time()-t0:.0f}s | 피크 RSS {proc.memory_info().rss/1e6:.0f}MB")
print(f"[pickle] {_CTX_CACHE.stat().st_size/1e6:.0f}MB | 노드 {len(ctx.scores):,} | 클러스터필지 {len(ctx.pnu_cluster):,}")

del ctx
t1 = time.time()
ctx2 = load_serve_context()                                # 재로드(직렬화 경로)
print(f"[reload] 직렬화 로드만 {time.time()-t1:.0f}s | RSS {proc.memory_info().rss/1e6:.0f}MB")
print(f"[check] name2code {len(ctx2.name2code)}구 | zone_vectors {len(ctx2.zone_vectors.meta)}구역")
print("DONE")

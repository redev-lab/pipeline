"""#3-c 후속: national-학습 모델로 infer_scores + serve_ctx 재계산. (매트릭스 캐시는 이미 national)"""
import sys, io, time, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
t0 = time.time()
from redev.serve.infer_districts import build_inference_scores
build_inference_scores(force_rebuild=True, log=lambda m: print(m, flush=True))
print(f"infer_scores done ({time.time()-t0:.0f}s)", flush=True)
from redev.serve.api import load_serve_context
ctx = load_serve_context(rebuild=True, log=lambda m: print(m, flush=True))
print(f"serve_ctx done buildings {len(ctx.buildings):,} ({time.time()-t0:.0f}s)", flush=True)
print("REBUILD_RETRAINED_DONE", flush=True)

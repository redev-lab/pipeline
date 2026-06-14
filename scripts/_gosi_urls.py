"""우선순위 고시 → 공식 URL 매니페스트 (토지이음 고시목록 인덱스 기반). ★사용자 로컬 실행.

선행: data.go.kr 15083101(토지이음 고시목록) CSV를 받아 아래 경로에 두거나, API로 인덱스 적재.
실행: python scripts/_gosi_urls.py [CSV경로]
→ 13개 우선순위(+완료 5)의 고시번호·고시명·URL 출력 → 그 URL을 클릭해 본문 PDF 수동 다운로드.
"""
import os, sys
sys.path.insert(0, os.getcwd()); sys.stdout.reconfigure(encoding="utf-8")
from redev.data.ingest.gosi_index import load_index, urls_for
from redev.paths import DATA

DEFAULT_CSV = DATA / "raw/추가데이터/토지이음_고시목록.csv"   # 여기에 받은 CSV를 두면 됨

# 1차 배치 우선순위 13(학습4구 + 신통/공공 활성 + 최근 풀플랜)
PRIORITY = ["2024-475", "2025-179", "2025-178", "2025-194", "2025-245", "2025-456",
            "2025-551", "2024-484", "2025-163", "2025-447", "2025-205", "2025-226", "2025-287"]
DONE = ["2024-448", "2025-426", "2024-519", "2026-18", "2020-133"]   # 이미 확보(참고)

csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
if not os.path.exists(csv):
    print(f"★CSV 없음: {csv}\n  data.go.kr 15083101(토지이음 고시목록) CSV를 받아 이 경로에 두세요(무로그인 다운로드).")
    sys.exit(1)
idx = load_index(csv)
print(f"인덱스 {len(idx):,}건 로드\n")
for title, nums in [("=== 1차 배치 13(다운로드 대상) ===", PRIORITY), ("=== 완료 5(참고) ===", DONE)]:
    print(title)
    for m in urls_for(nums, idx):
        print(f"  {m['고시번호']:<10} {m['url'] or '★인덱스 미발견(연간 갱신 전?)'}\n     {str(m['고시명'] or '')[:60]}")
    print()

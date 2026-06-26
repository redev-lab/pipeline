# DB 현황 + DB 전환 트리거 (2026-06-23)

> _experiments/gis_swap/ 전용 조사 기록, redev/·git 미변경. "풀 설계 전에 지금 데이터가 어디 사는지"부터.

## #1 데이터 저장 구조 — 순수 파일 기반
- redev 데이터(피처·필지·건물)는 **전부 parquet/SHP/csv 디스크 + `serve_ctx.pkl`(157MB) 메모리 로드**.
- 추론 = 파일 로드 + **사전계산된 점수 조회**(`infer_scores.parquet` 21MB, 25구). 실시간 DB 쿼리 없음.
- ★**PostGIS/Postgres 연결 코드 없음**(psycopg·sqlalchemy·create_engine grep 0건). api.py:26의 "전역 /report는 v2
  PostGIS 온디맨드"는 **미래 계획 주석**이지 현 구현 아님.
- R10(PostGIS 메모리·실시간 추출) 이슈는 **사전계산(parquet)으로 회피된 상태**.

## #2 Supabase — 순수 Auth + 회원 데이터 전용
- 테이블 = `profiles`(user id) + `watchlist`(user_id, 관심주소)뿐. **redev 건물/피처 데이터 일절 없음.**
- backend는 `redev.serve`를 직접 import(단일 파이썬 프로세스, 메모리 pickle). Supabase는 JWT 검증·회원만.
- → **2개 분리 세계**: Supabase Postgres(회원 1인스턴스) ↔ 파일+메모리(redev 데이터). 데이터 레벨 비연결.

## #3 규모 추정
| | 현재 | national swap 후 |
|---|---|---|
| 건물 행 | 695,769(서울 4유형) | **451k(national) + ~60k(backfill) ≈ 511k** — 더 작음 |
| 필지(serve_ctx) | 7구 291,891 | 동일(전역화는 별건) |
| 점수 캐시 | 25구 ~21MB parquet | 동일 규모 |
| pickle | 157MB | 비슷~약간 작음 |

- swap은 건물 소스가 **오히려 작아짐**. 사전계산 방식이 이미 25구를 처리(infer_scores 25구 존재) → 검증됨.
- 전역(25구) parcels-geometry(~860k, pickle ~400-500MB 추정)는 **전역화 이슈**지 swap 이슈 아님.

## #4 판정 — DB 재설계 선행 불필요
- national swap = **건물 소스 파일 교체(national SHP) + backfill parquet 추가 + 사전계산 재실행**
  (`build_inference_scores`/`build_serve_context`). 현 파일/parquet/pickle 구조 안에서 완결.
- 데이터 규모 비슷~작음 + 사전계산이 이미 25구 처리 → **현 구조로 swap 적용 가능.**
- PostGIS(v2)는 전역 실시간 서브그래프 스케일링 별도 주제로 swap과 **직교** — swap이 DB 재설계를 트리거하지 않음.

## ★언제 PostGIS(v2)로 가야 하나 — 전환 트리거
아래 중 **하나라도 닿으면** 그때 DB(PostGIS) 설계에 착수. 그 전까지는 파일+사전계산 구조 유지.

1. **전역 25구 실시간 서브그래프(2-hop) 온디맨드** — 사전계산 캐시로 못 덮는 임의 주소의 실시간 서브그래프
   추출이 필요해질 때(현재는 점수 사전계산으로 회피 중).
2. **메모리/콜드스타트 한계** — 사전계산 pickle이 전역화로 **157MB → 400-500MB**로 커져 메모리 압박·콜드스타트
   지연이 실측 한계를 칠 때.
3. **동시성·부하** — 동시 사용자·쿼리 부하가 **단일 파이썬 프로세스 + pickle**로 안 버틸 때(수평 확장·공유 캐시 필요).

→ ★**현재 셋 다 미해당** = 파일 구조 유지가 정답. national swap도 이 구조로 진행. DB는 위 트리거가 켜지는 시점에.

## 결론
현 구조(파일 + 사전계산 parquet + 메모리 pickle)로 **national swap 적용 가능, DB 재설계는 선행 조건 아님**.
Supabase는 회원 전용이라 swap과 무관. PostGIS 전환은 위 3트리거 중 하나가 켜질 때의 v2 사안.

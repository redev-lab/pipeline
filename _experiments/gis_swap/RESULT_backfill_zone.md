# #3-b-2 zone_vectors backfill (동작) — 회복 실패 + 원인 (2026-06-23)

> _experiments/gis_swap/ 측정, 현 national+backfill 산출물 유지, redev/·git 미변경.
> 가설: 닮은동네 하락 원인이 zone_vectors 기준구역 왜곡 → 그 구역만 backfill하면 회복.

## 적용
- 동작 backfill(`backfill_useapr.parquet`, 2,994 PNU)을 building_gis national 로드에 통합(`backfill_path`),
  serve_ctx/infer_scores 재계산(zone_vectors 포함). redev/ 반영(load_buildings_national + 2 callers).

## 결과 — ★회복 안 됨
| | 동작 닮은동네(서울 overlap 기준) |
|---|---|
| national(backfill 전) | 2.4/5 · 1위 50% |
| **national + 동작 backfill** | **2.4/5 · 1위 50%** (불변) |

## 원인 (3중)
**동작 zone 멤버 4,913 분해:** national 건물有 29% · **없음 71%(3,485)** · 서울有 62%.
1. **타깃 어긋남**: national 누락 zone 멤버 3,485 중 backfill csv 타깃은 39%(1,360). 나머지 2,125는 서울에도 건물
   없는 도로·공터(backfill 불가) 위주.
2. **수율**: 타깃 1,360 중 표제부 사용승인일 채워진 건 52% → 실제 복원 ~700(누락의 20%).
3. **★집계 둔감 + 랭킹 민감**: zone 노후도는 이미 안정(national 보유 건물도 노후), 부분 복원이 zone_vector를 거의 안
   움직임. 게다가 닮은동네 top5가 51구역 4D 공간에 몰려 있어 미세 변화에도 순위 reshuffle.

## 결론 — zone측 backfill로 "서울 일치"는 구조적으로 불가
- 볼륨 문제 아님: 누락 절반은 backfill 불가(도로·공터), 수율 52%, 결정적으로 랭킹이 미세 perturbation에 민감.
- → "서울 top5 일치" 추구는 막다른 길. ★**기준 자체가 틀림**(서울도 불완전본). 절대 타당성으로 전환 →
  `RESULT_similar_validity.md`에서 national 닮은동네가 **서울만큼 타당**함 확인 → zone backfill 불필요 결론.

## 남긴 것
- 코드 변경(load_buildings_national backfill_path + callers)은 **유지**(향후 backfill 확대 시 인프라로 재사용).
  동작 backfill 통합 자체는 점수/커버리지엔 소폭 기여(닮은동네 회복엔 무효).

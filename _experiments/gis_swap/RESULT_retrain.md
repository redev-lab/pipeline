# #3-c B1+ national 재학습 (2026-06-24)

> 본 파이프라인 수정. 건물 소스 national 교체로 학습 매트릭스를 national 피처로 재생성 → train-serve skew 해소.

## 변경 (redev/)
- `baseline.py`: import에 `load_buildings_national`. `_assemble`(학습 매트릭스)·`_load_parcels_buildings`(이웃집계)
  둘 다 **national+backfill**로 교체 → train_matrix가 national 피처로 생성됨.
- train_matrix.parquet · train_matrix_nb.parquet 재생성(national). positives 15,403 보존(라벨 동일).
- infer_scores.parquet · serve_ctx.pkl 재계산(national-served + **national-trained**).

## ★검증
**[1] train-serve skew 해소 — oof PR-AUC (공간CV)**
- 서울 학습/서울 피처(기존): **0.940**
- national 학습/national 피처(재학습): **0.954** (VERDICT OK, +0.014)
- → national-trained 모델이 national 라벨을 **서울 때만큼(오히려 더) 잘 적합** = 학습·서빙 분포 일치, skew 해소.

**[2] 닮은동네 타당성 유지** (재학습은 zone_vectors 불변 — 모델만 바뀜)
- 데모 1위 코사인: 노량진 0.930 · 응암 0.835 — 타당(재학습이 닮은동네 안 망침). case_search는 모델 비의존.

**[3] 데모 환경점수 정상**
- 정릉동 상위 70%(약), 노량진(동작) 상위 30%·top10% 837/4427(강), 응암 상위 48%. 노후·재개발 강도와 부합.

**[4] 동작 재학습 영향**
- national-train vs seoul-train(국) Spearman **0.739** — 재학습이 동작 점수를 national 적합 방향으로 재배치.
  상위10% candidate 수 3,056→3,060(수 유지, 구성 재배치) = skew 보정.

**[5] 기존 테스트** — 128 passed.

## 산출물·백업 상태
- 현행: train_matrix(national) · infer_scores/serve_ctx(national-served, national-trained).
- 백업: `*.seoul_bak`(서울) · `*.nat_nobackfill`(national, backfill 전) · `*.nat_seoultrain`(national-served, 서울-trained).

## 결론
#3-c 완료. B1+가 national 피처로 재학습돼 train-serve skew 해소(PR-AUC 0.954≥0.940). 닮은동네 타당성·데모·테스트
이상 없음. **건물 소스 1유형 swap 본 파이프라인 적용(#3-b-1·#3-b-2·#3-c) 일단락.**

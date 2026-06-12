# 설계 노트 — v1.2 PU 학습 (uncertain 활용, 과대예측 직접 겨냥)

> 규칙1: 코드 전에 이 노트. 승인 후 구현. ★타임박스 1세션 — 효과 유무를 측정으로 닫는다.

## 0. ★PU의 함정 (이 노트의 심장 — 잘못하면 제품을 죽인다)

uncertain(노후미지정 5.6만)은 **"음성"이 아니라 "미래의 양성이 섞인 미지"**다. *내일 지정될
동네*가 그 안에 있고, **그게 바로 우리 제품이 찾으려는 것**. uncertain을 hard negative로 박으면
미래 후보를 죽이는 모델이 된다.

→ 목표는 uncertain 점수를 **0으로 누르는 게 아니라**, "지정스러움"의 **순위 변별**을 세우는 것.
지금 모델은 "old-지정 vs old-미지정" 대조를 본 적이 없다(uncertain을 학습서 제외했으니) → 전부
old면 양성으로 본다(81% 과대예측). PU는 그 대조를 *약하게* 가르쳐 순위를 세운다. 신규 데이터 0원.

## 1. 방법 사다리 (싼 것부터, 측정으로 결정 — P2까지만)

- **P1 — uncertain 저가중 음성**: 학습에 uncertain을 음성으로 넣되 `sample_weight`를 낮게
  (positive·reliable_neg=1.0, uncertain=w). ★w를 config로 0.1/0.3/0.5 비교 — 함정(§0) 때문에
  낮게. 가장 단순. XGBoost `sample_weight` 한 줄.
- **P2 — spy 기법**(P1이 부족하면): positive 일부(예 10%)를 uncertain에 숨겨 학습 → 그 spy들이
  받는 점수 분포로 임계 설정 → 임계 아래(=진짜 음성스러운) uncertain만 **reliable negative로 승격**,
  나머지는 학습 제외(미지로 존중). 함정을 정직하게 다루는 표준 PU.
- ★**P2까지만.** bagging PU·반복 self-training 등은 백로그(타임박스).

## 2. 데이터 — uncertain 포함 PU 행렬

현 학습행렬은 uncertain을 drop(39,741). PU는 **uncertain 포함**(96,224)이 필요. → `_assemble`에
`keep_uncertain` 경로 추가(필터만 끔, v1.1 features 그대로). uncertain 피처는 t=현재(파생 시점,
labels §4-4). 이웃집계도 재계산(96K). production 피처셋(B1+ −용도지역) 그대로 사용.

## 3. 평가 — 같은 장비 + ★PU 전용 4지표 + 합격선

전부 ★같은 LODO + zone-block bootstrap. 격전지 recall도 병기(v1.1 0.712 유지 확인).

| # | 지표 | 합격선/방향 |
|---|---|---|
| ① | **known positive recall** | ★기존 양성 유지 — recall 하락 **3%p 이내**(억제 부작용 가드) |
| ② | **과대예측률**(전 노드 ≥thr) | 81%에서 **내려간 만큼이 성과**(절대 합격선 없음, 방향만) |
| ③ | **IoU / 핵심부 포착** | ★클러스터가 지정구역 쪽으로 조여지나 — **0.300 대비 개선**이 "올바른 방향" 증거 |
| ④ | hard 해제 FPR(n=24) | 참고만(통계 무의미) — old-미지정 학습의 부수 효과 |

## 4. ★채택 규칙 (타임박스 종료 조건)

- **①합격(recall −3%p 이내) + ③개선(IoU>0.300)** → production 교체 검토: infer·feasibility 재배선
  + ★보정 재적합(calibrate). w는 ②·③ 최적값.
- **아니면** → "PU v1 기법(P1/P2)으론 불충분, 라벨 확장(v2)이 답"을 **측정 기록으로 닫고** 그대로
  Phase 8. (모델 사이클 무한 반복 금지 — 측정이 '안 된다'면 그것도 결과.)

## 5. 함수/파일 분해

```text
models/pu.py
├── load_pu_matrix(*, force_rebuild)         # uncertain 포함 행렬(keep_uncertain) + nb
├── pu_weights(certainty, *, w)              # sample_weight(pos·neg=1, uncertain=w)
├── run_pu_cv(aug_pu, *, w, spy=False)       # LODO + 가중학습 → ①②③④ 지표
└── spy_promote(aug_pu, *, spy_frac)         # P2: spy로 음성스러운 uncertain 승격
eval: 기존 spatial_cv·iou·metrics 재사용(같은 장비). 신규 지표는 얇은 래퍼.
```

## 6. ★수검 (규칙9)

1. **함정 가드**: uncertain 점수가 일괄 0으로 눌리지 않는지(분포 확인 — 일부는 높아야 = 미래후보).
2. **신구 4지표 표**: v1.1(uncertain 제외) vs PU(w 3종) — recall·과대예측·IoU·격전지.
3. **시점정합 유지**: positive as-of-t 그대로(PU가 t 누수 안 만드는지).
4. **채택/기각 명문화**: 채택 규칙(§4)에 따라 결론 1줄 + 근거.

## 7. 검토했지만 버린 대안

- **uncertain = hard negative(w=1)**: §0 함정 — 미래 후보 죽임. 저가중만.
- **bagging PU·반복 self-training**: 타임박스 초과 → 백로그.
- **새 평가 장비**: 신구 비교 공정성 — 기존 LODO·bootstrap 재사용.

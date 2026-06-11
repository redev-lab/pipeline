# 설계 노트 — `eval/spatial_cv.py` (Phase 3, R3 공간 검증)

> 규칙1: 코드 전에 이 노트. 승인 후 구현. CLAUDE.md §7 우선.

## 0. 한 문단 요약

42개 positive 구역이 4개 자치구에 몰려 있고 필지는 강하게 공간자기상관(옆 필지 ≈ 같은
피처) 한다. 무작위 CV면 train 옆집이 test에 앉아 **커닝 → 성능 5배 뻥튀기**. 이 모듈은
(1) **구역/자치구 단위 hold-out**, (2) **수용영역 폭 버퍼**로 공간누수를 막고, (3)
**튜닝용 val까지 공간분리**해 누수가 뒷문(하파 선택)으로 못 들어오게 한다. 출력엔 PR-AUC·
F1뿐 아니라 **격전지 지표(aging=0 positive recall)**·**hard/easy negative 분리**를 내장.

## 1. 이 모듈이 하는 일

모델(B0/B-2/B1/B1+/GNN)에 **정직한 train/val/test 분할을 주입**하고, 그 위에서 돈 예측을
**불균형·격전지 인지 지표**로 채점한다. 모델은 split-agnostic(baseline.md §7) — 이 모듈이
split의 유일한 책임자다. "모델을 *돌리는* 순간 split이 필요하므로 split이 먼저"(순서 근거).

## 2. 왜 공간 CV인가 (R3, 실측 근거)

수검 실측: positive 15,403필지 = **단 42구역**(구역당 중앙 333필지). 즉 유효 독립표본은
15K가 아니라 **42**. 무작위 행 분할은 같은 구역을 train·test에 쪼개 넣어 사실상 답을 보고
시험 보는 꼴. → 분할 경계는 **행이 아니라 공간 단위**(구역/자치구)여야 한다.

## 3. Fold 전략 (item 1) — Leave-One-District-Out 주력

**outer CV = 자치구 단위 leave-one-district-out, 4 fold.** test=1구, train=나머지 3구.

왜 자치구 단위인가:
- **GNN 주장과 정합**: 심장1의 명시적 강점은 *inductive*(학습 안 한 구 추론, CLAUDE §8).
  LODO가 정확히 그걸 시험 — train에 없던 구를 test.
- **구조적 무누수**: `build_graph`가 구 단위 배치라 **구간 엣지가 0**(build.py 주석). 따라서
  메시지패싱·이웃집계 누수가 LODO에선 구조적으로 불가능(피처 누수만 버퍼로 막으면 됨).
- **negative 배정 자명**: 신축파생·해제도 PNU 앞5(시군구)로 fold 배정 — 깔끔.

채점은 **두 방식 항상 병기**: ① 4개 held-out test 예측을 모두 모은 **pooled(micro) PR-AUC**(헤드라인)
② **per-fold 표(성북·동작·은평·구로 각각)** — 헤드라인과 *언제나* 같이 낸다. **구별 성능차 =
"어느 동네에서 약한가"라는 서비스 품질 정보**(어떤 구는 입지가 균질해 모델이 쉽고, 어떤 구는
난개발로 어렵다 — 이 차이를 숨기면 안 됨).

> 보조 옵션(기록만, v1 미사용): 구역중심 좌표 클러스터링으로 zone-group K-fold(5~8). 분산은
> 낮으나 구를 섞어 inductive 시험이 흐려진다 → v1은 LODO, 필요 시 v1.1.

## 4. 버퍼 (item 2, R3) — 폭 = 수용영역

train에서 **test 노드의 N홉 이내 필지를 제외**. N은 모델 수용영역에 맞춘다:
**GNN 2층 = 2홉, B1+ 2홉집계 → N = 2**(이상). 같은 버퍼를 모든 모델에 동일 적용(R9 공정).

매체가 두 가지라 둘 다 적용:
- **그래프 N홉**: test 노드에서 edge로 N홉 내 train 노드 제외. ★LODO에선 구간 엣지가 없어
  *outer엔 거의 무효*지만, **inner(구역 단위, 같은 구 안)에선 이게 진짜 누수 방어**.
- **기하 버퍼**: test 구 경계에서 거리 `buffer_m`(config, 기본 **200m**) 이내 train 필지 제외.
  ★사실확인: 우리 4구(성북·동작·은평·구로)는 **서로 인접하지 않아** LODO 구 경계는 이미
  **자연 버퍼로 충족**된다 — 200m는 안전벨트일 뿐 실제 제거는 **~0필지로 예상**. → 수검에서
  **버퍼가 실제 제거하는 필지 수를 출력**(0이면 자연버퍼 확인, **0이 아니면 그게 더 흥미로운
  발견** = 구 경계가 생각보다 가깝다). 버퍼의 실질 무대는 inner zone-holdout(같은 구 안 구역
  끼리)이고 거기엔 2홉 그래프 버퍼가 실효.

## 5. ★튜닝용 val도 공간분리 (item 3) — 누수의 뒷문 차단

early stopping·하파(3~4조합) 선택엔 eval_set이 필요하다. 이걸 **무작위로 가르면 공간누수가
본평가가 아니라 *튜닝*으로 들어온다**(모델이 공간 커닝으로 고른 하파를 갖고 test에 옴).

→ **inner CV = outer-train(3구) 안에서 다시 구역 단위 zone-holdout**:
- 3구의 구역을 train-zones / val-zones로 공간분리(구역 단위, 무작위 행 금지).
- inner에도 **그래프 2홉 버퍼** 적용(같은 구 안이라 그래프홉이 실효).
- negative는 inner도 공간근접(가장 가까운 zone-block)으로 train/val 배정.

**★inner 비용 차등**(모델별):
- **XGBoost: inner k=3.** 조합당 수십 초라 사실상 공짜. 단일 holdout으로 4조합을 고르면
  *한 구역 빼기의 운*에 하파가 좌우된다 — 유효표본 42의 세계에선 특히 위험. k=3로 평균.
- **GNN: 단일 zone-holdout.** CPU 비용 존중. early stopping이 주력 용도(하파 탐색 아님).

**★하파는 outer fold마다 따로 고르지 않는다 — 한 세트 고정.** fold별 튜닝은 그 자체가
과적합(test구별로 다른 하파 = test에 맞춤). 절차: 각 (fold f, 조합 c)의 inner 점수를 구해
**c\* = argmax_c mean_f(inner PR-AUC)** 로 *한 조합* 선정·고정. 그 c\*로 각 outer fold를
재학습하되 **early stopping의 라운드 수만 fold별 inner로 조정**(이건 모델 복잡도를 가용
데이터에 맞추는 것이라 정당, 하파 과적합 아님). 선정 근거를 LEARNING_LOG에.

## 6. 지표를 평가에 내장 (item 4)

`evaluate()`가 모델·fold마다 반환:
- **PR-AUC**(헤드라인, 불균형, R8) + **F1**(임계 0.5 및 best-F1).
- ★**격전지 recall**: aging=0 positive에 대한 recall 분리(§baseline 6.5). B-2=0% 자명, B1/GNN
  격차가 R9 승부처.
- ★**negative 분리 리포트**: hard(해제) vs easy(신축파생). **hard n=24 → "통계적 무의미" 라벨을
  출력에 박아 그 수치로 결론 못 내게**(labels §13).
- **정확도 금지**(R8). IoU·핵심부 포착률(R13)은 폴리곤 산출(infer) 후 별도 — 여기선 노드 분류 지표.

## 7. 구로 fold 취약 처리 (item 5)

구로(11530): positive 1,084필지·소수 구역 → test일 때 표본 빈약·고분산.
- per-fold 표에 **n(test positive·구역수) 병기**, 구로 행에 **wide-CI 경고**.
- 헤드라인은 **pooled(micro)** 로 — 구로 단일 숫자에 끌려가지 않게.
- 구로 fold는 정밀 점수가 아니라 **inductive 스트레스 테스트**(소수 예시 일반화)로 해석.
- bootstrap CI(구역 단위 resample)로 불확실성 정직하게 — 점추정 단정 금지(§6 예언 정직성).

## 8. 함수/파일 분해

```text
eval/spatial_cv.py
├── lodo_folds(labels) -> list[Fold]          # 자치구 단위 outer 4 fold (sigungu=pnu[:5])
├── zone_inner_split(train_labels, *, buffer)  # outer-train 안 구역 단위 train/val (item3)
├── apply_buffer(train_idx, test_idx, edge_index, geom, *, hops, buffer_m)
│                                              # 그래프 N홉 + 기하 버퍼로 train 정제(R3)
├── Fold(dataclass)  train_idx/test_idx/inner_splitter
└── evaluate(predict_fn, folds, labels) -> report   # §6 지표(PR-AUC·격전지·hard/easy)
eval/metrics.py
├── pr_auc, f1_at, best_f1
├── battleground_recall(y, p, aging)          # aging=0 positive recall
└── neg_split_report(y, p, certainty)         # hard(해제)/easy(신축) 분리, n 병기
```

`Fold`는 인덱스만 들고 모델을 모름(R9 공정: 모든 모델이 같은 fold). `evaluate`는
`predict_fn(train_labels, test_labels)->p_test` 콜백을 받아 모델 종류에 무관.

## 9. 검토했지만 버린 대안

- **무작위 k-fold**: 공간누수로 성능 뻥튀기 — R3 정면 위반. 버림.
- **무작위 inner val**: §5 그대로 — 튜닝 뒷문 누수. 버림.
- **자치구 무시 zone-group만**: 분산 ↓이나 inductive 시험 흐려짐 → v1.1 보조.
- **버퍼 생략(LODO면 구간 엣지 0이니 충분)**: 기하 인접 구 경계 피처누수 잔존 → 기하버퍼 유지.

## 10. 수검 게이트 (규칙9, 구현 후)

1. **무누수 증명**: 임의 test 노드의 모든 그래프 N홉 이웃이 train에 0개(버퍼 적용 후).
2. **fold 커버리지**: 4 fold가 모든 행을 정확히 1회씩 test(중복·누락 0).
3. **inner 공간분리**: inner val-zones ∩ train-zones 구역 교집합 0(같은 구역 안 쪼개짐 없음).
4. **버퍼 효과 측정**: 버퍼로 빠진 train 행수 보고(구로 등 경계 손실 가시화).
5. **지표 sanity**: 무작위 CV vs LODO의 PR-AUC 격차를 한 번 측정 — 무작위가 부풀면 R3 실재 입증
   (B1 한 모델로 대조, "공간누수의 크기"를 LEARNING_LOG에).
6. **격전지·hard/easy 출력 형태**가 실제로 분리되어 찍히는지 한 fold로 확인.
```

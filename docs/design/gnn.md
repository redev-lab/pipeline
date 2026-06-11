# 설계 노트 — `models/gnn/` (Phase 3-D, 심장1 본체)

> 규칙1: 코드 전에 이 노트. 승인 후 구현. CLAUDE.md §5(R7·R8·R9·R12) 우선.

## 0. 한 문단 요약

심장1의 본체: 필지 인접 그래프 위 **GraphSAGE 2층 inductive 노드 분류기**. "이 필지의
*주변 블록*이 재개발 환경인가"를 학습된 메시지 패싱으로 푼다. 베이스라인 천장 **B1+ 0.934**
(2홉 손피처)를 *구조(학습된 가중 집계)*로 넘어야 존재가치가 증명된다(R9). 라벨 희소(42구역)
대비 **자기지도 사전학습**(R7)으로 비라벨 노드에서 표현을 먼저 배운다.

## 1. 이 모듈이 하는 일

- `model.py`: GraphSAGE(2층, 얕게) — 노드 임베딩 → 양성확률. inductive(학습 안 한 구 추론).
- `pretrain.py`: 비라벨 4구 노드로 자기지도 사전학습(R7) → 가중치 초기화.
- `train.py`: 파인튜닝 루프(focal/weighted, early stopping, CPU) + spatial_cv 채점.
- `infer.py`(나중): 노드 확률 → 히트맵+후보 클러스터(R12). Phase 3 후반/Phase 4 연계.

## 2. ★키스톤 — per-t "스냅샷" 학습은 깨진다 → inductive ego-graph

가장 중요한 설계 결정. neighbor_aggregate에서 이웃피처를 *중심노드의 t로* 계산했다(R1).
GNN의 메시지 패싱도 같은 시점정합이 필요한데, **순진한 "t별 전체그래프 스냅샷 + 그 t의
라벨에 loss"는 무너진다**:

- positive는 지정연도 t(2001~2025)에, 신축 negative는 t=2026에 몰림(실측: 2026에 신축
  24,314·positive 0 / 2009에 positive 2,723·negative 0).
- → **t=2009 스냅샷엔 양성만, t=2026 스냅샷엔 음성만.** 한 스냅샷 안에 두 클래스가 없으면
  대조 신호가 0 → GNN이 분류를 못 배운다.

**해법: inductive ego-graph(노드별 as-of-t 이웃).** 라벨노드 v(t_v)마다 2홉 이웃을 t_v
스냅샷에서 GraphSAGE forward → v의 임베딩. **t를 가로질러 모은 뒤 loss를 계산**하므로 한
gradient 스텝에 양성(옛 t)·음성(2026)이 공존한다. 이게 neighbor_aggregate의 per-t 논리를
GNN으로 올린 형태 — 같은 R1 시점정합, 같은 t-그룹 처리.

### 구현(효율): t-그룹 forward → concat → 단일 loss
distinct t(~14)별로 그룹화. 각 t-그룹: 폐포(2홉) 피처 as-of t(neighbor_aggregate가 쓰는
그 계산) + 로컬 엣지 → GraphSAGE forward → 그 t의 라벨노드 logits. 14그룹 logits를 concat →
전체 라벨에 BCE/focal 한 번 → backward. (t-그룹별 backward 누적으로 CPU 메모리 절약 가능.)

### 검토했지만 버린 대안
- **전체그래프 per-t 스냅샷 + per-t loss**: 위 클래스 부재로 붕괴. 버림.
- **단일 2026 스냅샷(모든 노드 현재피처)으로 학습**: positive를 미래(2026) 상태로 봄 = R1
  누수(지정 후 동결을 원인으로). 버림.
- **NeighborLoader 미니배치 샘플링**: 표준이나 per-t 피처계산과 얽혀 복잡 → v1은 t-그룹
  full-batch, 미니배치는 메모리 터지면 v1.1.

## 3. 모델 — GraphSAGE 2층 얕게 (R7)

- **2층**: 수용영역 2홉 = B1+와 동일(R9 공정, 도달범위 맞춤). 깊이면 oversmoothing + 라벨
  부족 과적합.
- **얕은 차원**(hidden 32~64) + **dropout·weight decay 강하게**(R7: 유효표본 42).
- **mean aggregator**(GraphSAGE): 이웃 평균 — neighbor_aggregate의 mean과 같은 연산을
  *학습된 변환과 함께*. "GNN이 손집계 위에 얹는 게 정확히 학습된 가중/비선형"이 R9 질문.
- inductive: 학습 안 본 구의 노드도 이웃만 있으면 임베딩 생성(LODO test가 이걸 요구).

## 4. 사전학습 `pretrain.py` (R7) — 측정으로 가치 검증

라벨 ~42구역은 적다. **비라벨 4구 노드(수만)로 자기지도** 표현학습 후 파인튜닝.
- 방식: **DGI(Deep Graph Infomax)** 또는 마스킹 피처 복원 중 단순한 것. 현재(2026) 그래프
  상태로(라벨·t 불요 — 일반 구조표현 학습).
- ★**가치를 측정**: pretrain 有/無로 GNN을 둘 다 돌려 LODO PR-AUC·격전지 비교. 리프트가
  없으면 **정직하게 버린다**(R7은 처방이지 보장이 아님). CPU 비용 대비 가치를 수치로.

## 5. 학습 `train.py` (R8 재프레이밍·CPU)

- **loss**: BCE + `pos_weight`(균형 1.6:1이라 거의 1). focal는 *추론* 불균형 대비 옵션 —
  학습셋 균형이라 우선 BCE, focal는 ablation. 정확도 지표 금지(평가는 spatial_cv).
- **입력 정규화**: ★area_m2 outlier(555K㎡) → `log1p` + z-score. 노후도·접도는 0~1이라
  그대로. (트리인 XGBoost는 불요였지만 GNN은 필수.) 정규화 통계는 **train fold에서만** 적합
  (test 누수 차단).
- **CPU**: `torch.device` 자동감지(cuda 가정 금지). **early stopping**(inner zone-holdout val,
  spatial_cv k=1). **좁은 탐색**: hidden{32,64}×dropout{0.3,0.5} 정도 수동 4조합, 하파는
  fold횡단 한세트 고정(베이스라인과 동일 규약). **학습 1회 wall-clock 기록**(v2 GPU 판단).
- 재현성: seed 고정(Math.random류 없음).

## 6. 평가 — 같은 무대 (R9)

spatial_cv.evaluate에 `predict_fn(train_idx, test_idx)` 콜백으로 연결 — **B-2/B1/B1+와
완전히 같은 fold·버퍼·지표.** 천장 **0.934(B1+)**. 단 헤드라인 PR-AUC는 쉬운 음성에
부풀려졌으니 **진짜 승부처 = 격전지 recall(현재 B1+ 0.62)·은평 fold(최저 0.920)·hard-neg**.
- test predict: 라벨노드의 t로 ego-graph forward(학습분포 일치). 실배포는 t=2026(infer.py).

## 7. 함수/파일 분해

```text
models/gnn/
├── model.py
│   └── class RedevSAGE(nn.Module)         # 2층 SAGEConv + dropout → 1 logit
├── pretrain.py
│   ├── pretrain_dgi(graph, feats2026, cfg) -> state_dict   # 자기지도, R7
│   └── (가치 측정은 train에서 有/無 비교)
├── train.py
│   ├── _tgroup_forward(model, aug, t_groups, parcels, buildings)  # §2 per-t ego forward
│   ├── _normalizer(train_idx)             # area log1p+z, train에서만 적합
│   ├── fit_gnn(aug, edge_index, pnu_to_idx, *, params, pretrained=None)
│   └── run_gnn_cv(...)                    # 하파선택 + LODO 채점 (run_xgb_cv 대응)
└── infer.py                              # (나중) 확률 히트맵 + 후보 클러스터(R12)
```

`run_gnn_cv`는 `run_xgb_cv`와 같은 시그니처·같은 spatial_cv.evaluate를 써서 점수표에 한 줄로
얹힌다(R9 공정의 코드적 보장).

## 8. CPU 비용 관리 (현실)

t-그룹 full-batch에서 2026 폐포(~120K 노드)가 최대 비용. 2층 SAGE·작은 hidden이라 forward는
가볍지만 14그룹×에폭이 쌓인다. early stopping + 좁은탐색으로 학습횟수를 제한. **1회 wall-clock을
LEARNING_LOG에** — v2 GPU 필요성의 정량 근거. 너무 느리면(>분/에폭) 2026 그룹만 이웃
샘플링(미니배치)으로 떨어뜨리는 게 v1.1 1순위 최적화.

## 9. 수검 게이트 (규칙9, 구현 후)

1. **시점정합**: 한 옛-t positive의 ego forward가 미래(2026) 피처를 안 봄 — t-그룹 폐포
   피처가 as-of-t인지 재확인(neighbor_aggregate 수검 재사용).
2. **클래스 혼합**: 한 gradient 배치(concat logits)에 양성·음성이 실제로 공존하는지 카운트.
3. **정규화 무누수**: area 정규화 통계가 train fold에서만 적합됐는지(test 분포 미사용).
4. **inductive 작동**: LODO test 구의 노드가 학습에 없었는데 임베딩·예측이 나오는지.
5. **R9 공정**: GNN 피처 입력 = B1+와 동일 원천(self+이웃), 수용영역 동일 2홉인지.
6. **베이스라인 대조**: 점수표에 B-2/B1/B1+/GNN 한 표. 이기든 지든 격전지·은평·hard-neg를
   같이. pretrain 有/無 리프트도. (지면 "데이터 병목" 프레이밍, R9.)
```

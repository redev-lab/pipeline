# 인수인계 (다음 세션)

**Phase 2(그래프) 완료** — `build.py`(인접 그래프) + `features.py`(시점 피처) 둘 다 수검 통과.

**다음 = Phase 3 (심장1 학습) — `phase-3-models` 브랜치.**

claude_code_phases.md Phase 3 순서:
1. `models/baseline.py` — region-growing + **XGBoost+이웃집계** (대조군 먼저, R9). ★GNN과
   **동일 v1 피처**(노후도·면적·형상·호수밀도·접도) 써야 공정(R9).
2. `models/gnn/model.py` — GraphSAGE(얕게 2층), **inductive**(학습 안 한 구도 추론, R7).
3. `train.py` / `infer.py` — 학습 루프 / 후보 폴리곤.
4. `eval/spatial_cv.py` — ★구역 단위 hold-out **+ 버퍼**(R3 공간누수). 무작위 CV 금지.
   불균형 지표 PR-AUC·F1(R8, 정확도 금지). IoU + 핵심부 포착률(R13).

## ★학습 전 필수 결정 (Phase 3 시작 시)
- **완공 9구역(R2 contaminated 130필지) drop 여부** — labels §13 생존편향. drop vs v1.1
  말소대장 부활. 학습 데이터 들어가기 전에 정한다.
- **라벨 reconcile**: `reconcile_labels_to_graph`로 비노드(도로 등) 라벨 drop 후 학습.
- **피처 정규화**: area_m2 outlier(3.7M㎡)·공시지가류 (연도,구) 백분위(labels §9).

## 입력 준비됨
labels(99,964행→reconcile 96,224) · graph(노드141K/엣지665K) · features(노후도 시점·면적·
형상·호수밀도·접도). 데이터 대기 피처(용도지역·공시지가·역거리·배제)는 v1.1.

# 설계 노트 — Phase 6 파이프라인 등뼈 (infer · B0/IoU · orchestration)

> 규칙1: 코드 전에 이 노트. 승인 후 구현(모듈 순서대로). CLAUDE.md §8(plain 파이썬)·R12·R13 우선.

## 0. 한 문단 요약

조립을 사례검색·NLP보다 먼저 세운다 — 등뼈가 서야 꽂을 자리가 생긴다. ① `infer.py`가
B1로 전 노드를 스코어링해 **확률 히트맵 + 후보 클러스터**(R12)를 낸다. ② 연기했던 B0
(region-growing)를 여기서 만들어 **IoU 통제 비교**(R13)를 한다(B1 vs B0 vs 실제 지정경계).
③ `orchestration/pipeline.py`의 `run(address)`가 주소→PNU→클러스터→stage1→avm→feasibility→
eligibility를 **plain 파이썬 직선**으로 호출한다(§8). LLM ⑨는 Phase 7 — 자리만 판다.

## ★공유 전제 — 전 노드 피처 캐시 (infer·B0 공통)

학습행렬(39,741)은 라벨 노드뿐. 추론은 **전 노드(141,217)**를 스코어링해야 한다. → 현재시점
(2026) 기준 전 노드 self 피처 + per-t 아닌 **현재 이웃집계**(추론은 현재 상태)를 한 번 계산해
`_data/processed/infer_features.parquet` 캐시. (build_neighbor_features를 current_ym 전 노드로
재사용 — t 그룹이 단일(현재)이라 단순.) B0 region-growing의 노후도도 이 캐시에서.

## 1. `infer.py` — 전 노드 스코어링 → 히트맵 + 후보 클러스터 (R12)

- **production B1**: LODO가 아니라 **전 라벨로 학습한 B1**(추론용 — 검증은 Phase 3에서 끝).
- **score_all**: 전 노드 B1 확률(uncertain 포함). 출력 1차 = **확률 히트맵**(노드별 prob).
- **candidate_clusters**: 확률 ≥ `thr`(config) 노드의 **그래프 연결요소**(인접 고확률 = 한 후보).
  최소 크기 `min_nodes` 필터. → 후보 클러스터(PNU 집합) 목록. 이게 stage1·avm의 입력.
- **(보너스) cluster_polygon**: 클러스터 필지 **alpha shape + 최소면적 + 도로망 스냅핑**으로 매끈
  폴리곤. R12 — 기본은 히트맵+클러스터, 폴리곤은 후처리 보너스(품질 불완전 인정).
- ★출력 정직성: 확률은 보정 전 raw면 feasibility의 보정확률을 쓰거나 "순위" 우선(R8·R12).

## 2. B0 region-growing + IoU 평가 (R13) — 연기했던 그 무대

- **B0 region_grow**(baseline.md 설계 재활용): 노후 필지 seed(현재 노후도 ≥ seed컷) → 인접
  노후 필지로 BFS 확장(확장컷). 무학습 공간 바닥선. 임계값 config.
- **IoU 평가**: 실제 지정구역(의제처리 51개) 경계 대비 — B1 클러스터 / B0 클러스터 각각의
  - **IoU**(교집합/합집합),
  - ★**핵심부 포착률**(R13): 구역 내부(경계 버퍼 제외 코어)를 예측이 덮는 비율 — IoU 천장을
    우회하는 보조지표,
  - **사람 기준선**(R13): 완벽 모델도 IoU 100% 불가(소유·정치 경계) → 구역간 경계 변동성 또는
    "코어 포착"을 기준으로 해석.
- **통제 비교표**: B1 vs B0 vs (상한 참고) — 이기든 지든 정직하게(심장1 R9 정신).

## 3. `orchestration/pipeline.py` — `run(address)` (§8 plain 파이썬)

```text
run(address, *, property_type=None, stage=None) ->  결과 dict
  ① 주소 → PNU         : 구 파싱 + location.parse_location + jibun_index (4구). PNU 직접입력도 허용.
  ② PNU → 서브그래프    : 전 노드 캐시에서 그 PNU가 속한 후보 클러스터 조회.
  ③ B1 점수            : 그 노드/클러스터 확률.
  ④ stage1(요건)       : 클러스터 → 재개발/모아타운/해당없음 + 근거.
  ⑤ avm(시세 맥락)      : PNU 반경집계 대지지분 평당가 + 비교신축(빼기 금지).
  ⑥ feasibility(환경점수): 보정확률 → "재개발 환경 점수 상위 X%" + 리스크 자리표시.
  ⑦ eligibility(진입)   : 토허(물건유형)·잔여기간(단계). property_type·stage 입력.
  ⑧ 종합 결과 dict      : 진단/예언 분리 + 전체 caveats.
  ⑨ LLM 설명·종합       : ★Phase 7 — run()에 자리만(placeholder), v1은 구조화 dict.
```

- ★**plain 파이썬 직선 DAG**(§8 — LangGraph 아님). 동적 지점은 둘: **B1 저신뢰 → 폴백 if**
  하나(확률 낮으면 "후보 아님" 조기 반환), **⑨ 근거부족 재호출 루프 자리**(Phase 7).
- ★**각 단계 try/except**: 부분 실패(예: 거래 0건 → avm 결측)도 전체가 죽지 않게 — 단계별
  status('ok'/'skipped'/'error') + 사유. 서비스 견고성.
- 출력은 §6 진단(요건·토허·시세)과 예언(환경점수·잔여기간)을 분리, 모든 caveat 동봉(R15).

## 4. 함수/파일 분해

```text
models/infer.py
├── build_all_node_features(parcels, buildings, graph, pnu_to_idx) -> df(cache)
├── train_production_b1(aug) -> model           # 전 라벨 학습(추론용)
├── score_all(model, all_feats) -> Series        # 전 노드 확률
├── candidate_clusters(scores, edge_index, *, thr, min_nodes) -> list[set[pnu]]
└── cluster_polygon(cluster, parcels) -> geom    # (보너스) alpha shape+최소면적+스냅
models/baseline.py  (추가)
└── region_grow(scores_or_aging, edge_index, *, seed_cut, grow_cut) -> list[set]  # B0
eval/iou.py
├── zone_iou(pred_cluster, zone_geom) ; core_capture(pred, zone, *, buffer)
└── compare_clusters(b1_clusters, b0_clusters, zones) -> 비교표
orchestration/pipeline.py
└── run(address, *, property_type, stage) -> dict   # ①~⑨ 직선 + try/except + 폴백 if
```

## 5. ★수검 (규칙9, 구현 후)

1. **엔드투엔드 1회**: `run("실제 4구 주소")` 통과 — 단계별 status·출력 sanity(요건·시세·점수·토허).
2. **IoU 비교표**: B1 vs B0 vs 지정구역 — IoU·핵심부 포착률. 어느 쪽이든 정직 보고(B0가 단순해도
   IoU 천장·코어 포착 해석). 무작위 아닌 패턴이면 발견.
3. **폴리곤 육안**: 히트맵·후보 클러스터 시각 1장(품질 불완전 인정, R12).
4. **★CPU 추론 시간(주소당) 측정** — 서비스 체감의 첫 실측(전 노드 캐시 후 조회는 빨라야).
5. **부분 실패 견고성**: 거래 0건 PNU·미상 단계 등에서 run()이 죽지 않고 status로 보고하는지.

## 6. 검토했지만 버린 대안

- **LangGraph 오케스트레이션**: 직선 DAG라 과한 도구·블랙박스(§8 위반) → plain 파이썬, 복잡해지면 이전.
- **주소 지오코딩 외부 API**: v1은 4구 (구+동+지번) 파싱+jibun_index로 충분 → 외부 의존 회피(v1.1 확장).
- **매끈 폴리곤 우선**: 노드 분류 이어붙이면 구멍·삐죽(R12) → 기본은 히트맵+클러스터, 폴리곤은 보너스.
- **B1 확률 그대로 표시**: 보정 안 됨 → feasibility 보정확률/순위 우선(R8).

# 인수인계 (다음 세션)

## ★v1.1 피처 사이클 완료 (2026-06-12) — 병목 해부
- ingest 3종(공시지가 as-of-t·용도지역·역세권 as-of-t) + features 5→10 + 같은 LODO 재경기.
- ★결과: B1+ 격전지 0.622→0.683·AVM 격차 절반(피처 병목 완화) / GNN 0.929→0.913 악화·infer 81%
  불변(라벨 병목). leakage_ablation: 용도지역 2026스냅샷 누수 net-negative(§13 확인).
- ★v2 우선순위 측정 확정: 심장1/infer=전역 라벨 확장(R7-①)+PU 학습, 심장2=가치피처 유지+as-of-t 용도지역.
- ★**production 심장1 = B1+ −용도지역** 확정(`production_feature_set`): 용도지역 2026스냅샷 누수가
  LODO net-negative(B1+ 0.937→0.940·격전지 0.683→0.712). infer·feasibility 재배선. AVM·stage1은
  현재시점이라 용도지역 유지(과제별 분리).
- ★**백로그 v1.2 = PU 학습**(uncertain 5.6만 활용, 과대예측 81% 직접 겨냥, ★신규 데이터 불요) —
  전역 확장(v2)보다 싼 선행 레버. 다음 사이클 1순위 후보.
- 다음: v1.2 PU 학습 / Phase 8 데모 3종(마포·강남 ingest) / v2 전역 라벨 확장 중 택1.


## ★현재 상태 (2026-06-12): Phase 0~7 완료 — v1 파이프라인 등뼈+LLM 완성
- **Phase 7 = LLM 종합 완료** (`phase-7-llm`): ① retrieval(numpy 코사인, t제외)·② nlp/layer3
  (Gemini 사회신호, 무신호=정상)·③ llm/report(⑨ 5종 판단, ★환각 0 검증)·run(with_report=True) 연결.
  ★숫자 포맷 LLM에 안 맡기고 사전포맷 verbatim → verify_numbers 정확대조. 환각 3주소 0·CPU 13s/주소.
  client=Gemini 한 겹(백오프 재시도+폴백). 모델 config 핀(gemini-2.5-flash, 2.0-flash 무료폐지).
- **★주소 한 줄 → 5종 판단 한국어 리포트**가 돈다(82 테스트 green).
- **다음 = Phase 8**: 데모 3종(광흥창·역삼)용 마포(11440)·강남(11680) 추론용 구 ingest +
  stage1 역삼 negative 데모. v1.1: PU학습·가치/입지 피처(공시지가 26년·역사·용도지역 입고됨)·
  hard-neg k-hop·뉴스 크롤·pgvector(전역).

---

## (이전) Phase 0~6 완료
- **Phase 6 = 파이프라인 등뼈 완료** (`phase-6-infer`): infer(전 노드 히트맵+후보 클러스터)·
  B0/IoU(R13)·`orchestration/pipeline.py run(address)` 7단계 직선. CPU 68ms/주소. 전체 74 테스트 green.
- **데이터병목 3호 실증**: infer 81% 과대예측(R4 PU갭). IoU B1넓은 0.294>B0 0.129>타이트 0.019.
  릴레이 IoU(B1→stage1)=0.294 동일(stage1도 노후도 축, 무이득). 심장1·심장2·infer 모두 같은 결.
- **★Phase 번호 = 기획서와 다름**(순서 변경): 조립(infer+orchestration)을 사례검색·NLP보다 먼저
  세웠다(등뼈 우선). → **사례검색(retrieval) + Layer3(NLP 사회신호) + LLM ⑨(종합·설명) = Phase 7**로
  밀림. run()에 ⑨ LLM 자리는 placeholder로 파둠.
- **다음 = Phase 7**: retrieval(pgvector 사례검색) + nlp/layer3(추론시점 사회신호) + LLM ⑨(run에 연결).

---

**Phase 3 완료** — `phase-3-models` 브랜치. 심장1 노드 분류 + R9 통제비교 마무리.

## ★Phase 3 최종 판정 (R9)
- 통합 점수표(LODO pooled): B-2 0.640 / B1 0.932 / B1+ **0.934** / GNN 0.929 / GNN+pretrain 0.928.
- **GNN은 천장(B1+) 못 넘고 미세 열세** — zone-block bootstrap Δ +0.006, CI [+0.002,+0.012](0 불포함).
  사전학습 리프트 ≈0. → **데이터 병목(R9), v1 추론모델 = B1.** GNN 보존(README v2 조건).

## Phase 4 완료 (Stage1 룰셋, `rules/stage1.py`)
- 클러스터(PNU집합)→요건 점수+경로(재개발/모아타운/해당없음). 결정론·config·caveat 동봉.
- 수검(지정구역 51 역검증): 본진 UQ1221 47개 재개발 57%. 해당없음 14=완공(노후0/NaN, 정상),
  모아타운 6=노후0.6~0.96인데 접도율(폭무관)·호밀(동단위) 한계로 광역미달(측정으로 caveat 확인).
- 다음 = **Phase 5 (심장2 AVM, `models/avm.py`)** 또는 **infer/orchestration**(B1 점수→클러스터
  →stage1 연결). claude_code_phases.md 참조.
- ★메모: Phase 8 데모 3종(광흥창·역삼)이 마포(11440)·강남(11680) 추론용 구 ingest 요구 →
  그 시점에 stage1 역삼 negative 데모·infer 클러스터 조립도 함께.

## 완료 (커밋)

## 완료 (커밋)
- **3-A** `load_training_matrix`(39,741행 캐시) + per-t `neighbor_aggregate`(R1 누수차단 실증). [5017a5b]
- **3-B** `eval/spatial_cv.py`(LODO 4구+버퍼+구역 inner) + `metrics.py`(PR-AUC·격전지·hard/easy). [50bb04a]
- **3-C 일부** 베이스라인 노드 사다리: B-2(aging바닥선)·B1(1홉)·B1+(2홉). 첫 점수표:
  - pooled PR-AUC: B-2 0.640 / B1 0.932 / B1+ 0.934. 격전지 recall(aging=0): 0%/63%/62%.
  - ★발견: ①이웃집계가 동어반복 압도 ②**B1+≈B1**(2홉 손피처 무의미→GNN은 구조로 천장 넘어야)
    ③헤드라인 0.93 부풀려짐(쉬운 신축 99%), 진짜난도=hard해제 FPR 0.5~0.58(R18·labels §13).

## ★다음 = Phase 3-D GNN (심장1 본체)
순서: `models/gnn/pretrain.py`(자기지도, R7) → `models/gnn/model.py`(GraphSAGE 2층 inductive)
→ `models/gnn/train.py`(focal/weighted, R8) → spatial_cv로 채점.
- **넘어야 할 천장: 0.934**(B1+). 단 헤드라인 PR-AUC 말고 **격전지 recall·은평 fold·hard-neg**가
  진짜 승부처(헤드라인은 쉬운 음성에 부풀려짐).
- **★GNN 설계 결정(노트 먼저, 규칙1)**: per-t 피처 스냅샷(메시지패싱도 as-of-t, neighbor_aggregate
  와 동일 논리) · pretrain은 비라벨 4구 노드 현재상태(2026) 자기지도 · area outlier 입력정규화 ·
  CPU(device 자동감지·early stopping·좁은탐색·wall-clock 기록).

## ★연기(skip 아님)
- **B0 region-growing → infer 단계로.** 무대가 노드 PR-AUC가 아니라 폴리곤 IoU(R13 좌표축).
  GNN 후보 클러스터→폴리곤(R12) 평가 때 같은 무대에 세운다. 전노드 노후도 캐시도 그때 추론용
  으로 필요 → 함께 만든다.

## 입력 준비됨 (캐시 `_data/processed/`, gitignore)
- `train_matrix.parquet`(self피처+neg_reason+좌표) · `train_matrix_nb.parquet`(+nb1·nb2 31컬럼)
  · `graph_edge_index.npy`·`graph_pnu_idx.parquet`(전역 141K 노드). 재빌드 `force_rebuild=True`.

## v1.1 백로그
- ★**v1.1 데이터 3종 입고·검수완료(2026-06-12), 투입 대기** — `_data/raw/추가데이터/`,
  DATA_SOURCES.md 참조. 공시지가(AL_D151, 4구 17만필지)·역사마스터(784역 WGS84)·용도지역
  (AL_D124, 4구 1097폴리곤 EPSG:5186 세분류). 이게 두 심장 데이터병목(R9)을 푸는 v2 가치/입지
  피처. ⚠️**공시지가 2026 단일연도 → 시점분리 학습엔 과거연도 추가 입수 필요**(현재가치만이면 OK).
- hard-neg(해제) k-hop 확장 **1순위**(labels §13: 24노드뿐→"요건됐는데 무산" 학습불가).
- uncertain PU 정식처리 · 역사마스터 5186 reproject · forest size-hub guard
  · buildings 4구 선필터(조립 100s↓) · 잔여 보류 3구역 회수 · 상승여력 수식(용적률·비례율·분양가).

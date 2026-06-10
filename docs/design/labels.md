# 설계 노트 — `data/labels.py`: (필지, 시점) 시점 라벨 테이블 (Phase 1 키스톤)

> 규칙1(설계 먼저). 승인 전 구현 코드 없음.
> 이 모듈은 Phase 1의 **핵심 산출물이자 최고 레버리지** — R1·R4·R5·R11이
> 여기서 한 번에 풀린다(§5 키스톤). ingest는 이 테이블의 *재료 공급책*이다.

---

## 0. 한 문단 요약

라벨을 `(필지)` 한 줄이 아니라 **`(필지 PNU, 라벨 시점 t, 라벨, 확신도)`**
한 줄로 만든다. 각 학습 예시 = "그 필지를 **시점 t의 상태로 본** 한 장면".
이 한 줄짜리 설계 변경이 노후도 시점 누수(R1)·우측절단 네거티브(R4)·해제→재지정
충돌(R5)·분합필(R11)을 *동시에* 푼다. ★그러나 이건 여전히 **정적 노드 분류**다 —
동적/시계열 GNN이 아니다. 시간은 모델이 아니라 *데이터 준비*에만 산다.

---

## 1. 이 모듈이 하는 일 (①)

ingest가 적재한 원천들(건물·정비구역경계+고시일·신통·해제·배제레이어)을 받아,
**학습 가능한 라벨 테이블**로 조립한다. 출력 한 행 = "PNU p를 시점 t에 봤을 때
이건 positive/negative였고, 그 네거티브가 얼마나 확실한가". 부산물로 R11 매핑
실패·R2 오염 의심 목록과 결측률 리포트를 같이 낸다(정직성).

이 테이블이 **데이터 파이프라인의 첫 산출물**이다. Phase 2(graph/features)는 각
행의 t에 맞춰 피처를 계산하고, Phase 3(GNN)은 이 테이블로 학습한다.

---

## 2. 왜 `(필지)`가 아니라 `(필지, 시점)`인가 (②) — 이 노트의 심장

### 2-1. naive `(필지, 라벨)` 테이블은 네 군데서 동시에 무너진다

한 PNU당 한 줄, label=1(정비구역)/0(아님). 단순해 보이지만:

- **R1 (노후도 시점 누수).** 2021 지정 구역을 label=1로 두고 피처(노후도)를
  *2026* 데이터로 계산하면? 지정되면 신축이 금지돼 노후도가 그 시점에 **동결**
  되거나, 철거·신축이 끝나 오히려 **낮게** 찍힌다. 모델은 "지정의 *결과*"를
  "지정의 *원인*"으로 거꾸로 학습한다. 그런데 `(필지)` 테이블엔 **피처를 어느
  시점으로 계산할지 고정할 닻(t)이 없다.**
- **R5 (해제→재지정 충돌).** 장위뉴타운 일부는 2014 해제, 2022 신통 재지정.
  `(필지)` 테이블은 이 한 PNU에 label을 **하나만** 줄 수 있다 → 0이든 1이든 거짓.
- **R4 (우측절단 네거티브).** "지정 안 됨"이 "확실히 안 될 땅"인지 "**아직** 안
  된 땅"인지 구분이 없다. 미확정을 전부 negative로 밀면 모델이 "후보지를
  네거티브"로 학습(PU 문제).
- **R11 (분합필).** 2021 PNU가 2026엔 쪼개지거나(분필) 합쳐져(합필) 사라진다.
  현재 PNU 한 줄에 과거 라벨을 못 붙인다.

### 2-2. `(필지, 시점 t)`가 넷을 동시에 푸는 메커니즘

각 행에 **t를 명시**하는 순간:

| 리스크 | (필지,시점)이 푸는 방식 |
|---|---|
| **R1** | t가 피처 계산의 닻이 된다. 그 행의 노후도 = "t에 존재하던 건물(사용승인일 ≤ t)만으로 t 시점에 계산". 지정의 결과가 원인으로 새는 경로가 끊긴다. |
| **R5** | 같은 PNU가 **자동으로 두 행**이 된다: (p, 2014, neg) + (p, 2022, pos). 모순이 아니라 서로 다른 두 장면. 특수 케이스 처리 불필요 — 구조가 알아서 푼다. |
| **R4** | 각 행에 `확신도(certainty)` 컬럼: positive / reliable_negative / uncertain. PU 학습기가 이 컬럼을 읽어 uncertain을 가중치 차등 또는 제외. |
| **R11** | 각 행의 PNU를 *피처 계산에 쓸 현재 필지*로 해석(resolve)해야 하므로, 매핑 실패가 자연히 이 단계에서 잡힌다 → drop + 결측률 기록. |

추가로 **R2(완공·철거 오염)**: positive인데 *t 시점 노후도조차* 50% 미만이면
(config `label_hygiene.min_old_ratio_for_positive`), 이미 신축됐거나 데이터가
오염된 것 → `contaminated=True` 플래그(Phase 1은 목록만, 삭제는 검토 후).

> **구체 예시 (장위, 숫자로):**
> - PNU `1129010800...`, t=2014: 해제됨 → `(p, 2014, label=0, reliable_negative,
>   reason=cancelled)`. 피처는 2014 상태(노후도 ~72%)로 계산.
> - 같은 PNU, t=2022: 신통 선정 → `(p, 2022, label=1, positive)`. 피처는 2022
>   상태(노후도 ~78%)로 계산.
> 모델이 보는 것: "2014 동네 모습 → 무산된 사례 / 2022 동네 모습 → 지정". 둘 다
> 참이고 둘 다 학습에 유효하다.

### 2-3. ★여전히 정적 노드 분류다 (가장 중요한 오해 차단)

"시점이 들어갔으니 동적/시계열 GNN인가?" **아니다.** 이걸 못 박는 게 이 설계의
핵심 우아함이다.

- GNN은 여전히 `(그래프 G, 노드 피처 X) → 노드 라벨`인 **정적** 분류기다.
  recurrence도, 시간 인코딩도, temporal message passing도 없다(GraphSAGE 2층).
- t가 하는 일은 단 하나: **그 행의 피처 X를 어느 스냅샷으로 계산할지 고르는 것.**
  (p, 2014) 행과 (p, 2022) 행은 *서로 다른 정적 피처 벡터*를 가진 **독립된 두
  예시**일 뿐이다.
- 즉 **시간은 모델 안이 아니라 `labels.py`+`features.py`(데이터 준비)에만 산다.**
  덕분에 우리는 동적 GNN의 복잡도 없이 시점 정합성을 얻는다.

> 한 줄 정리: **"시점 라벨 = 시계열 모델"이 아니라 "시점 라벨 = 시점별로 올바르게
> 스냅샷된 정적 예시들의 모음"**.

---

## 3. 라벨 테이블 스키마

한 행 = 한 학습 예시. (v1: pandas DataFrame → parquet 저장)

| 컬럼 | 타입 | 의미 |
|---|---|---|
| `pnu` | str(19) | 필지 고유키(현재 필지로 resolve된, R11 통과분). |
| `t` | int(연도) | 라벨 시점. 고시/선정/해제 *연도*. (원본 날짜는 `t_date`에 보존) |
| `t_date` | date\|null | 감사용 원본 일자(결정고시일 등). 연도 granularity가 노후도 계산엔 충분. |
| `label` | int(0/1) | 1=재개발 진행/지정, 0=네거티브. |
| `certainty` | enum | `positive` / `reliable_negative` / `uncertain` (R4 키). |
| `source` | enum | `의제처리`(OA-20957) / `신통` / `해제` / `배제레이어` / `신축파생` / `노후미지정`. |
| `neg_reason` | enum\|null | 네거티브 사유: `cancelled` / `excluded_layer` / `new_construction` / `not_yet`. |
| `zone_id` | str\|null | 결정고시관리코드/구역 식별자. ★R3 공간 CV를 *구역 단위로* 묶는 키. |
| `contaminated` | bool | R2 오염 의심(positive인데 t-노후도<컷). 학습 전 drop 후보. |

부산물(테이블 아님, 사이드카 리포트):
- `dropped`: R11 매핑 실패 PNU + 사유 + 소스별 결측률.
- `stats`: 소스별 행 수, label 분포, certainty 분포, contaminated 수.

---

## 4. 출처 → 행 생성 규칙

- **positive (label=1, certainty=positive)**
  - `의제처리`: OA-20957 폴리곤 ∩ 현재 필지 → 그 안의 PNU들. t = OA-20283
    결정고시일 연도(`결정고시관리코드`로 조인). zone_id = 결정고시관리코드.
  - `신통`: 신통 선정구역 ∩ 필지. t = 선정연도. zone_id = 구역명/코드.
- **reliable_negative (label=0, certainty=reliable_negative)** — R4의 "확실한 쪽"
  - `해제`(reason=cancelled): 해제구역(~389) ∩ 필지. t = 해제연도. **요건은
    됐으나 무산된 최강 네거티브.** (R5에 의해 이후 t에 positive로 다시 등장 가능.)
  - `배제레이어`(reason=excluded_layer): 그린벨트·문화재·군사 등 구조적으로 정비
    불가. t = 해당 시점(또는 상시). 
  - `신축파생`(reason=new_construction): **building_gis에서 파생** — 노후도가
    매우 낮고(신축 밀집) 미지정인 구역. "곧 재개발 안 됨"의 값싼 확실 네거티브.
- **uncertain (label=0, certainty=uncertain)** — R4의 "미확정"
  - `노후미지정`(reason=not_yet): 노후하지만 지정·해제·배제 어디에도 안 걸린 땅.
    "아직 안 된 것"일 수 있다 → PU 학습기가 가중치↓ 또는 제외.

**충돌 해소:** 같은 `(pnu, t)`에 두 소스가 들어오면 우선순위(positive > reliable_
negative > uncertain)로 1행. 단 **다른 t면 별개 행으로 둔다**(R5는 충돌이 아니다).

---

## 5. 데이터 흐름 & ingest 의존성

```
building_gis ─┬─→ 노후도 as-of-t (R1 helper) ─┬─→ R2 오염 플래그
              └─→ 신축파생 네거티브 ───────────┘
zone_boundary(OA-20957+20283) ─→ positive(의제처리) + t + zone_id
shintong ─────────────────────→ positive(신통) + t
cancelled ────────────────────→ reliable_negative(cancelled) + t   ← R5 자동 두 행
배제레이어 ────────────────────→ reliable_negative(excluded_layer)
                         │
                         ▼
                  labels.build_label_table()
                         │
            ┌────────────┼─────────────┐
            ▼            ▼              ▼
      라벨 테이블    dropped(R11)    stats 리포트
```

★ **labels는 ingest #1~4 + 배제레이어에만 의존.** `transactions.py`(AVM)·
`regulation.py`(eligibility)는 라벨과 무관 → Phase 5 재료다. 그래서 **구현
순서**: building_gis → zone_boundary → shintong → cancelled → **labels.py**(키스톤
먼저 닫는다) → 그 다음 transactions·regulation.

---

## 6. 함수/파일 분해 (④)

`data/labels.py`
| 함수 | 입력 → 출력 | 역할 |
|---|---|---|
| `build_label_table(sources, cfg) -> (df, report)` | ingest 산출물들 → 라벨 테이블 + 리포트 | 오케스트레이터. 아래를 순서대로 호출·병합·충돌해소. |
| `_positives_from_zones(zones)` | 폴리곤+고시일 → rows | 의제처리 positive. |
| `_positives_from_shintong(sht)` | 신통 → rows | 신통 positive. |
| `_negatives_from_cancelled(canc)` | 해제 → rows | reliable_neg(cancelled). |
| `_negatives_from_excluded(layers)` | 배제레이어 → rows | reliable_neg(excluded). |
| `_negatives_from_newbuild(bldg)` | building_gis → rows | 신축파생 reliable_neg. |
| `_uncertain_old_undesignated(bldg, assigned)` | 노후·미지정 → rows | R4 uncertain. |
| `_resolve_pnu_over_time(rows)` | rows → (rows', dropped) | R11 매핑·drop. |
| `_flag_contamination(rows, bldg, cfg)` | rows → rows | R2 오염 플래그(t-노후도<컷). |
| `_resolve_conflicts(rows)` | rows → rows | 같은 (pnu,t) 우선순위 병합. |

`data/aging.py` (★신설 제안 — 공유 헬퍼)
| 함수 | 역할 |
|---|---|
| `old_ratio_as_of(pnu, t, buildings, cfg) -> float` | "t 시점, 사용승인일≤t 건물만으로 계산한 노후도". **R1의 심장.** labels.py(R2)와 graph/features.py(Phase 2)가 **둘 다** 쓴다 → 중복 방지 위해 별도 모듈. |

> 왜 `aging.py`를 따로 빼나: 노후도 as-of-t는 R2(라벨 오염)와 노드 피처(Phase 2)
> 양쪽에서 필요하다. labels 안에 묻으면 features가 import하기 어색하고, 정의가
> 갈라질 위험. 시점 정합성의 단일 정의처(single source of truth)로 둔다.

---

## 7. 검토했지만 버린 대안 (③)

| 대안 | 왜 버렸나 |
|---|---|
| `(필지)` 정적 라벨 + 나중에 시점 보정 | R1/R5가 구조적으로 안 풀림. t 닻이 없으면 피처 시점을 못 고정. |
| 동적/시계열 GNN(TGN 등) | 과한 도구. 우리에게 필요한 건 "시점별 올바른 스냅샷"이지 시간 진화 모델링이 아니다(§2-3). 복잡도·라벨부족(R7) 악화. |
| 네거티브를 전부 0으로(단일 클래스 가정) | R4 PU 문제 무시 → 후보지를 네거티브로 학습. certainty 분리가 정답. |
| 과거 t별로 인접 그래프도 완전 재구축 | 분합필까지 시점 복원하면 비용 폭발. v1은 현재 그래프에 과거 PNU를 resolve(R11 drop). →§9 미해결로 명시. |
| labels와 features에서 노후도 각자 계산 | 정의 분기 위험. `aging.py` 단일 정의처로. |

---

## 8. 시점 정합성 닻 — 노후도 as-of-t (R1)

`old_ratio_as_of(pnu, t, ...)`의 개념: 필지 위(또는 반경)의 건물 중 **사용승인일이
t 이전인 것만** 추려, config의 경과연수 기준(`building_aging.rc_years` 등)을
**t 시점 기준으로** 적용해 노후 비율을 낸다. 핵심은 "t 이후에 생긴 건물·정보는
존재하지 않는 것처럼" 본다는 것. 이게 R1을 막는 단 하나의 규율이다.
(R2: 이렇게 계산해도 positive가 컷 미만이면 철거·완공 오염으로 본다.)

---

## 9. 확정된 결정 (작업자 승인 2026-06-10)

1. **데이터 제공 = 작업자.** 합성 픽스처가 아니라 실데이터로 ingest를 짠다.
   → ingest 리더(building_gis 등)는 원천 파일의 실제 컬럼·인코딩·SHP 구조를
   봐야 구현 가능 → **데이터 입수가 ingest 구현의 선행조건.** 단, 포맷에
   *독립적인* 두 모듈(`aging.py`, `labels.py` 조립 로직)은 ingest **출력 스키마**
   에만 의존하므로 데이터 입수 전에도 먼저 짤 수 있다.
2. **`data/aging.py` 신설 확정** — 노후도 as-of-t 단일 정의처(§6).
3. **t granularity = 연도** (원본 날짜 `t_date` 보존).
4. **v1 인접 그래프 = "현재 필지 그래프" 하나**, 과거 PNU는 resolve(R11 drop).
   시점별 그래프 복원은 v2.
5. **이번 세션 범위 = 키스톤까지** — building_gis·zone_boundary·shintong·
   cancelled(+배제레이어) → labels.py. transactions·regulation은 Phase 5로.

### 데이터 핸드오프 — 작업자가 줄 것 (라벨 키스톤에 필요한 최소 집합)
| # | 소스 | 필요 형식/키 | 무엇에 쓰나 |
|---|---|---|---|
| 1 | GIS건물통합정보 | PNU + 사용승인일(A13) + geometry | 노후도 as-of-t(R1), 신축파생 neg, R2 |
| 2 | OA-20957 의제처리구역 SHP + OA-20283 결정고시(`결정고시관리코드` 조인) | 폴리곤 + 결정고시일 | positive(의제처리) + t |
| 3 | 신통 선정구역 (정보몽땅) | 구역 경계/주소 + 선정연도 | positive(신통) + t |
| 4 | 해제구역(~389) | 구역 경계/주소 + 해제연도 | reliable_negative(cancelled) |
| 5 | 배제레이어 (그린벨트·문화재·군사) | 폴리곤 | reliable_negative(excluded) |

> 핸드오프 방법: 파일을 한 경로(예: `pipeline/_data/raw/`, .gitignore 처리)에
> 두고 알려주거나, 위치를 알려주면 된다. 포맷을 보고 ingest 리더를 맞춘다.

---

*승인되면: aging.py(노후도 닻) → 라벨 소스 ingest(한 소스씩, PNU매핑→적재→EDA) →
labels.py(조립) 순으로, 파일마다 "어디에 앉는지" 한 문단 붙여 구현한다.*

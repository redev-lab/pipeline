# 설계 노트 — `data/labels.py`: (필지, 시점) 시점 라벨 테이블 (Phase 1 키스톤)

> 규칙1(설계 먼저). 승인 전 구현 코드 없음.
> 이 모듈은 Phase 1의 **핵심 산출물이자 최고 레버리지** — R1·R4·R5·R11이
> 여기서 한 번에 풀린다(§5 키스톤). ingest는 이 테이블의 *재료 공급책*이다.
>
> **갱신 이력**: 2026-06-10 최초 / 2026-06-11 positive 소스 1→3, WHERE·WHEN 분리,
> t 소스맵, 신축파생 정의·config, 시점 불일치 교란 완화, 정량 균형 분석 반영.

---

## 0. 한 문단 요약

라벨을 `(필지)` 한 줄이 아니라 **`(필지 PNU, 라벨 시점 t, 라벨, 확신도)`**
한 줄로 만든다. 각 학습 예시 = "그 필지를 **시점 t의 상태로 본** 한 장면".
이 한 줄짜리 설계 변경이 노후도 시점 누수(R1)·우측절단 네거티브(R4)·해제→재지정
충돌(R5)·분합필(R11)을 *동시에* 푼다. ★그러나 이건 여전히 **정적 노드 분류**다 —
동적/시계열 GNN이 아니다. 시간은 모델이 아니라 *데이터 준비*에만 산다.

★**WHERE/WHEN 분리**(2026-06-11): "어느 필지(WHERE)"와 "언제(WHEN, t)"를 분리한다.
의제처리 폴리곤은 경계(WHERE)만 주고, t(WHEN)는 신통·정비사업이 준다.

---

## 1. 이 모듈이 하는 일 (①)

ingest가 적재한 원천들(건물·의제처리경계·신통·정비사업·해제)을 받아,
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
| `t` | int(연도) | 라벨 시점. (원본 날짜는 `t_date`에 보존) |
| `t_date` | date\|null | 감사용 원본 일자. 연도 granularity가 노후도 계산엔 충분. |
| `t_source` | enum | t 출처: `oa_first_decision`(1차)/`promotion_parent`/`shintong_select`/`first_designation_gosi`/`cancel_date`/`derived_current`/`gosi_list`/`public_redev`. ★틀린 t 섞임 추적·차단. |
| `t_alt` | date\|null | **보조 t**(불일치 케이스의 다른 소스 날짜 보존). 감사·재판정용 — 채택값=`t`, 후보=`t_alt`. |
| `label` | int(0/1) | 1=재개발 진행/지정, 0=네거티브. |
| `certainty` | enum | `positive` / `reliable_negative` / `uncertain` (R4 키). |
| `source` | enum | `신통`/`정비사업`/`의제처리경계`/`해제`/`신축파생`/`노후미지정`. |
| `neg_reason` | enum\|null | `cancelled`/`new_construction`/`not_yet`. (배제레이어는 v1 보류) |
| `zone_id` | str\|null | 구역 식별자(NTFC_SN 등). ★R3 공간 CV를 *구역 단위로* 묶는 키. |
| `zone_type` | enum\|null | `UQ1221`(주택정비형)/`UQ1222`(도시정비형). **학습엔 둘 다 positive**, 분석·Phase4 경로검증에서 구분용. |
| `contaminated` | bool | R2 오염 의심(positive인데 t-노후도<컷). 학습 전 drop 후보. |

부산물(테이블 아님, 사이드카 리포트):
- `dropped`: R11 매핑 실패·t-less 보류 PNU + 사유 + 소스별 결측률.
- `stats`: 소스별 행 수, label·certainty 분포, contaminated 수.

---

## 3.5 ★t 소스맵 (어느 라벨이 어느 t를 쓰는가)

★**2026-06-11 개정 — OA-20283(결정고시)이 positive의 1차 t 소스.** 의제처리 폴리곤의
NTFC_SN → `normalize_zone_name` 토큰 → 동명→구 가드 → OA `MIN(결정, 재건축·무관 배제)`
= **최초 정비구역 지정일**. 73 폴리곤 중 64개(88%)가 이걸로 t 확보(보류 66%→12%).
**신통선정일·정비사업고시일은 1차에서 교차검증·보완으로 강등**(§4).

| source(positive) | **t =** | t_source | 비고 |
|---|---|---|---|
| 의제처리(OA 회수) | **최초 결정고시일** | `oa_first_decision` | 1차. earliest-genuine(재건축 배제) |
| 의제처리(촉진 sub) | **부모 촉진지구 결정일** | `promotion_parent` | 근사치(sub 고유일 아님) |
| 의제처리(keyless 단일지정) | **NTFC 고시일** | `ntfc_direct` | ★조건: NTFC 제목 "지정"&¬"변경" + OA 더이른결정 없음. t<2018이면 플래그·육안. "변경"이면 탈락→보류(추정 금지) |
| 공공재개발 | 선정일 | `public_redev` | 1-b. t_kind=designation은 OA 교차검증(OA 우선, CSV→t_alt) |
| 신통 | 신통선정일 | `shintong_select` | OA 없을 때/교차검증 |
| 정비사업 | 정비사업 고시일 | `first_designation_gosi` | 교차검증·보완 |
| (향후) 고시목록 | 고시일 | `gosi_list` | 토지이음(1-a) — UQ 밖 standalone positive용 |

| source(negative) | **t =** | t_source |
|---|---|---|
| 해제 | 해제일자 | `cancel_date` |
| 신축파생 / 노후미지정 | 현재(최신 데이터 연도) | `derived_current` |

> 불일치(OA vs 정비사업) 시 **earliest-genuine 채택**(t), 다른 날짜는 `t_alt`에 보존.
> 의심 회수는 토큰이 제목에 *독립 구역명*으로 등장하는지 1건 검증 후 통과(흑석1=1986
> 검증 완료: '흑석제1주택개량재개발구역사업계획결정' 1986 — 진짜).

---

## 4. 출처 → 행 생성 규칙 (WHERE/WHEN 분리)

### 4-1. positive (label=1, certainty=positive) — WHERE=의제처리 폴리곤, WHEN=OA 1차
★**2026-06-11 개정: t 1차 소스 = OA-20283(결정고시) 이력 마이닝.** 의제처리 폴리곤이
WHERE(경계)와 NTFC_SN을 동시에 준다 — NTFC_SN→토큰→OA `MIN(결정)` = **최초 지정일**.
- **의제처리 폴리곤 + OA 최초결정** → positive. WHERE=폴리곤 ∩ parcels, t=OA(`oa_first_decision`).
- **촉진 sub-구역**: OA 토큰 직매칭 실패 시 **부모 촉진지구 최초결정**을 t로(`promotion_parent`, 근사).
- **신통선정일·정비사업 고시일 = 교차검증·보완**(강등). OA 없을 때 보완, 있을 때 일치검증.
  불일치는 earliest-genuine 채택 + `t_alt` 보존(§3.5).
- **t 안 붙는 폴리곤 = 보류(제외)** — §11. 절대 추정 날짜로 채우지 않는다.
  (실측: 73 폴리곤 중 64개 t 확보, 9개만 진짜 보류 — §11.)

### 4-2. reliable_negative (label=0, certainty=reliable_negative) — R4의 "확실한 쪽"
- **해제**(reason=cancelled): 해제구역 위치(지번)→cancelled.py 파서→parcels seed PNU.
  t = 해제일자. **요건은 됐으나 무산된 최강 네거티브.** (R5로 이후 t에 positive 재등장 가능.)
  ⚠️폴리곤 없어 **seed 기반(full-zone 미커버)** — §4.6.
- **신축파생**(reason=new_construction): building_gis 파생. 필지 as-of-현재 노후도가
  `newbuild.max_old_ratio` 미만이고 positive·해제 어디에도 안 걸리면 → "새 건물 위주라
  가까운 시일 재개발 대상 아님". v1 reliable_neg의 *주력 공급원*(§4.6).

### 4-3. uncertain (label=0, certainty=uncertain) — R4의 "미확정"
- **노후미지정**(reason=not_yet): as-of-현재 노후도 `undesignated.min_old_ratio` 이상
  인데 지정·해제 어디에도 안 걸린 땅. "아직 안 된 것"일 수 있다 → PU 학습기가 가중치↓
  또는 제외. (노후도 0.2~0.5 모호대역은 미라벨/제외 — §4.6.)

### 4-4. 신축파생/노후미지정의 시점: as-of-현재 (t맵 `derived_current`)
신축파생·노후미지정 판정은 as-of-**현재(최신 데이터)** 로 한다. 근거:
1. **우리 설계는 신축파생에 t=현재 행만 생성**하므로 라벨-피처 시점이 정합하다
   (현재 신축 = 현재 기준 neg, 피처도 현재 기준). 〔주의: "현재 신축이면 과거에도
   더 새것"이라는 추론은 **재건축 필지에서 거짓**이다 — 2024 신축 필지는 2019엔
   구축이었고 현재 대장엔 새 건물만 남는다. 그래서 과거 시점 행을 만들지 않고
   *현재 행만* 만든다는 점이 정당화의 핵심이다.〕
2. R1 누수는 "지정의 결과가 원인으로 새는" **positive 문제** — 미지정 negative엔
   동결 효과가 없어 해당 없음.
- 단 positive(과거 t)와 파생 negative(현재 t)의 시대 불일치는 교란 → §9에서 완화.

### 4-5. 충돌 해소
같은 `(pnu, t)`에 두 소스가 들어오면 우선순위(positive > reliable_negative >
uncertain)로 1행. 단 **다른 t면 별개 행으로 둔다**(R5는 충돌이 아니다).

---

## 4.5 zone → parcels 확장 경로 (지뢰3 해소 경로)

| 소스 | WHERE 해석 경로 | 한계 |
|---|---|---|
| 신통(구역명) | ①구역명→의제처리 폴리곤(이름/공간 매칭)→∩parcels ②실패시 대표지번 seed | 폴리곤 미매칭 신통은 PNU 해석 부분적 |
| 정비사업(지번) | 지번 seed→parcels jibun index→contains 폴리곤→∩parcels(full zone) | 시드가 폴리곤 밖이면 seed만 |
| 해제(위치 지번) | cancelled.py 파서(행정동→법정동·일대·산·공백없음)→jibun index→seed PNU | ⚠️폴리곤 없어 **seed/근방만, full-zone 미커버** |

공유 헬퍼 `_resolve_zone_to_pnus(polygon, parcels)` = 폴리곤 ∩ parcels 대표점(within).

---

## 4.6 ★네거티브 커버리지 & 클래스 균형 (R4/R8) — v1 실측

[4구 실측, 2026-06-11]
| 클래스 | 필지 수(추정) | 비고 |
|---|---|---|
| positive (full-zone) | ~18,400 | 73폴리곤 ∩ parcels, 구역당 평균 340·중앙값 263 |
| 해제 reliable_neg (seed) | ~38 | 구역당 1필지 |
| 신축파생 reliable_neg | ~29,200 | 노후도<0.2 & 미지정 |
| 노후미지정 uncertain | ~45,000 | 노후도≥0.5 중 positive 제외 |

★4구 노후도는 **bimodal**: `<0.2`에 29,190필지, `≥0.7`에 63,015, 그 사이(0.2~0.5)는
**단 73필지**. 신축/노후가 자연히 갈린다(→ 0.2 컷 정당화, §config).

**[결론] 해제 seed-only는 R4 균형을 깨지 않는다** — reliable_neg는 해제(38)가 아니라
**신축파생(29K)이 떠받친다**. 해제 38은 "요건됐으나 무산"의 소수·고품질 신호로 유지.
positive:전체negative ≈ **1:4**(R8 정상 불균형) → weighted/focal loss + PR-AUC·F1.
정확도 평가 금지(R8).

**[v1.1 해제 확장 옵션 — 설계만, v1 미적용]**
- **k-hop 확장**: 해제 seed에서 인접 그래프 k-hop 이내 + 노후도 조건 충족 필지만 추가,
  `certainty='neg_expanded'`로 별도 표시(원 seed와 구분, 가중치 차등 가능).
- **면적 기반**: 해제 CSV `면적_m2`로 seed 중심 반경 추정 → 그 안 필지.
- 둘 다 "해제구역 경계 폴리곤 부재"의 우회책. 폴리곤 입수 시 불필요.

---

## 5. R4·R5·R11 처리 위치 (명시)

| 리스크 | 어디서 | 어떻게 |
|---|---|---|
| **R4** | `certainty` 컬럼 | {신통,정비사업}=positive / {해제,신축파생}=reliable_negative / 노후미지정=uncertain. PU 학습기가 uncertain 차등. |
| **R5** | 자연 발생 | 같은 PNU가 해제(t1,neg)+신통/정비사업(t2,pos) → 자동 두 행. 충돌해소는 *같은 (pnu,t)* 에서만. |
| **R11** | `_resolve_pnu_over_time` | 과거 PNU가 **현재 지적도(parcels)에 존재하나** 확인 → 실패 시 drop + dropped 리포트. parcels가 "현재 필지" 기준. |

---

## 6. 데이터 흐름 & ingest 의존성

```
parcels(지적도) ─ 스파인(현재 필지·geometry·지번 index) ─┐
building_gis ─┬→ aging.노후도 as-of-t ─┬→ R2 오염 / 신축파생·노후미지정      │
              └──────────────────────┘                                  │
의제처리 폴리곤 ── WHERE(경계) ──┐                                          │
신통(선정일) ───── WHEN+pos ────┼→ positive (폴리곤 ∩ parcels)             │
정비사업(지정일) ─ WHEN+pos ────┘                                          │
해제(위치) ─ cancelled.py 파서 → seed PNU → reliable_neg(t=해제일) ←R5     │
                         │                                                │
                         ▼                                                │
                  labels.build_label_table() ◀───────────────────────────┘
                         │
            ┌────────────┼─────────────┐
            ▼            ▼              ▼
      라벨 테이블   dropped(R11·보류)   stats 리포트
```

★ labels는 parcels·building_gis·의제처리·신통·정비사업·해제에 의존. `transactions.py`
(AVM)·`regulation.py`는 Phase 5. 배제레이어는 v1 보류(§11).

---

## 7. 함수/파일 분해 (④)

`data/labels.py`
| 함수 | 역할 |
|---|---|
| `build_label_table(sources, cfg) -> (df, report)` | 오케스트레이터. 아래 호출·병합·충돌해소. |
| `_positives_from_shintong(sht, zones, parcels)` | 신통 + 폴리곤매칭 → positive rows. |
| `_positives_from_jeonbisaeop(jb, zones, parcels)` | 정비사업CSV 지번→폴리곤 → positive rows + t. |
| `_negatives_from_cancelled(canc, parcels)` | cancelled.py 파서 → seed PNU → reliable_neg. |
| `_negatives_from_newbuild(parcels, aging, cfg)` | as-of-현재 노후도<cut → 신축파생 reliable_neg. |
| `_uncertain_old_undesignated(parcels, aging, cfg)` | 노후도≥cut & 미지정 → uncertain. |
| `_resolve_zone_to_pnus(polygon, parcels)` | 폴리곤 ∩ parcels (공유). |
| `_match_name_to_zone(name, zones)` | 신통 구역명→의제처리 폴리곤. |
| `_resolve_pnu_over_time(rows, parcels)` | R11 현재필지 존재확인·drop. |
| `_flag_contamination(rows, aging, cfg)` | R2 오염 플래그(t-노후도<컷). |
| `_resolve_conflicts(rows)` | 같은 (pnu,t) 우선순위 병합. |

`data/aging.py` (구현됨) — `old_ratio_as_of`/`old_ratio_by_parcel`: 시점 정합성 단일 정의처.

---

## 8. 시점 정합성 닻 — 노후도 as-of-t (R1)

`old_ratio_as_of(buildings, t)`: 건물 중 **사용승인일이 t 이전인 것만** 추려, config의
경과연수 기준(`building_aging.rc_years` 등)을 t 시점으로 적용해 노후 비율을 낸다.
핵심은 "t 이후 건물·정보는 없는 것처럼" 본다는 것 — R1을 막는 단 하나의 규율.
(R2: 이렇게 계산해도 positive가 컷 미만이면 철거·완공 오염.)

---

## 9. ★시점 불일치 교란 완화 (positive 과거 t vs 파생 negative 현재 t)

positive는 t=2002~2023, 파생 negative(신축파생·노후미지정)는 t=현재다. 이 **시대 차이는
실서비스에서 체계적 오예측으로 직결되는 교란**이다(특정 연식 분포의 동네를 일괄 오판).
v1에서 **싼 부분부터** 줄인다 — 시대 의존 피처의 상대화(구현은 `features.py`/Phase 2,
config on/off):
- **가격류 피처(공시지가 등): 절대값 금지 → 동일 (연도, 자치구) 내 백분위로 정규화.**
  (절대 가격은 시대마다 레벨이 달라 t-불일치를 그대로 학습한다.)
- **노후도·접도·거리류: 시간 안정적이라 그대로.**
- t-분포 매칭 샘플링(파생 negative의 t를 positive t분포에 맞춤)은 **v1.1**. §13 리스크 등급.

---

## 10. 검토했지만 버린 대안 (③)

| 대안 | 왜 버렸나 |
|---|---|
| `(필지)` 정적 라벨 | R1/R5 구조적으로 안 풀림. t 닻 부재. |
| 동적/시계열 GNN(TGN) | 과한 도구. 필요한 건 "시점별 올바른 스냅샷"(§2-3). R7 악화. |
| t-less 의제처리 폴리곤을 negative로 | 진짜 재개발 구역을 negative로 학습 = R4 재앙. → 보류(제외). |
| 의제처리 고시일(최신 변경)을 t로 | R1 누수(2025 변경일→t 밀림). 정비사업 CSV 최초지정일로 대체. |
| 해제 seed를 무조건 확장 | 경계 폴리곤 부재로 부정확. v1은 seed, 확장은 v1.1 옵션(§4.6). |
| 가격 피처 절대값 사용 | t-불일치를 그대로 학습(§9). (연도,구) 백분위로 상대화. |

---

## 11. ★보류(제외) 정책

- **t 없는 UQ181 의제처리 잔여 구역**(신통·정비사업 미매칭)은 **제외(보류)**. 절대
  추정 날짜로 채우지 않는다. dropped 리포트에 `reason=no_clean_t`로 기록. 지정일
  확보(상세 스크래핑 등) 시 v1.1 합류.
  - ★**UQ1222(도시정비형) 8개는 전부 보류**(t 미확보) — 신통·정비사업 CSV가
    주택재개발 중심이라 도시정비형 t가 안 나올 가능성 높다. dropped에
    `reason=no_clean_t, note=uq1222`로 표기. **v1.1 고시 스크래핑 1순위 후보.**
  - ※in-scope 73 유지(재건축·타법 오염 0 실측). **매칭 개선으로 보류 66%(48)→4%(3)**:
    OA 최초결정 49 + 촉진부모 17 + ntfc_direct 4 = 70/73 t 확보(2026-06-11 최종측정).
  - **최종 진짜 보류 3개 = 연신내·독바위·사당동252** — 전부 NTFC 제목이 "변경"이라
    ntfc_direct 탈락 → 깨끗한 지정 증거 없음 → **추정 금지로 보류 유지**.
  - ★**남구로 판정**(역명 중복 주의): UQ 보류 "남구로" = **남구로역세권 도시정비형
    재개발(2020-08-04)** 으로, 1-b 공공재개발 `구로동252일대`(2022-08-26, 검증됨)와
    **별개 구역**(같은 역세권, 다른 사업). 구로동252는 UQ 폴리곤 매칭 시 라벨, 없으면
    폴리곤 부재 기록만.
- **노후도 0.2~0.5 모호대역**(~73필지): 미라벨/제외.
- **배제레이어**: 문화재 SHP 1건뿐(불완전) → v1 보류. 데이터 보강 후 재도입.

---

## 12. 확정된 결정

**2026-06-10**
1. 데이터 제공 = 작업자(실데이터). 2. `aging.py` 신설. 3. t granularity = 연도.
4. v1 인접 그래프 = 현재 필지 그래프(과거 PNU resolve, R11 drop). 5. 범위 = 키스톤까지.

**2026-06-11 (이번 갱신)**
6. positive 소스 1→3(신통/정비사업/의제처리-geometry), **WHERE·WHEN 분리**.
7. t 소스맵 확정(§3.5). 의제처리 t는 정비사업 CSV로 **부분 해결**(2021 스냅샷 한계).
8. 신축파생 정의 + `newbuild.max_old_ratio`(0.2, bimodal 갭 근거), `undesignated.min_old_ratio`(0.5).
9. 신축파생/노후미지정 = as-of-현재(t=현재 행만 생성 → 라벨-피처 정합, §4-4).
10. 해제 seed-only 수용(R4 균형은 신축파생이 떠받침, §4.6). k-hop 확장은 v1.1.
11. 배제레이어 v1 보류.

---

## 13. 미해결 / 리스크 등급

- **[서비스 품질 리스크] 시점 불일치 교란.** positive(과거)·파생 negative(현재)의
  시대 차이 → 실서비스 체계적 오예측 가능. v1: 가격 피처 (연도,구) 백분위 상대화로
  부분 완화(§9). v1.1: t-분포 매칭 샘플링. **출력 정직성에 직접 영향 → 우선 추적.**
- **[커버리지] 해제 full-zone 미커버**(seed 기반). v1.1 k-hop/면적 확장(§4.6).
- **[커버리지] 신통 폴리곤 미매칭분** PNU 해석 부분적(§4.5).
- **[t 부분해결] 의제처리 잔여 구역 보류**(§11) — 지정일 확보 시 합류.
- **[학습분포 왜곡 리스크 — 대부분 해소] 보류 66%→4%(3/73).** 최종 구별: 성북 0% /
  구로 0% / 동작 5% / 은평 8%. 쏠림 사실상 소멸. 잔여 3(연신내·독바위·사당동252)은
  "변경" 제목이라 추정 금지로 보류 — 깨끗한 지정 증거 확보 시(상세 고시 추적) 합류.
  영향 미미(3건)라 v1 진행에 지장 없음.

---

*승인 완료(2026-06-11). 구현 순서: cancelled.py(해제 파서) → shintong/정비사업 리더
→ labels.py(조립). 각 파일 "어디에 앉는지" 한 문단 + 검증.*

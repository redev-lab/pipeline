# 설계 노트 — zone_matching 승격 + `zone_boundary.py` 정식화 (positive 라벨)

> 규칙1. **t-전쟁(2026-06-11, 수검 4/4 통과)에서 검증한 로직을 scripts/ 분석코드
> → redev/ 파이프라인 정식 모듈로 승격**한다. 이 노트는 지뢰2 시절 초안(단순
> 결정고시관리코드 조인)을 *전면 대체*한다 — 그 단순 조인은 "최신 변경고시일"을
> 줘서 R1 누수를 일으켰고, OA 이력 마이닝으로 대체됐다.

## 1. 이 모듈이 하는 일 (①)
의제처리구역(UQ181) 폴리곤 73개(4구)에 **최초 정비구역 지정일(t)**을 붙여
**positive ZoneTable**을 만든다. WHERE=폴리곤(경계), WHEN=OA-20283 이력 마이닝.
출력은 §11 소스 추상화 스키마: `(zone_id, geometry, t, t_source, t_alt, zone_type,
sigungu, source)`. 보류(t 미확보)는 dropped 리포트로.

## 2. 무엇을 승격하나 (검증된 로직)
**A. 순수 매칭 헬퍼** `scripts/zone_matching.py` → **`redev/data/zone_matching.py`**
  - `clean_text`(연결자·cp949 깨짐) / `normalize_zone_name`(구역 토큰, '제' 비그리디)
  / `is_redev_title`(재건축·후속단계 배제) / `region_of`(동-strip) / `parent_of`(촉진
  부모) / `cycle_done`(관리처분·준공). 순수함수·`re`만 의존 → 이동 무위험.
  - scripts/ 원본은 **thin re-export로 축소**(분석 재현 호환), redev/가 canonical.

**B. t-해소 로직** (지금 audit 스크립트에만 있음) → **`zone_boundary.py`**
  검증된 6소스 earliest-genuine + 사이클 가드를 정식 함수로.

## 3. t-해소 알고리즘 (②) — 수검 통과본 그대로
1. **OA 토큰 인덱스(사이클 인지)**: 결정고시(OA-20283)를 토큰·동맵으로 묶어,
   각 (시군구,토큰)의 **마지막 완료고시(관리처분/준공) 이후 최초 지정결정**을 t로.
   (흑석2: 1985 개량재개발→1986 관리처분(완료)→2025 새 촉진 ⇒ 1985 배제, 2008 채택.)
2. **후보 t 수집(6소스)**: ①OA ②촉진 부모(promotion_parent, 근사) ③ntfc_direct
   (NTFC 제목 "지정"&¬"변경" + OA없음) ④정비사업CSV ⑤신통선정일 ⑥공공재개발선정일.
3. **earliest-genuine 채택**: 후보 중 최소 날짜 = t. **나머지 날짜 = `t_alt`(감사 보존).**
4. **보류**: 후보 0 또는 "변경"뿐 → t 없음, dropped(`reason=no_clean_t`). 추정 금지.

## 4. 설계 결정 (②)
- **★t의 의미 = "결정이 내려져 동결(행위제한)이 시작된 최초 시점".** 신통은 선정
  즉시 토허구역 지정·권리산정기준일·건축허가 제한이 걸리므로 동결 시작점은
  지정고시가 아니라 **선정일**이다. t=지정고시일로 잡으면 선정~지정 사이 동결된
  기간의 피처(노후도 등)에 *선정의 결과*가 새어 들어가 R1 위반. 따라서 earliest가
  신통선정일을 채택하는 건 보수적 선택이 아니라 **인과적으로 옳은 선택**. '선정'과
  '지정' 구분은 t_source(`shintong_select` vs `first_designation_gosi`)로 추적되고,
  지정고시일은 `t_alt`에 보존.
- **소스 우선순위가 아니라 earliest-genuine**: 수검이 "OA 우선이라 더 늦은 t 채택"
  버그를 잡았다(흑석11 OA2021 vs 정비2012). 전 소스 통틀어 최소를 t로.
- **사이클 가드**: 재개발은 한 땅이 여러 사이클(완공→재지정). 완료고시(관리처분)
  *이전* 지정은 옛 사이클 → R1상 현재 freeze가 아니다. 완료 이후만 현재 사이클.
- **헬퍼 순수성**: zone_matching은 I/O 없는 순수함수 → 단위테스트로 고정(수검 §1).
- **소스 추상화(§11)**: 출력 ZoneTable은 소스 무관. 상업 전환 시 의제처리→상업소스
  교체 + 마이닝 재실행만.
- **WHERE=폴리곤 ∩ parcels는 labels.py 책임**(여기선 폴리곤+t까지). 신통/공공재개발의
  UQ-밖 standalone positive도 labels.py에서.

## 5. 함수 분해 (④)
`redev/data/zone_matching.py` (헬퍼 승격, 순수)
| 함수 | 역할 |
|---|---|
| clean_text / normalize_zone_name / region_of / parent_of / is_redev_title / cycle_done | t-전쟁 검증본. 단위테스트 동반. |

`redev/data/ingest/zone_boundary.py`
| 함수 | 역할 |
|---|---|
| `build_dong_to_sigungu(parcels) -> dict` | 동명→구 맵(자치구 가드. 고시코드 금지). |
| `build_oa_token_index(gosi, dong_map) -> dict` | (sig,토큰)→(최초지정일,제목), **사이클 인지**. |
| `_promotion_parent_dates(gosi, dong_map) -> dict` | 촉진 부모 최초결정. |
| `resolve_zone_t(zone_row, oa_idx, pmin, xsrc) -> (t, t_source, t_alt)` | 6소스 earliest-genuine. |
| `load_zones(uq_path, gosi_path, parcels, xsrc_paths) -> (ZoneTable, report)` | 메인. 폴리곤 5186 + t. |

## 6. 검토했지만 버린 대안 (③)
| 대안 | 기각 |
|---|---|
| 단순 NTFC_SN↔고시관리코드 조인의 고시일 | "최신 변경"→R1 누수(t-전쟁 출발점). OA 이력 마이닝으로. |
| 소스 우선순위(OA 먼저) | earliest 위반(수검). 전 소스 최소. |
| 완료 무시하고 전 이력 최소 | 멀티사이클 옛 사이클 오채택(흑석2=1985). 사이클 가드. |
| 헬퍼를 zone_boundary에 인라인 | 공공재개발·labels 재사용 막힘. 별도 순수 모듈. |
| scripts/ 직접 import | 분석코드를 파이프라인이 의존=역방향. redev/ canonical. |

## 7. ★수검 계획 (규칙9 — 종료 게이트)
승격 후 **redev 코드로 동일 수검 재실행**, scripts 결과(보류 4%, 70/73, t_source
분포)와 **일치 확인**이 곧 종료 증거:
- ① 표본 감사: t_source별 2~3건 근거 추적
- ② 분포: 연도 bimodal(2007~2011/2024~25), 이상치<2003 = 2001·2002만
- ③ 정합성: t>오늘 0, NTFC 충돌 0, t_alt 보존 17, 흑석2=2008, 잔여 3=변경
- ④ 단위테스트: zone_matching 헬퍼(돈암제6→돈암6, 안암동2가 보존, 재건축/시행인가 배제, 사이클 가드)

## 8. 미해결 / 주의
- 잔여 3(연신내·독바위·사당동252) = "변경"뿐 → 보류 유지(v1.1 상세 고시 추적).
- 멀티사이클 가드는 4구 1건(흑석2) 검증 — v2 전역에선 빈발, 재수검.
- 공공재개발 응암동101/구로동252 t=2022-08-26(보도자료 검증), 출처URL → ingest t_alt 메타.

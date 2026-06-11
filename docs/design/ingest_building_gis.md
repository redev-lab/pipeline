# 설계 노트 — `data/ingest/building_gis.py` (GIS건물통합정보 적재)

> 규칙1. 라벨 키스톤의 첫 재료. 이 리더의 출력이 `aging.py`(R1)의 입력이다.
> 상위 설계: [labels.md](labels.md) §정제 건물 테이블 계약.

## 1. 하는 일 (①)
원천 SHP(AL_D010, 695,769 건물, 1.2GB dbf)를 읽어, **정제 건물 테이블**로
변환한다. 노후도(R1)에 필요한 최소 컬럼만 뽑고, 깨진 행을 거르고, 좌표계를
통일한다. 출력은 `aging.py`가 그대로 먹는 스키마.

## 2. A-코드 → 정제 스키마 매핑 (실데이터 EDA로 확정)
| 원천 | 정제 컬럼 | 처리 |
|---|---|---|
| A2 | `pnu` | `redev.data.pnu.normalize_pnu` 통과(19자리 보장·float거부) |
| A13 | `approval_year` | `"1991-09-02"` → 1991(int). None/이상 → NA |
| A11 | `structure` | 구조명(cp949) → `'rc'`/`'other'` 분류(키워드) |
| A14 | `gross_floor_area` | 연면적(float). 결측 허용 |
| A7 | `land_div` | `'일반'`/`'산'` 토지대장구분. ⚠️EDA로 확인: 집합건물 플래그 아님(PNU 필지구분과 동일). 집합건물 식별은 별도 필드 미해결 |
| A23 | `sigungu` | 시군구코드(5) — 구 필터 편의 |

## 3. 설계 결정 (②)
- **EPSG:5186 reproject 규칙(지뢰1).** 모든 ingest는 geometry를 읽으면 즉시
  `TARGET_CRS=EPSG:5186`으로 재투영한 뒤 넘긴다. 건물은 이미 5186이라 사실상
  no-op이지만 **규칙은 모든 리더에 균일 적용**(의제처리=2097에서 진짜 필요).
  → 공유 헬퍼 `redev.data.geo.to_target_crs()` 신설.
- **속성 우선(R10 메모리).** 노후도는 건물 *속성*만 필요(geometry 불필요).
  기본 `with_geometry=False` + 필요한 컬럼만 읽어 1.2GB를 다 안 올린다.
  geometry는 옵션(호수밀도·형상은 Phase 2).
- **cp949 인코딩.** 건물 dbf엔 .cpg가 없어 기본 latin로 깨진다 → `encoding='cp949'`.
- **structure 분류는 taxonomy(코드 아님).** rc 키워드(철근콘크리트·철골·강구조 등)
  substring 매칭, 그 외/결측 → 'other'(보수적: 더 짧은 경과연수 → 노후 판정이
  관대 → R2 오탐 drop 감소). 경과연수 *임계값*은 config(규칙5), 분류 키워드는 코드.
- **PNU/결측 위생(R10).** normalize 실패·PNU 결측 행은 drop + 카운트 리포트.
  approval_year 결측은 **drop 안 함**(aging이 t필터에서 자연 제외) — 단 비율 기록.

## 4. 검토했지만 버린 대안 (③)
| 대안 | 기각 이유 |
|---|---|
| 전체 컬럼·geometry 통째 로드 | 1.2GB+129MB 메모리 폭발(R10). 필요한 6컬럼만. |
| structure 임계값까지 코드에 | 규칙5 위반. 연수는 config, 분류만 코드. |
| approval_year 결측 행 drop | 정보 손실. aging이 시점필터로 알아서 뺀다(거짓0 대신 NaN). |
| PNU를 그대로 신뢰 | float/길이오류 잠입. normalize_pnu 관문 통과 강제. |

## 5. 함수 분해 (④)
`data/geo.py` (신설, 공유)
- `TARGET_CRS = "EPSG:5186"`, `to_target_crs(gdf)` — 모든 ingest의 좌표계 닻.

`data/ingest/building_gis.py`
| 함수 | 역할 |
|---|---|
| `load_buildings(path, *, with_geometry=False) -> (df, report)` | SHP→정제 테이블. 메인. |
| `_classify_structure(name) -> 'rc'\|'other'` | 구조명 키워드 분류. |
| `_parse_approval_year(s) -> int\|NA` | A13 → 연도. |
| (내부) PNU normalize·결측 drop·리포트 집계 | R10 위생. |

## 6. 검증
실데이터로: ① 컬럼·결측률 리포트 ② normalize 통과율 ③ **aging.old_ratio_by_parcel
로 R1 재검증** — 같은 필지를 과거 t / 현재 t로 노후도 계산해 차이 확인(합성이 아닌
진짜 서울 건물로).

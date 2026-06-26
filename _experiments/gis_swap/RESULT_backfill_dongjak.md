# 동작 backfill 수집 + hybrid aging 회복 검증 (2026-06-23)

> _experiments/gis_swap/ 전용, redev/·git 미변경. 건축HUB 표제부로 national 누락 PNU 사용승인일 보충 → 회복 측정.

## 1. 수집 현황 (한도 내 완료)
- 동작 backfill 대상 **5,783 PNU** 전량 호출(우리 키 1개, 호출 5,783 — 일 한도 10,000 내).
- ★**사용승인일 채워진 PNU 2,994 / 5,783 = 52%** · 건물행 3,055(다중 표제부 44건).
- ★**빈 48%(2,789 PNU)**: 건축HUB 표제부에도 응답 0. 종로 샘플(86%)보다 크게 낮음 → 동작 특이.
- useAprDay 포맷 YYYYMMDD 확정(1956~2026), PNU→주소 정확.

## 2. ★hybrid aging candidate 회복률 (핵심)
national 기본 + 동작 backfill 2,994 PNU 합쳐 동작 점수 재계산:
| | 값 |
|---|---|
| 서울 기준 top10% candidate | 3,069 |
| national에서 top10% 밖 추락(손실) | 914 (30%) |
| ★**hybrid로 top10% 복귀** | **227/914 (25% 회복)** |
| Spearman(서울 기준) | national 0.824 → **hybrid 0.870** |
| 백분위 \|Δ\| 중앙 | national 7.7%p → **hybrid 5.7%p** |

→ backfill이 **개선은 시키나(0.82→0.87) candidate는 25%만 회복**. 절반(52%) PNU를 채웠는데 회복이 25%인 이유:
**① 48% PNU 여전히 빈 표제부 ② 점수는 2-hop 이웃집계 의존** — 한 필지 aging을 복원해도 *이웃*이 아직 national-0이면
이웃집계 피처가 낮게 유지돼 top10% 복귀 못 함. 즉 커버리지 구멍이 군집 단위로 점수를 계속 끌어내림.

## 3. ★48% 빈 표제부 — 추정 원인 (다음 관문)
backfill 대상은 "서울은 사용승인일 보유"인 필지인데 건축HUB 표제부가 빈 응답 = **번지 단위 조회 미스** 추정:
- 건물 대장이 **대표지번**에 등록(여러 필지 걸친 건물·집합) → PNU의 정확한 부번으로 조회하면 0.
- 일부는 **총괄표제부(getBrRecapTitleInfo)** 에만 있거나 멸실/재축으로 표제부 구조 상이.
→ getBrTitleInfo(번지 정확매칭)만으론 동작 절반을 못 잡음. **총괄표제부·부속지번(getBrAtchJibunInfo) 보완 조회**로
커버리지를 올리면 회복률도 올라갈 여지.

## ★결론
- 건축HUB 표제부 backfill = **부분 해결**(동작 candidate 25% 회복, Spearman 0.82→0.87). **단독으론 불충분.**
- 병목은 **건축HUB 표제부의 동작 커버리지 52%** + 이웃집계. → (a) 총괄표제부·부속지번 추가 조회로 커버리지 ↑,
  또는 (b) backfill 후에도 남는 빈 필지를 '데이터 부족' 정직 표기.
- national swap의 노후도 관문은 **표제부 단일 조회로는 안 닫힘** — 멀티-오퍼레이션 보완이 다음 단계.

## 산출물
- `backfill_dongjak.parquet`(3,055 건물행) · `backfill_dongjak_done.csv`(5,783 처리) · `collect_dongjak.py` · `hybrid_recovery.py`.

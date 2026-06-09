"""data.ingest — 소스별 적재기 (Phase 1, 일정 최대 병목).

역할: 각 파일이 한 원천을 담당해 적재한다. 모든 적재기는 (1) PNU 매핑
확인 → (2) 적재 → (3) 결측·이상치 EDA 순서를 따른다.

★불변식: PNU 컬럼은 반드시 dtype=str로 읽는다 (float 추론 시 19자리
정밀도가 깨짐 — docs/design/foundation.md 결정 A, 1차 방어선).

(아직 비어 있음 — Phase 1에서 building_gis / zone_boundary / shintong /
cancelled / transactions / regulation 적재기를 채운다.)
"""

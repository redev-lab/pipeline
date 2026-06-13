# 설계 노트 — Phase 8 데모 (추론 구 확장 + 백엔드 + 프론트 + 데모 3종)

> 규칙1: 코드 전에 이 노트. 승인 후 구현(한 모듈씩). ★학습 동결 — production B1+ 안 건드린다.

## 0. 한 문단 요약

처음으로 화면이 나온다. ① 추론을 학습 4구 밖(마포·강남)으로 확장(첫 증명) ② FastAPI가
`pipeline.run()`을 직접 import(파이썬이라 다리 불필요) ③ React+Leaflet 지도+리포트+스크리너
④ 데모 3종(후보지·광흥창·역삼). ★학습은 동결 — 추론 구는 production B1+로 *점수만* 낸다.

## 0-1. 추론 구 확장 (마포 11440·강남 11680 — 학습 밖 추론 첫 증명)

config 추론 구 목록엔 이미 6구 등록(`inference_districts`). 지적도·건물은 서울 전체분이라 클립만.
- **빌드**: inference_sigungu_codes(6구) parcels → `build_graph`(구 단위 배치, 구간 엣지 0) →
  전 노드 현재시점 피처(node_features 10차원) → 이웃집계 → **production B1+(−용도지역)로 스코어**.
  → `infer_features_6gu.parquet` + 점수 캐시. ★학습행렬·모델 불변(동결).
- ★**수검**: 마포·강남 점수 분포가 4구와 상식 정합 — 강남 신축지대 점수 낮음, 마포 노후 일부 높음.
  분포·백분위 비교표 + 표본 육안.

## 1. 백엔드 — FastAPI (`demo/backend/`)

`pipeline.run()` 직접 import 한 겹. (★Spring Boot 등 별도 BE는 v2 운영 전환 시 검토 — 지금은 파이썬
단일 프로세스라 다리 불필요. "버린 대안"에 기록.)
- **POST /report** {address, property_type?, stage?} → `run(address, ctx, with_report=True)` JSON.
- **GET /screen** ?gu&min_pct&toheo → ★스크리너: 전 노드 점수 캐시를 정렬·필터 → 상위 필지·클러스터
  리스트. (캐시 조회라 거의 공짜 — 응답 빠름이 수검.)
- ★**도로명주소**: `juso.go.kr` 무료 API로 도로명→지번 변환을 run() 앞단에. 키 `.env`(JUSO_API_KEY).
  ★키 없거나 실패 → 기존 지번 파서 폴백(데모는 지번 주소로도 동작). 변환 경로 수검.
- ★**예측 로그**: 모든 추론(주소·PNU·점수·경로·토허·타임스탬프)을 `_data/logs/predictions.jsonl`
  append — track record의 시작. ★개인정보 없음(주소는 공개 지번, 사용자 식별 정보 미수집) 확인.
- ctx(build_context)는 서버 기동 시 1회 로드(196s) — 이후 요청은 캐시 조회(68ms/주소).

## 2. 프론트 — React + Leaflet (`demo/frontend/`)

무료 타일(OSM). 두 화면:
- **주소 검색**: 주소 → 지도(해당 동네 ★백분위 히트맵 + 후보 클러스터 외곽선) + 리포트 패널.
- **스크리너 탭**: 필터(구·점수백분위·토허) → 결과 리스트 → 클릭 시 지도 이동.
- ★**렌더 절제**: 14만 필지 통짜 렌더 금지 — 화면 영역(viewport bbox) 필지만 요청, 또는 클러스터·
  그리드 집계 표시(줌 레벨별). 백엔드가 bbox 필터 제공.
- ★**리포트 UX 위계**: 맨 위 ★팩트 3줄(환경 상위 X% · 요건 N/5 · 토허 여부), caveat은 접힘/펼침
  (내용 동일, 위계만 — 정직성 유지하되 첫 화면은 요약). 출처 태그는 hover/펼침.

## 3. 데모 3종 + README

- ① **학습구 내 실제 후보지**(성북/은평 중 1) — 정상 후보 흐름.
- ② **마포 광흥창**(추론 전용 구) — 학습 밖 추론 증명. (신축 혼재 → 모아타운 가능성.)
- ③ **강남 역삼**(명백한 비대상) — ★"해당없음"이 정확히 나오는 것도 데모(정직성).
- 각각 스크린샷 + README 검증 리포트: 점수표 사다리·IoU·★세 사이클 서사(R9/재경기/PU) 요약 + 한계.

## 4. 함수/파일 분해 (★조직 레포 3개 — 레포별 PR)

```text
pipeline/ (현재 repo, phase-8-demo)         # [0] + export 인터페이스(backend가 import)
├── redev/serve/infer_districts.py   # 6구 graph·피처·점수 빌드·캐시(학습 동결)
└── redev/serve/api.py               # ★export: report(addr)·screen(filters)·geocode 폴백 — 순수 파이썬
backend/ (redev-lab/backend repo)           # [1] FastAPI, pip install -e ../pipeline
├── geocode.py(juso)·logging(JSONL)·app.py(/report·/screen+bbox)
frontend/ (redev-lab/frontend repo)         # [2] React+Leaflet: 지도·리포트·스크리너(렌더절제·위계)
docs/demo_report.md (pipeline)              # 데모 3종 검증 + 스크린샷
```

★**경계**: pipeline은 *순수 파이썬 export*(`redev.serve`)까지만 — FastAPI·HTTP는 backend. juso
호출도 backend(키·네트워크). pipeline의 geocode는 지번 파서 폴백(순수). 의존 방향 = backend→pipeline.

## 5. ★수검 (규칙9)

1. **추론 구 확장**: 마포·강남 점수 분포 vs 4구 정합(강남 신축 낮음 등) + 표본 육안.
2. **3주소 e2e**: ①학습구 ②마포(추론전용) ③강남 — run(with_report) 통과·환각 0 유지.
3. **스크리너 응답속도**: /screen 캐시 조회 < N ms(수치 측정).
4. **도로명 변환**: juso 경로(키 있으면) + 폴백(키 없으면) 둘 다 동작.
5. **예측 로그 적재**: 추론 후 JSONL에 행 추가 확인 + 개인정보 없음.
6. **화면 육안**: 히트맵·클러스터·리포트 위계 스크린샷 3장.

## 6. 검토했지만 버린 대안

- **Spring Boot 등 별도 BE**: 파이썬 단일 프로세스라 다리 불필요(§8 과한 도구) → FastAPI 직접 import. v2 운영 전환 시 검토.
- **14만 필지 통짜 렌더**: 브라우저 폭사 → viewport bbox·집계.
- **학습 구 재학습으로 6구 포함**: 학습 동결 원칙 위반·불필요 → production B1+로 점수만(inductive).

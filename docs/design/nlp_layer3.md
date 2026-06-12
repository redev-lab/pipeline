# 설계 노트 — `llm/client.py` + `nlp/layer3.py` (Phase 7, ② 사회신호)

> 규칙1: 코드 전에 이 노트. 승인 후 구현. CLAUDE.md 규칙4(숫자 금지)·§8·§5(R18) 우선.

## 0. 한 문단 요약

`client.py`는 Gemini를 **한 겹으로 감싼다** — `complete(system, user) -> text` 하나, 재시도+백오프,
실패 시 호출부가 템플릿 폴백 가능(견고성). `layer3.py`는 노후도가 *아닌* **직교 축(갈등·정체·진행
신호)**을 텍스트에서 뽑는다 — R18 외생변수의 프록시. ★LLM은 숫자를 짓지 않고(규칙4) 주어진
텍스트에서 **구조화 신호만 추출**한다. ★신호 부재 시 "신호 없음"이 정상 출력(억지 신호=불합격).

## 1. `llm/client.py` — Gemini 한 겹 래퍼

- **인터페이스 하나**: `complete(system: str, user: str, *, temperature=0.0) -> str`. 모델은 무료
  티어 친화 flash(예: gemini-2.0-flash, config). 키는 `.env`의 `GEMINI_API_KEY`(하드코딩 금지).
- **재시도+백오프**: rate limit(429)·일시 오류에 지수 백오프 재시도(max_retries config). 무료 티어 대비.
- **실패 처리**: 재시도 소진 시 `LLMError` raise → ★호출부(layer3·report)가 **템플릿 폴백**(LLM
  없어도 결과가 나오는 구조). client는 폴백을 모름(관심사 분리) — "LLM 가능하면 쓰고, 아니면 템플릿".
- temperature=0(결정론 지향 — 같은 입력 같은 출력에 가깝게, 환각·변동 ↓).

## 2. `nlp/layer3.py` — 직교 축(사회신호) 추출

- **목적**: 노후도(물리)로 안 잡히는 축 — 갈등(주민 반대·소송)·정체(분쟁·중단)·진행(속도) 신호.
  R18 외생변수의 텍스트 프록시. 심장1·2가 못 보는 차원.
- **입력 코퍼스(v1)**: 데모 대상 구역의 **뉴스·공고 소수**(구역당 0~수 건). ★수집 방법 제안:
  - v1 데모 = **수동 수집 텍스트 픽스처**(`_data/corpus/<zone>.jsonl`: {text, source, url, date}).
    출처 약관/라이선스 확인 후 *요약·인용 범위*만 저장(전문 재배포 금지). 데모 품질이 목표(스코프 절제).
  - v1.1 = 뉴스 API/공고 상시 크롤링(전 구역) — 약관 검토 포함.
- **추출(LLM)**: 각 텍스트를 client에 넣어 ★**구조화 JSON만** 받음(자유 서술 금지):
  `{signals: [{type: 갈등|정체|진행, direction: 악재|호재|중립, evidence: "근거 문장", source_url}]}`.
  LLM은 *주어진 문장에서* 분류·인용만(규칙4 — 숫자·사실 창작 금지). 프롬프트에 "근거 문장은
  원문 인용", "신호 없으면 빈 배열".
- ★**신호 부재 = "신호 없음" 정상**: 코퍼스 없거나 신호 없으면 `{signals: [], status: "신호 없음"}`.
  억지 신호 생성은 불합격(수검).

## 3. 함수/파일 분해

```text
llm/client.py
├── class LLMError(Exception)
└── complete(system, user, *, temperature=0.0, max_retries=4) -> str   # 재시도+백오프
nlp/layer3.py
├── load_corpus(zone_id) -> list[{text, source, url, date}]            # 데모 픽스처 로드
├── _extract_one(text, client) -> list[signal]                        # LLM 구조화 추출
└── social_signals(zone_id, *, client=None) -> dict                   # 공개: 신호 집계(+무신호 정상)
```

## 4. ★수검 (규칙9, 구현 후)

1. **알려진 갈등 사례 1건 추출**: 갈등 문장이 든 텍스트 → type=갈등·direction=악재·evidence=원문
   인용이 나오는지. (LLM이 분류·인용만 했는지, 사실 창작 0.)
2. **무신호 구역 "없음"**: 코퍼스 빈 구역 → `signals: []`, status "신호 없음"(억지 생성 0).
3. **client 재시도**: rate limit 모의(또는 실패 1회 주입) → 백오프 재시도 동작 확인.
4. **폴백 경로**: client 강제 실패 → 호출부가 템플릿/빈 신호로 죽지 않는지 1회.

## 5. 검토했지만 버린 대안

- **LLM 자유 서술**: 환각·근거 불명 → 구조화 JSON + 원문 인용 강제(규칙4·정직성).
- **전 구역 상시 크롤링**: v1 스코프 초과·약관 리스크 → v1 데모 픽스처, v1.1 크롤.
- **신호 강제 생성**: 무신호를 "약한 신호"로 포장 = 거짓 → "신호 없음"이 정상 출력.
- **client에 폴백 내장**: 관심사 혼합 → client는 raise만, 폴백은 호출부(견고성 책임 분리).

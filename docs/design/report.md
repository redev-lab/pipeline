# 설계 노트 — `llm/report.py` + run() 연결 (Phase 7, ③ ⑨ 종합)

> 규칙1: 코드 전에 이 노트. 승인 후 구현. CLAUDE.md 규칙4(숫자 금지)·§6·§8·R15 우선.

## 0. 한 문단 요약

run(address)의 7단계 JSON(+retrieval+social)을 받아 **5종 판단(될까·얼마·언제·리스크·진입)**
구조의 한국어 리포트로 언어화한다. ★LLM은 숫자를 짓지 않고(규칙4) JSON 값을 **전달·연결·설명**만.
모든 주장 문장에 **[모듈.값] 출처 태그**, 각 모듈의 **caveat 누락 금지**(정직성이 여기서 증발하면
전부 헛것). API 실패 시 **템플릿 폴백**(LLM 없어도 리포트가 나온다).

## 1. 입력 → 출력

- 입력: `run()` 결과 dict(진단_요건·시세맥락·진입 / 예언_환경점수, b1_score, candidate) + retrieval
  (유사 구역) + social(사회신호). 전부 구조화 숫자·사실.
- 출력: `{report_text(한국어), citations_ok(bool), hallucination(다음 §4), source: "llm"|"template"}`.
- 5종 판단 = 될까(환경점수·요건)·얼마(시세맥락)·언제(단계 잔여기간)·리스크(사회신호·hard-neg 천장)·
  진입(토허). 각 절에 근거 태그.

## 2. ★근거 인용 강제

프롬프트에 "모든 주장 문장 끝에 `[모듈명.필드=값]` 출처 태그를 붙여라(예: `[stage1.path=재개발]`).
태그 없는 주장 금지." → 사용자가 어느 수치에서 나온 말인지 추적 가능(정직성·검증성).

## 3. ★caveat 전달 의무

각 모듈 출력의 caveats(근사·미검증·천장)를 리포트 말미 "한계·주의" 절에 **누락 없이** 옮긴다.
시스템 프롬프트가 "입력 JSON의 모든 caveats를 빠짐없이 포함" 강제. R15(권유 아님)도 고정 문구로.

## 4. ★숫자 금지 + 환각 자동검증 (수검 핵심)

- LLM은 JSON에 **없는 숫자를 쓰지 않는다**(프롬프트 강제). 일·월·정확률을 창작하면 거짓.
- **`verify_numbers(report_text, source_json)`**: 리포트의 모든 숫자를 추출해 source JSON의 숫자
  집합과 대조 → **불일치(JSON에 없는 숫자) 목록**. ★합격선 = 불일치 0(구조적 서수 등은 allowlist).
  3개 주소 반복(수검). 불일치 발견 시 리포트에 ⚠️ 표시 또는 폴백.

## 5. 템플릿 폴백 (견고성)

LLM 실패(LLMError) 시 **결정론 템플릿**으로 리포트 생성(JSON 값을 고정 양식에 채움). 언어화 품질은
낮아도 **모든 수치·caveat가 정확**(환각 0 보장). "LLM 가능하면 풍부하게, 아니면 템플릿."

## 6. ReAct 최소 (§8)

동적 지점 하나만: **저신뢰(candidate=False 또는 b1_score 낮음) → 사례검색 추가 조회 if**. 그 외는
직선. LangGraph·복잡 에이전트 루프 금지(§8 — 블랙박스).

## 7. run() 연결 (⑨ 자리 채우기)

`pipeline.run()`의 `llm_summary` placeholder를 대체: 7단계 조립 후 retrieval(search_cases)·
social(social_signals) 호출 → generate_report(전체 JSON). ★retrieval은 context에 zone_vectors
사전구축, social은 데모 코퍼스(대개 "신호 없음"). report는 옵션(`with_report=True`)으로.

## 8. 함수/파일 분해

```text
llm/report.py
├── _template_report(data) -> str               # 폴백(결정론, 환각 0)
├── verify_numbers(report, source) -> {ok, unmatched}   # ★환각 자동검증
└── generate_report(data, *, complete_fn=None) -> dict  # LLM 언어화 + 검증 + 폴백
pipeline.run(..., with_report=True)              # ⑨ 연결(retrieval+social+report)
```

## 9. ★수검 (규칙9, 구현 후)

1. **환각 diff 3주소**: 실제 4구 주소 3개로 generate_report → verify_numbers 불일치 0(합격선).
   불일치 있으면 어떤 숫자인지(서수/창작 구분).
2. **출처 태그 존재**: 리포트 주장 문장에 `[모듈.값]` 태그가 실제로 붙는지.
3. **caveat 누락 0**: 입력 caveats가 리포트에 전부 등장하는지(집합 포함 검사).
4. **폴백**: LLM 강제실패 → 템플릿 리포트 생성·환각 0.
5. **run 연결**: run(address, with_report=True)가 리포트까지 한 번에 나오는지 + CPU 시간.

## 10. 검토했지만 버린 대안

- **LLM이 수치 재계산**: 규칙4 위반·환각 → JSON 전달만 + 검증.
- **자유 서술 리포트**: 출처 불명 → 태그 강제 + caveat 의무.
- **LangGraph 에이전트**: §8 과한 도구 → 직선 + if 하나.

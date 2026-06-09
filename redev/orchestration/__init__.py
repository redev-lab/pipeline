"""orchestration — 명시적 오케스트레이터 (Phase 7).

역할: run(address) 가 ①~⑧ 단계를 순서대로 호출하는 plain 파이썬 흐름.
동적 지점은 둘뿐(GNN 저신뢰→집계구 폴백 if 하나, 근거 부족→사례검색·NLP
재호출 bounded for 하나)이라 LangGraph 같은 프레임워크는 과한 도구다
(CLAUDE.md §8). 각 단계는 독립 함수 + try/except 부분 실패 처리.

(pipeline.py — 아직 비어 있음.)
"""

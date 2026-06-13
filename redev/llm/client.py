"""client.py — Gemini 한 겹 래퍼 (Phase 7). 설계: nlp_layer3.md §1.

인터페이스 하나: complete(system, user) -> text. rate limit(429)·일시오류에 지수 백오프
재시도(무료 분당 한도 대비 — 실증됨). 재시도 소진 시 LLMError raise → ★호출부가 템플릿
폴백(client는 폴백을 모름, 관심사 분리). 키는 .env GEMINI_API_KEY(하드코딩 금지).
"""
from __future__ import annotations

import os
import re
import time

from redev.config import load_llm_config


class LLMError(Exception):
    """LLM 호출 실패(재시도 소진 포함). 호출부는 이걸 잡아 템플릿 폴백."""


_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY 미설정 — .env 확인")
        _CLIENT = genai.Client(api_key=key)
    return _CLIENT


def _retry_delay(msg: str):
    """서버가 알려준 retryDelay(초) 우선 사용 — 없으면 None."""
    m = re.search(r"retryDelay['\":\s]+(\d+)", msg)
    return float(m.group(1)) if m else None


def _is_transient(msg: str) -> bool:
    # 한도(429)·서버오류(5xx) + ★네트워크 끊김/타임아웃(RemoteProtocolError 등) — 일시적이라 재시도.
    return any(t in msg for t in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL",
                                  "disconnect", "RemoteProtocol", "ConnectionError", "ConnectError",
                                  "timeout", "timed out", "ReadError", "Temporary"))


def complete(system: str, user: str, *, temperature: float | None = None, cfg=None) -> str:
    """system+user → 모델 텍스트. 재시도+백오프. 실패 소진 시 LLMError.

    ★숫자·사실 창작은 호출부 프롬프트가 막는다(규칙4) — client는 전송·재시도만 책임.
    """
    from google.genai import types
    conf = (cfg or load_llm_config())["llm"]
    temp = conf["temperature"] if temperature is None else temperature
    last = None
    for attempt in range(conf["max_retries"]):
        try:
            r = _client().models.generate_content(
                model=conf["model"], contents=user,
                config=types.GenerateContentConfig(system_instruction=system, temperature=temp),
            )
            if not r.text:
                raise LLMError("빈 응답")
            return r.text
        except LLMError:
            raise
        except Exception as e:                       # SDK 예외(429 등)
            last, msg = e, str(e)
            if _is_transient(msg) and attempt < conf["max_retries"] - 1:
                delay = _retry_delay(msg) or conf["base_backoff_s"] * (2 ** attempt)
                time.sleep(min(delay, 60))           # 지수 백오프(서버 retryDelay 우선, 상한 60s)
                continue
            raise LLMError(f"LLM 호출 실패: {msg[:200]}") from e
    raise LLMError(f"재시도 {conf['max_retries']}회 소진: {str(last)[:200]}")

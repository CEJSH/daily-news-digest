from __future__ import annotations

import ast
import json
import os
import re
import time
from typing import Any

import requests

from daily_news_digest.utils import clean_text

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

_AI_UNAVAILABLE_LOGGED: set[str] = set()

if load_dotenv:
    from pathlib import Path

    _repo_root = Path(__file__).resolve().parents[3]
    load_dotenv(dotenv_path=_repo_root / ".env")

GEMINI_API_BASE = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_TIMEOUT_SEC = int(os.getenv("GEMINI_TIMEOUT_SEC", "60"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
GEMINI_RETRY_BACKOFF_SEC = float(os.getenv("GEMINI_RETRY_BACKOFF_SEC", "1.5"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1000"))
AI_EMBED_MAX_CHARS = int(os.getenv("AI_EMBED_MAX_CHARS", "1200"))


def log_ai_unavailable(reason: str) -> None:
    # AI 비활성 사유를 중복 없이 로그 출력
    if reason in _AI_UNAVAILABLE_LOGGED:
        return
    print(f"⚠️ AI 요약 비활성: {reason}")
    _AI_UNAVAILABLE_LOGGED.add(reason)


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    # Gemini REST 응답에서 텍스트만 추출
    try:
        return payload["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return ""


def _extract_gemini_embedding(payload: dict[str, Any]) -> list[float] | None:
    # Gemini embedContent 응답에서 임베딩 벡터만 추출
    try:
        embedding = payload.get("embedding") or {}
        values = embedding.get("values") or embedding.get("value")
        if isinstance(values, list):
            return values
    except Exception:
        return None
    return None


def parse_json(text: str) -> dict[str, Any] | None:
    # 문자열에서 JSON 객체를 파싱(직접 파싱 실패 시 중괄호 블록 탐색)
    if not text:
        return None
    raw = text.strip()
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).replace("```", "").strip()

    def _try_load_json(payload: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(payload)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _strip_trailing_commas(payload: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", payload)

    def _extract_json_block(payload: str) -> str | None:
        start = payload.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(payload)):
            ch = payload[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return payload[start : i + 1]
        return payload[start:] if depth > 0 else None

    parsed = _try_load_json(raw)
    if parsed is not None:
        return parsed

    candidate = _extract_json_block(raw)
    if candidate:
        parsed = _try_load_json(candidate)
        if parsed is not None:
            return parsed
        cleaned = _strip_trailing_commas(candidate)
        parsed = _try_load_json(cleaned)
        if parsed is not None:
            return parsed
        try:
            obj = ast.literal_eval(cleaned)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    # 모델이 여분 텍스트를 섞을 때 대비한 백업 파서
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    cleaned = _strip_trailing_commas(match.group(0))
    parsed = _try_load_json(cleaned)
    if parsed is not None:
        return parsed
    try:
        obj = ast.literal_eval(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def gemini_generate_json(system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
    # Gemini REST API로 JSON 응답을 요청
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        log_ai_unavailable("GEMINI_API_KEY 미설정")
        return None
    url = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL}:generateContent"
    request_payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}],
        },
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
        },
    }
    max_attempts = max(1, GEMINI_MAX_RETRIES + 1)
    last_err = ""
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(
                url,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=request_payload,
                timeout=GEMINI_TIMEOUT_SEC,
            )
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_attempts:
                time.sleep(GEMINI_RETRY_BACKOFF_SEC * (2 ** (attempt - 1)))
                continue
            log_ai_unavailable(f"Gemini 호출 실패: {last_err}")
            return None

        if not resp.ok:
            last_err = f"{resp.status_code} {resp.text}"
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < max_attempts:
                time.sleep(GEMINI_RETRY_BACKOFF_SEC * (2 ** (attempt - 1)))
                continue
            log_ai_unavailable(f"Gemini 호출 실패: {last_err}")
            return None

        try:
            data = resp.json()
        except Exception:
            last_err = "Gemini 응답 JSON 파싱 실패"
            if attempt < max_attempts:
                time.sleep(GEMINI_RETRY_BACKOFF_SEC * (2 ** (attempt - 1)))
                continue
            log_ai_unavailable(last_err)
            return None

        text = _extract_gemini_text(data)
        if not text:
            last_err = "Gemini 응답 텍스트 비어있음"
            if attempt < max_attempts:
                time.sleep(GEMINI_RETRY_BACKOFF_SEC * (2 ** (attempt - 1)))
                continue
            log_ai_unavailable(last_err)
            return None

        parsed = parse_json(text)
        if not isinstance(parsed, dict):
            last_err = "Gemini 응답 JSON 형식 아님"
            if attempt < max_attempts:
                time.sleep(GEMINI_RETRY_BACKOFF_SEC * (2 ** (attempt - 1)))
                continue
            snippet = re.sub(r"\s+", " ", text)[:160]
            truncated_hint = ""
            if text.strip().startswith("{") and not text.strip().endswith("}"):
                truncated_hint = " (truncated?)"
            log_ai_unavailable(f"{last_err}{truncated_hint}: {snippet}")
            return None

        return parsed

    if last_err:
        log_ai_unavailable(f"Gemini 호출 실패: {last_err}")
    return None


def get_embedding(text: str) -> list[float] | None:
    # 텍스트 임베딩 생성 (중복 제거용, Gemini)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        log_ai_unavailable("GEMINI_API_KEY 미설정")
        return None
    cleaned = clean_text(text or "")
    if not cleaned:
        return None
    if len(cleaned) > AI_EMBED_MAX_CHARS:
        cleaned = cleaned[:AI_EMBED_MAX_CHARS]
    try:
        resp = requests.post(
            f"{GEMINI_API_BASE}/models/{GEMINI_EMBEDDING_MODEL}:embedContent",
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json={
                "model": f"models/{GEMINI_EMBEDDING_MODEL}",
                "content": {"parts": [{"text": cleaned}]},
            },
            timeout=30,
        )
    except Exception as e:
        log_ai_unavailable(f"Gemini 임베딩 호출 실패: {e}")
        return None
    if not resp.ok:
        log_ai_unavailable(f"Gemini 임베딩 호출 실패: {resp.status_code} {resp.text}")
        return None
    try:
        data = resp.json()
    except Exception:
        log_ai_unavailable("Gemini 임베딩 응답 JSON 파싱 실패")
        return None
    embedding = _extract_gemini_embedding(data)
    if not embedding:
        log_ai_unavailable("Gemini 임베딩 응답 값 없음")
        return None
    return embedding

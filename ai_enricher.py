import json
import os
import re
from typing import Any

import requests

from utils import clean_text, ensure_three_lines, split_summary_to_3lines

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

_AI_UNAVAILABLE_LOGGED: set[str] = set()

if load_dotenv:
    load_dotenv()

GEMINI_API_BASE = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
AI_INPUT_MAX_CHARS = int(os.getenv("AI_INPUT_MAX_CHARS", "1800"))
AI_SUMMARY_MIN_CHARS = int(os.getenv("AI_SUMMARY_MIN_CHARS", "200"))
AI_EMBED_MAX_CHARS = int(os.getenv("AI_EMBED_MAX_CHARS", "1200"))

_ALLOWED_IMPACT_SIGNALS = {
    "policy",
    "budget",
    "sanctions",
    "capex",
    "earnings",
    "market-demand",
    "security",
    "infra",
}
_ALLOWED_QUALITY_TAGS = {
    "clickbait",
    "promo",
    "opinion",
    "event",
    "report",
    "entertainment",
    "crime",
    "local",
    "emotion",
}
SYSTEM_PROMPT = """You are a meticulous news editor for a daily digest.

Use ONLY the provided title and article text (use full_text if available, otherwise summary).  
Do not add any facts, context, or knowledge beyond the provided text.  
ImpactSignals are hints only; verify and adjust strictly based on the article text.  

If the article is in English, translate and write all outputs in Korean.  
Translate faithfully without adding interpretation or extra context.  
Write all Korean sentences in polite "~입니다/~합니다" style.

Topic & filtering intent (must align with these)

This digest prioritizes issues that:
1) will still matter tomorrow (structural or decision-relevant),  
2) avoid excessive emotional consumption,  
3) avoid duplication with yesterday’s news.

Primary topics of interest include:

- 실적_가이던스: corporate earnings, guidance, margins, forecasts  
- 반도체_공급망: HBM, advanced packaging, foundry, equipment, export controls, supply constraints  
- 전력_인프라: power grid, transmission, utilities, electricity pricing, nuclear/gas, data center power  
- AI_저작권_데이터권리: AI copyright, training data, licensing, privacy, data protection  
- 보안_취약점_패치: CVE, zero-day, patches, incident response, breach notifications  
- 투자_MA_IPO: funding rounds, mergers & acquisitions, IPOs, major deal terms  
- 국내_정책_규제: legislation, enforcement decrees, regulator guidance, official policy changes  

Respond ONLY in valid JSON.  
Do not include any markdown, explanations, or extra text.  
All fields are required.

Output schema:

{
  "title_ko": string,
  "summary_lines": [string, string, string],
  "why_important": string,
  "dedupe_key": string,
  "importance_score": integer,
  "impact_signals": [string],
  "category_label": string,
  "quality_label": string,
  "quality_reason": string,
  "quality_tags": [string]
}

Field rules:

- title_ko: If the title is in English, translate to natural Korean; if already Korean, keep as-is. No source/publisher names, no dates.
- summary_lines: exactly 3 short, clear Korean sentences capturing the core facts. No fluff.
- why_important: one concise Korean sentence explaining long-term significance (decision-relevant, not emotional).
- dedupe_key: 4-8 core concepts only, hyphen-separated, lowercase, alphanumeric and Korean characters only; no dates, no numbers, no stopwords, no source/publisher names.

- importance_score must follow these rules:
  5 = major structural impact (policy, regulation, major earnings, supply chain shifts)
  4 = significant industry-level impact
  3 = meaningful but limited scope
  2 = minor update
  1 = low relevance or routine news

- impact_signals: choose only from this list; include only signals explicitly supported by the text; return [] if none apply.  
  Valid impact_signals: policy, budget, sanctions, capex, earnings, market-demand, security, infra.

- category_label rules (choose exactly one):
  IT = technology, AI, semiconductors, cloud, security, digital infrastructure  
  경제 = macroeconomy, markets, finance, energy transition, corporate earnings  
  글로벌 = geopolitics, trade, sanctions, international relations  

Quality policy (strict):

Mark quality_label as "low_quality" if ANY apply (even if importance is high):

- PR or promotional content  
- opinion/editorial/column  
- event, webinar, conference, or whitepaper announcements  
- entertainment, crime, or local human-interest stories  
- emotionally manipulative or clickbait headlines  

If low_quality:

- set quality_label = "low_quality"  
- set quality_reason = short Korean phrase (e.g., "칼럼/의견", "홍보성", "이벤트 공지", "자극적 헤드라인")  
- include relevant quality_tags from:  
  clickbait, promo, opinion, event, report, entertainment, crime, local, emotion  

Otherwise:

- set quality_label = "ok"  
- set quality_reason = short Korean phrase like "정보성 기사"  
- set quality_tags = [] or only truly applicable tags.

Additional constraints:

- No source names, no dates, no clickbait language.
"""


def _log_ai_unavailable(reason: str) -> None:
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


def _parse_json(text: str) -> dict[str, Any] | None:
    # 문자열에서 JSON 객체를 파싱(직접 파싱 실패 시 중괄호 블록 탐색)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # 모델이 여분 텍스트를 섞을 때 대비한 백업 파서
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _normalize_dedupe_key(raw: str) -> str:
    # dedupe 키를 토큰 규칙에 맞게 정규화
    if not raw:
        return ""
    t = clean_text(raw).lower()
    t = re.sub(r"[^a-z0-9가-힣\s-]", " ", t)
    t = re.sub(r"[\s_]+", "-", t).strip("-")
    parts = [p for p in t.split("-") if p]
    cleaned: list[str] = []
    seen = set()
    for p in parts:
        # 중복/너무 짧은 토큰 제거
        if p in seen:
            continue
        if re.search(r"[가-힣]", p):
            if len(p) < 2:
                continue
        else:
            if len(p) < 3:
                continue
        cleaned.append(p)
        seen.add(p)
        if len(cleaned) >= 8:
            break
    return "-".join(cleaned) or t


def _fallback_why_important(impact_signals: list[str]) -> str:
    # AI가 비워둔 이유 문장을 임팩트 시그널로 보정
    if not impact_signals:
        return ""
    if "policy" in impact_signals or "sanctions" in impact_signals:
        return "정책 변화가 산업 전반의 규칙과 비용 구조에 영향을 줄 수 있습니다."
    if "capex" in impact_signals or "infra" in impact_signals:
        return "인프라 투자 변화가 공급망과 비용 구조에 직결됩니다."
    if "earnings" in impact_signals:
        return "실적 변화가 시장 기대와 밸류에이션에 영향을 줍니다."
    if "market-demand" in impact_signals:
        return "수요 흐름 변화가 기업 전략과 가격에 영향을 줄 수 있습니다."
    if "security" in impact_signals:
        return "보안 리스크는 서비스 운영과 규제 대응에 직접 영향을 줍니다."
    return "산업 전반의 방향성과 리스크에 영향을 줄 수 있습니다."


def _fallback_importance_score(impact_signals: list[str]) -> int:
    # AI가 비워둔 중요도를 임팩트 시그널로 추정
    if not impact_signals:
        return 2
    if "policy" in impact_signals or "sanctions" in impact_signals:
        return 4
    if "capex" in impact_signals or "infra" in impact_signals or "security" in impact_signals:
        return 3
    if "earnings" in impact_signals or "market-demand" in impact_signals:
        return 2
    return 2


def _normalize_importance_score(value: Any, impact_signals: list[str]) -> int:
    # 중요도 점수를 1~5로 보정하고 실패 시 fallback 사용
    try:
        score = int(value)
    except Exception:
        return _fallback_importance_score(impact_signals)
    return max(1, min(5, score))

def _pick_ai_input_text(summary_raw: str, full_text: str) -> str:
    if summary_raw and len(summary_raw) >= AI_SUMMARY_MIN_CHARS:
        return summary_raw[:AI_INPUT_MAX_CHARS]
    text = full_text or summary_raw or ""
    if len(text) > AI_INPUT_MAX_CHARS:
        text = text[:AI_INPUT_MAX_CHARS]
    return text


def _normalize_label_list(value: Any, allowed: set[str]) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        raw_list = value
    elif isinstance(value, str):
        raw_list = re.split(r"[,\s]+", value)
    else:
        return []
    cleaned: list[str] = []
    seen = set()
    for v in raw_list:
        token = clean_text(str(v)).lower()
        if not token or token not in allowed or token in seen:
            continue
        cleaned.append(token)
        seen.add(token)
    return cleaned


def _normalize_impact_signals(value: Any) -> list[str]:
    # 임팩트 시그널 값을 허용 목록 기준으로 정리
    return _normalize_label_list(value, _ALLOWED_IMPACT_SIGNALS)


def _normalize_quality_label(value: Any) -> str:
    # 품질 라벨을 ok/low_quality로 정규화
    if not value:
        return "ok"
    label = clean_text(str(value)).lower()
    if label in {"low_quality", "low", "bad", "reject"}:
        return "low_quality"
    if label in {"ok", "good", "keep"}:
        return "ok"
    return "low_quality" if "low" in label or "bad" in label else "ok"


def _normalize_quality_tags(value: Any) -> list[str]:
    # 품질 태그를 허용 목록 기준으로 정리
    return _normalize_label_list(value, _ALLOWED_QUALITY_TAGS)


def _normalize_category_label(value: Any) -> str:
    # 카테고리 라벨을 IT/경제/글로벌로 정규화
    if not value:
        return ""
    raw = clean_text(str(value)).lower()
    if raw in {"it", "tech", "technology"}:
        return "IT"
    if raw in {"경제", "economy", "economic", "macro", "finance"}:
        return "경제"
    if raw in {"글로벌", "global", "geopolitics", "world"}:
        return "글로벌"
    if "경제" in raw:
        return "경제"
    if "global" in raw or "글로벌" in raw:
        return "글로벌"
    if "it" in raw or "tech" in raw:
        return "IT"
    return ""

def _gemini_generate_json(system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
    # Gemini REST API로 JSON 응답을 요청
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        _log_ai_unavailable("GEMINI_API_KEY 미설정")
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
            "maxOutputTokens": 350,
            "responseMimeType": "application/json",
        },
    }
    try:
        resp = requests.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=request_payload,
            timeout=30,
        )
    except Exception as e:
        _log_ai_unavailable(f"Gemini 호출 실패: {e}")
        return None
    if not resp.ok:
        _log_ai_unavailable(f"Gemini 호출 실패: {resp.status_code} {resp.text}")
        return None
    try:
        data = resp.json()
    except Exception:
        _log_ai_unavailable("Gemini 응답 JSON 파싱 실패")
        return None
    text = _extract_gemini_text(data)
    if not text:
        _log_ai_unavailable("Gemini 응답 텍스트 비어있음")
        return None
    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        _log_ai_unavailable("Gemini 응답 JSON 형식 아님")
        return None
    return parsed

def enrich_item_with_ai(item: dict) -> dict:
    # 기사 아이템을 AI로 요약/분류/중요도 평가
    title = clean_text(item.get("title") or "")
    summary_raw = clean_text(item.get("summaryRaw") or item.get("summary") or "")
    full_text = clean_text(item.get("fullText") or "")
    if full_text and len(full_text) > 6000:
        full_text = full_text[:6000]
    source = clean_text(item.get("source") or "")
    published = clean_text(item.get("published") or "")
    impact_signals = item.get("impactSignals") or []

    # 모델 입력 구성
    input_text = _pick_ai_input_text(summary_raw, full_text)
    user_prompt = (
        f"Title: {title}\n"
        f"ImpactSignals: {', '.join(impact_signals)}\n"
        f"Text: {input_text}\n"
        "Return only JSON."
    )

    # Gemini만 사용 (OpenAI 폴백 없음)
    payload = _gemini_generate_json(SYSTEM_PROMPT, user_prompt)
    if not isinstance(payload, dict):
        return {}

    # 요약/중요도/라벨 결과를 정규화
    title_ko = clean_text(payload.get("title_ko") or "")
    if not title_ko:
        title_ko = title
    summary_lines = ensure_three_lines(payload.get("summary_lines") or [], full_text or summary_raw)
    why_important = clean_text(payload.get("why_important") or "")
    if not why_important:
        why_important = _fallback_why_important(impact_signals)

    dedupe_key = _normalize_dedupe_key(payload.get("dedupe_key") or "")
    if not dedupe_key:
        dedupe_key = _normalize_dedupe_key(item.get("dedupeKey") or title or summary_raw)

    impact_signals_ai = _normalize_impact_signals(payload.get("impact_signals"))
    importance_score = _normalize_importance_score(
        payload.get("importance_score") or payload.get("importance"),
        impact_signals,
    )
    quality_label = _normalize_quality_label(payload.get("quality_label") or payload.get("quality"))
    quality_reason = clean_text(payload.get("quality_reason") or "")
    quality_tags = _normalize_quality_tags(payload.get("quality_tags"))
    category_label = _normalize_category_label(payload.get("category_label") or payload.get("category"))

    return {
        "title_ko": title_ko,
        "summary_lines": summary_lines,
        "why_important": why_important,
        "dedupe_key": dedupe_key,
        "importance_score": importance_score,
        "impact_signals": impact_signals_ai,
        "quality_label": quality_label,
        "quality_reason": quality_reason,
        "quality_tags": quality_tags,
        "category_label": category_label,
    }


def get_embedding(text: str) -> list[float] | None:
    # 텍스트 임베딩 생성 (중복 제거용, Gemini)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        _log_ai_unavailable("GEMINI_API_KEY 미설정")
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
        _log_ai_unavailable(f"Gemini 임베딩 호출 실패: {e}")
        return None
    if not resp.ok:
        _log_ai_unavailable(f"Gemini 임베딩 호출 실패: {resp.status_code} {resp.text}")
        return None
    try:
        data = resp.json()
    except Exception:
        _log_ai_unavailable("Gemini 임베딩 응답 JSON 파싱 실패")
        return None
    embedding = _extract_gemini_embedding(data)
    if not embedding:
        _log_ai_unavailable("Gemini 임베딩 응답 값 없음")
        return None
    return embedding

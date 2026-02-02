import json
import os
import re
from typing import Any

from utils import clean_text, ensure_three_lines, split_summary_to_3lines

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - runtime dependency
    OpenAI = None

_CLIENT = None
_AI_UNAVAILABLE_LOGGED: set[str] = set()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

SYSTEM_PROMPT = """You are a meticulous news editor for a daily digest.
Use ONLY the provided title and article text (use full_text if available, otherwise summary).  
Do not add any facts, context, or knowledge beyond the provided text.
ImpactSignals are hints only; verify and adjust strictly based on the article text.
If the article is in English, translate and write all outputs in Korean.  
Translate faithfully without adding interpretation or extra context.

Respond ONLY in valid JSON.  
Do not include any markdown, explanations, or extra text.  
All fields are required.

Output schema:

{
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

- summary_lines: exactly 3 short, clear Korean sentences capturing the core facts.
- why_important: one concise Korean sentence explaining long-term significance.
- dedupe_key: 4-8 core concepts only, hyphen-separated, lowercase, alphanumeric and Korean characters only; no dates, no numbers, no stopwords, no source names.
- importance_score must follow these rules:
  5 = major structural impact (policy, regulation, major earnings, supply chain shifts)
  4 = significant industry-level impact
  3 = meaningful but limited scope
  2 = minor update
  1 = low relevance or routine news
- impact_signals: choose only from the allowed list below; include only signals explicitly supported by the text; return an empty array if none apply.
- category_label rules:
  IT = technology, AI, semiconductors, cloud, security, digital infrastructure
  경제 = macroeconomy, markets, finance, energy transition, corporate earnings
  글로벌 = geopolitics, trade, sanctions, international relations

Allowed values:

Valid impact_signals: policy, budget, sanctions, capex, earnings, market-demand, security, infra.
Valid quality_tags: clickbait, promo, opinion, event, report, entertainment, crime, local, emotion.
Low quality criteria:
Mark quality_label as "low_quality" for any of the following:
- PR or promotional content
- opinion/editorial/column
- event, webinar, conference, or whitepaper announcements
- entertainment, crime, or local human-interest stories
- emotionally manipulative or clickbait headlines

If any low_quality condition applies, quality_label must be "low_quality" regardless of importance.
No source names, no dates, no clickbait language.
"""


def _get_client() -> Any:
    global _CLIENT
    if OpenAI is None:
        return None
    if _CLIENT is None:
        _CLIENT = OpenAI()
    return _CLIENT


def _log_ai_unavailable(reason: str) -> None:
    if reason in _AI_UNAVAILABLE_LOGGED:
        return
    print(f"⚠️ AI 요약 비활성: {reason}")
    _AI_UNAVAILABLE_LOGGED.add(reason)


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = getattr(response, "output", None) or []
    texts: list[str] = []
    for item in output:
        item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
        if item_type != "message":
            continue
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None) or []
        for part in content:
            part_type = getattr(part, "type", None) or (part.get("type") if isinstance(part, dict) else None)
            if part_type == "output_text":
                text = getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else "")
                if text:
                    texts.append(text)
    return "\n".join(texts).strip()


def _parse_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _normalize_dedupe_key(raw: str) -> str:
    if not raw:
        return ""
    t = clean_text(raw).lower()
    t = re.sub(r"[^a-z0-9가-힣\s-]", " ", t)
    t = re.sub(r"[\s_]+", "-", t).strip("-")
    parts = [p for p in t.split("-") if p]
    cleaned: list[str] = []
    seen = set()
    for p in parts:
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
    try:
        score = int(value)
    except Exception:
        return _fallback_importance_score(impact_signals)
    return max(1, min(5, score))


def _normalize_impact_signals(value: Any) -> list[str]:
    allowed = {"policy", "budget", "sanctions", "capex", "earnings", "market-demand", "security", "infra"}
    if not value:
        return []
    raw_list = []
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


def _normalize_quality_label(value: Any) -> str:
    if not value:
        return "ok"
    label = clean_text(str(value)).lower()
    if label in {"low_quality", "low", "bad", "reject"}:
        return "low_quality"
    if label in {"ok", "good", "keep"}:
        return "ok"
    return "low_quality" if "low" in label or "bad" in label else "ok"


def _normalize_quality_tags(value: Any) -> list[str]:
    allowed = {"clickbait", "promo", "opinion", "event", "report", "entertainment", "crime", "local", "emotion"}
    if not value:
        return []
    raw_list = []
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


def _normalize_category_label(value: Any) -> str:
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


def enrich_item_with_ai(item: dict) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        _log_ai_unavailable("OPENAI_API_KEY 미설정")
        return {}
    client = _get_client()
    if client is None:
        _log_ai_unavailable("openai 패키지 미설치 또는 초기화 실패")
        return {}

    title = clean_text(item.get("title") or "")
    summary_raw = clean_text(item.get("summaryRaw") or item.get("summary") or "")
    full_text = clean_text(item.get("fullText") or "")
    if full_text and len(full_text) > 6000:
        full_text = full_text[:6000]
    source = clean_text(item.get("source") or "")
    published = clean_text(item.get("published") or "")
    impact_signals = item.get("impactSignals") or []

    user_prompt = (
        f"Title: {title}\n"
        f"ImpactSignals: {', '.join(impact_signals)}\n"
        f"Article: {full_text or summary_raw}\n"
        "Return only JSON."
    )

    try:
        response = client.responses.create(
            model=DEFAULT_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            text={"format": {"type": "json_object"}},
            temperature=0.2,
            max_output_tokens=350,
        )
    except Exception:
        return {}

    payload = _parse_json(_extract_output_text(response))
    if not isinstance(payload, dict):
        return {}

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
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    client = _get_client()
    if client is None:
        return None
    cleaned = clean_text(text or "")
    if not cleaned:
        return None
    if len(cleaned) > 2000:
        cleaned = cleaned[:2000]
    try:
        response = client.embeddings.create(
            model=DEFAULT_EMBEDDING_MODEL,
            input=cleaned,
        )
    except Exception:
        return None
    data = getattr(response, "data", None) or []
    if not data:
        return None
    first = data[0]
    embedding = getattr(first, "embedding", None) or (first.get("embedding") if isinstance(first, dict) else None)
    if not isinstance(embedding, list):
        return None
    return embedding

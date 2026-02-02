import json
import os
import re
from typing import Any

from utils import clean_text, split_summary_to_3lines

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - runtime dependency
    OpenAI = None

_CLIENT = None

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

SYSTEM_PROMPT = (
    "You are a meticulous news editor for a daily digest. "
    "Use ONLY the provided title and summary. Do not add new facts. "
    "Respond as JSON with keys: summary_lines (array of 3 short sentences), "
    "why_important (one sentence), dedupe_key (4-8 keywords, hyphen-separated, lowercase), "
    "importance_score (integer 1-5), impact_signals (array), category_label (one of IT, 경제, 글로벌). "
    "quality_label (ok or low_quality), quality_reason (short phrase), quality_tags (array). "
    "Valid impact_signals labels: policy, budget, sanctions, capex, earnings, market-demand, security, infra. "
    "Valid quality_tags: clickbait, promo, opinion, event, report, entertainment, crime, local, emotion. "
    "Mark low_quality for PR/promotions, opinion/editorial, event/webinar/whitepaper, "
    "entertainment/crime/local human-interest, or emotionally manipulative headlines. "
    "No source names, no dates, no clickbait."
)


def _get_client() -> Any:
    global _CLIENT
    if OpenAI is None:
        return None
    if _CLIENT is None:
        _CLIENT = OpenAI()
    return _CLIENT


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


def _ensure_three_lines(lines: list[str], fallback_text: str) -> list[str]:
    cleaned = [clean_text(x) for x in (lines or []) if clean_text(x)]
    if len(cleaned) >= 3:
        return cleaned[:3]

    fallback = split_summary_to_3lines(fallback_text)
    for line in fallback:
        line = clean_text(line)
        if line and line not in cleaned:
            cleaned.append(line)
        if len(cleaned) >= 3:
            break

    if len(cleaned) < 3 and fallback_text:
        s = clean_text(fallback_text)
        if s:
            step = max(40, len(s) // 3)
            chunks = [s[i:i + step].strip() for i in range(0, len(s), step)]
            for c in chunks:
                if c and c not in cleaned:
                    cleaned.append(c)
                if len(cleaned) >= 3:
                    break

    return cleaned[:3]


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
        return {}
    client = _get_client()
    if client is None:
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
        f"Source: {source}\n"
        f"Published: {published}\n"
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

    summary_lines = _ensure_three_lines(payload.get("summary_lines") or [], summary_raw)
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

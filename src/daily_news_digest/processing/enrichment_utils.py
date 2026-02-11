from __future__ import annotations

import re
from typing import Any

from daily_news_digest.core.constants import (
    ALLOWED_IMPACT_SIGNALS,
    DEDUPE_NOISE_WORDS,
    DEDUPE_EVENT_GROUPS,
    IMPACT_SIGNALS_MAP,
    MARKET_DEMAND_EVIDENCE_KEYWORDS,
    SANCTIONS_EVIDENCE_KEYWORDS,
    SANCTIONS_KEYWORDS,
    SECURITY_EVIDENCE_KEYWORDS,
    STOPWORDS,
    TRADE_TARIFF_KEYWORDS,
)
from daily_news_digest.utils import clean_text, split_summary_to_lines

_ALLOWED_IMPACT_SIGNALS = set(ALLOWED_IMPACT_SIGNALS)
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

_GENERIC_SUMMARY_PHRASES = [
    "자세한 내용은",
    "구체적인 내용은",
    "아직 알려지지",
    "추가 정보는",
    "확인되지",
    "details are not",
    "details were not",
    "not disclosed",
    "not yet known",
]

_GENERIC_WHY_PHRASES = [
    "영향을 줄 수 있습니다",
    "기여할 수 있습니다",
    "강화할 수 있습니다",
    "도움이 될 수 있습니다",
    "의미가 있습니다",
    "주목됩니다",
    "변화를 가져올 수 있습니다",
]

_IMPACT_KEYWORDS = [
    "영향", "파장", "리스크", "비용", "수요", "공급", "공급망", "가격", "시장", "규제", "정책", "투자", "실적",
    "impact", "risk", "cost", "demand", "supply", "price", "market", "regulation", "policy",
]

_LOCATION_KEYWORDS = [
    "미국", "중국", "eu", "유럽", "한국", "일본", "영국", "독일", "프랑스", "인도", "브라질", "러시아",
    "중동", "아시아", "서울", "뉴욕", "워싱턴", "캘리포니아", "u.s.", "us", "china", "europe", "korea",
    "japan", "uk", "germany", "france", "india",
]

_QUALITY_TAG_REASON = {
    "promo": "홍보성",
    "local": "지역/생활",
    "emotion": "감정 과잉",
    "entertainment": "연예/오락",
    "crime": "사건/범죄",
    "opinion": "칼럼/의견",
    "event": "이벤트 공지",
    "report": "리포트/보고서",
    "clickbait": "자극적 헤드라인",
}


def _is_date_like_token(token: str) -> bool:
    if not token:
        return False
    if token.isdigit():
        if len(token) == 4:
            try:
                year = int(token)
            except Exception:
                return True
            return 1900 <= year <= 2100
        return False
    if re.match(r"^\d{1,4}(년|월|일|분기)$", token):
        return True
    if re.match(r"^(q[1-4]|[1-4]q)$", token):
        return True
    if re.match(r"^\d{1,2}분기$", token):
        return True
    return False


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
    event_labels = set(DEDUPE_EVENT_GROUPS.keys())
    for p in parts:
        # 중복/너무 짧은 토큰 제거
        if p in seen:
            continue
        if p in event_labels:
            continue
        if p in STOPWORDS or p in DEDUPE_NOISE_WORDS:
            continue
        if _is_date_like_token(p):
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
    if len(cleaned) < 4:
        # 부족하면 노이즈 필터를 완화해 보강
        for p in parts:
            if p in seen:
                continue
            if _is_date_like_token(p):
                continue
            if re.search(r"[가-힣]", p):
                if len(p) < 2:
                    continue
            else:
                if len(p) < 3:
                    continue
            cleaned.append(p)
            seen.add(p)
            if len(cleaned) >= 4:
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


def _normalize_label_list(value: Any, allowed: set[str]) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        raw_iter = (str(v) for v in value)
    elif isinstance(value, str):
        raw_iter = re.split(r"[,\s]+", value)
    else:
        return []
    tokens = (
        token
        for token in (clean_text(v).lower() for v in raw_iter)
        if token and token in allowed
    )
    return list(dict.fromkeys(tokens))


def _rule_based_impact_signals(text: str) -> list[str]:
    text_lower = clean_text(text).lower()
    signals: list[str] = []
    for signal, keywords in IMPACT_SIGNALS_MAP.items():
        if signal == "sanctions":
            continue
        if any(kw.lower() in text_lower for kw in keywords):
            signals.append(signal)

    if any(kw.lower() in text_lower for kw in SANCTIONS_KEYWORDS):
        signals.append("sanctions")

    if any(kw.lower() in text_lower for kw in TRADE_TARIFF_KEYWORDS):
        if "policy" not in signals:
            signals.append("policy")

    seen = set()
    ordered: list[str] = []
    for s in signals:
        if s in _ALLOWED_IMPACT_SIGNALS and s not in seen:
            ordered.append(s)
            seen.add(s)
    priority = [
        "policy",
        "sanctions",
        "capex",
        "infra",
        "security",
        "earnings",
        "market-demand",
    ]
    ordered = [s for s in priority if s in ordered]
    return ordered


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = [
        p.strip()
        for p in re.split(r"(?<=[\.\!\?。])\s+|(?<=다\.)\s+", text)
        if p.strip()
    ]
    return parts if parts else [text.strip()]


def _normalize_impact_signal_objects(value: Any) -> list[dict[str, str]]:
    if not value:
        return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    cleaned: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        label = clean_text(str(item.get("label") or "")).lower()
        evidence = clean_text(str(item.get("evidence") or ""))
        if not label or not evidence:
            continue
        cleaned.append({"label": label, "evidence": evidence})
    return cleaned


def _normalize_evidence_key(text: str) -> str:
    t = clean_text(text or "").lower()
    if not t:
        return ""
    t = re.sub(r"[^a-z0-9가-힣]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _evidence_has_trigger(label: str, evidence: str) -> bool:
    if not label or not evidence:
        return False
    text = evidence.lower()
    if label not in _ALLOWED_IMPACT_SIGNALS:
        return False
    if label == "sanctions":
        return any(kw.lower() in text for kw in SANCTIONS_KEYWORDS)
    if label == "market-demand":
        return any(kw.lower() in text for kw in MARKET_DEMAND_EVIDENCE_KEYWORDS)
    keywords = IMPACT_SIGNALS_MAP.get(label, [])
    if any(kw.lower() in text for kw in keywords):
        return True
    if label == "policy":
        return any(kw.lower() in text for kw in TRADE_TARIFF_KEYWORDS)
    return False


def _pick_evidence_by_keywords(
    items: list[dict[str, str]],
    full_lower: str,
    keywords: list[str],
) -> str:
    if not items or not keywords or not full_lower:
        return ""
    for item in items:
        evidence = item.get("evidence") or ""
        if not evidence:
            continue
        evidence_clean = clean_text(evidence).lower()
        if not evidence_clean:
            continue
        if evidence_clean not in full_lower:
            continue
        if any(kw.lower() in evidence_clean for kw in keywords):
            return evidence
    return ""


def _filter_impact_signal_objects(
    items: list[dict[str, str]],
    candidates: list[str],
    full_text: str,
) -> tuple[list[str], dict[str, str]]:
    full_lower = clean_text(full_text or "").lower()
    kept: list[dict[str, str]] = []
    seen_evidence: set[str] = set()
    for item in items:
        label = item.get("label") or ""
        evidence = item.get("evidence") or ""
        if label not in candidates or label not in _ALLOWED_IMPACT_SIGNALS:
            continue
        if not evidence:
            continue
        if clean_text(evidence).lower() not in full_lower:
            continue
        if not _evidence_has_trigger(label, evidence):
            continue
        evidence_key = _normalize_evidence_key(evidence)
        if not evidence_key or evidence_key in seen_evidence:
            continue
        seen_evidence.add(evidence_key)
        kept.append(item)

    sanctions_evidence = _pick_evidence_by_keywords(
        items,
        full_lower,
        SANCTIONS_EVIDENCE_KEYWORDS,
    )
    security_evidence = _pick_evidence_by_keywords(
        items,
        full_lower,
        SECURITY_EVIDENCE_KEYWORDS,
    )
    if sanctions_evidence and not any(k.get("label") == "sanctions" for k in kept):
        evidence_key = _normalize_evidence_key(sanctions_evidence)
        if evidence_key and evidence_key not in seen_evidence:
            kept.append({"label": "sanctions", "evidence": sanctions_evidence})
            seen_evidence.add(evidence_key)
    if security_evidence and not any(k.get("label") == "security" for k in kept):
        evidence_key = _normalize_evidence_key(security_evidence)
        if evidence_key and evidence_key not in seen_evidence:
            kept.append({"label": "security", "evidence": security_evidence})
            seen_evidence.add(evidence_key)

    deduped: list[dict[str, str]] = []
    seen = set()
    for item in kept:
        label = item.get("label") or ""
        if not label or label in seen:
            continue
        deduped.append(item)
        seen.add(label)

    priority = [
        "policy",
        "sanctions",
        "capex",
        "infra",
        "security",
        "earnings",
        "market-demand",
    ]
    deduped = [k for k in deduped if k["label"] in priority]
    deduped.sort(key=lambda x: priority.index(x["label"]))

    kept = deduped[:2]
    if sanctions_evidence and not any(k["label"] == "sanctions" for k in kept):
        keep_primary = [k for k in deduped if k["label"] != "sanctions"][:1]
        kept = keep_primary + [{"label": "sanctions", "evidence": sanctions_evidence}]
        kept.sort(key=lambda x: priority.index(x["label"]))

    labels = [k["label"] for k in kept]
    evidence_map = {k["label"]: k["evidence"] for k in kept}
    return labels, evidence_map


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
    # 카테고리 라벨을 확장 체계로 정규화
    if not value:
        return ""
    raw = clean_text(str(value)).lower()
    if raw in {"경제", "economy", "economic", "macro", "macroeconomy"}:
        return "경제"
    if raw in {"산업", "industry", "industrial", "manufacturing", "supply chain"}:
        return "산업"
    if raw in {"기술", "tech", "technology", "ai", "software", "semiconductor", "it"}:
        return "기술"
    if raw in {"금융", "finance", "financial", "markets", "market", "capital"}:
        return "금융"
    if raw in {"정책", "policy", "regulation", "legislation", "trade policy"}:
        return "정책"
    if raw in {"국제", "글로벌", "global", "geopolitics", "world", "international", "diplomacy"}:
        return "국제"
    if raw in {"사회", "social", "society", "public"}:
        return "사회"
    if raw in {"라이프", "lifestyle", "consumer", "retail"}:
        return "라이프"
    if raw in {"헬스", "health", "healthcare", "medical", "biotech"}:
        return "헬스"
    if raw in {"환경", "environment", "climate", "sustainability", "esg"}:
        return "환경"
    if raw in {"에너지", "energy", "power", "utility", "oil", "gas"}:
        return "에너지"
    if raw in {"모빌리티", "mobility", "transport", "automotive", "ev", "autonomous"}:
        return "모빌리티"

    if "경제" in raw or "inflation" in raw or "gdp" in raw or "macro" in raw:
        return "경제"
    if "산업" in raw or "supply chain" in raw or "manufactur" in raw:
        return "산업"
    if "기술" in raw or "tech" in raw or "ai" in raw or "software" in raw or "cloud" in raw or raw.startswith("it"):
        return "기술"
    if "금융" in raw or "finance" in raw or "market" in raw or "earnings" in raw:
        return "금융"
    if "정책" in raw or "규제" in raw or "입법" in raw or "통상" in raw:
        return "정책"
    if "global" in raw or "국제" in raw or "글로벌" in raw or "geopolitic" in raw or "diplomac" in raw:
        return "국제"
    if "사회" in raw or "social" in raw or "labor" in raw:
        return "사회"
    if "라이프" in raw or "lifestyle" in raw or "consumer" in raw:
        return "라이프"
    if "헬스" in raw or "health" in raw or "medical" in raw or "biotech" in raw:
        return "헬스"
    if "환경" in raw or "environment" in raw or "climate" in raw:
        return "환경"
    if "에너지" in raw or "energy" in raw or "power" in raw or "oil" in raw or "gas" in raw:
        return "에너지"
    if "모빌리티" in raw or "mobility" in raw or "transport" in raw or "automotive" in raw:
        return "모빌리티"
    return ""


def _tokenize_basic(text: str) -> list[str]:
    t = clean_text(text or "").lower()
    t = re.sub(r"[^a-z0-9가-힣\s]", " ", t)
    return [x for x in t.split() if x]


def _token_jaccard(a: str, b: str) -> float:
    toks_a = set(_tokenize_basic(a))
    toks_b = set(_tokenize_basic(b))
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def _looks_like_sentence(line: str) -> bool:
    if not line:
        return False
    clean_line = clean_text(line)
    if not clean_line:
        return False
    if clean_line[-1] in ".?!":
        return True
    return bool(re.search(r"(다|니다|합니다|되었습니다|됐습니다|입니다|됩니다)\.?$", clean_line))


def _has_number(text: str) -> bool:
    t = clean_text(text or "").lower()
    if re.search(r"\d", t):
        return True
    return any(unit in t for unit in ["억", "조", "만", "%", "달러", "원", "billion", "million", "trillion", "usd", "$"])


def _has_location(text: str) -> bool:
    t = clean_text(text or "").lower()
    return any(loc in t for loc in _LOCATION_KEYWORDS)


def _has_impact_keyword(text: str) -> bool:
    t = clean_text(text or "").lower()
    return any(k in t for k in _IMPACT_KEYWORDS)


def _has_entity_overlap(why: str, title: str, summary_lines: list[str]) -> bool:
    why_tokens = set(_tokenize_basic(why))
    context_tokens = set(_tokenize_basic(title)) | set(_tokenize_basic(" ".join(summary_lines)))
    if not why_tokens or not context_tokens:
        return False

    def _valid(tok: str) -> bool:
        if re.search(r"[가-힣]", tok):
            return len(tok) >= 2
        return len(tok) >= 3

    why_tokens = {t for t in why_tokens if _valid(t)}
    context_tokens = {t for t in context_tokens if _valid(t)}
    return len(why_tokens & context_tokens) >= 2


def _contains_any(text: str, phrases: list[str]) -> bool:
    if not text:
        return False
    t = clean_text(text or "").lower()
    return any(p.lower() in t for p in phrases)


def _is_line_similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a_clean = clean_text(a)
    b_clean = clean_text(b)
    if not a_clean or not b_clean:
        return False
    if a_clean in b_clean or b_clean in a_clean:
        return True
    return _token_jaccard(a_clean, b_clean) >= 0.8


def _is_bad_summary_line(line: str, title: str) -> bool:
    if not line:
        return True
    clean_line = clean_text(line)
    if not clean_line:
        return True
    if _contains_any(clean_line, _GENERIC_SUMMARY_PHRASES):
        return True
    if _is_line_similar(clean_line, title):
        return True
    if "…" in clean_line or "..." in clean_line:
        return True
    if len(clean_line) < 12 and not _has_number(clean_line):
        return True
    if not _looks_like_sentence(clean_line):
        return True
    return False


def _normalize_summary_lines(lines: list[str], title: str, fallback_text: str) -> list[str]:
    cleaned: list[str] = []
    for raw in lines or []:
        line = clean_text(raw)
        if not line:
            continue
        if _is_bad_summary_line(line, title):
            continue
        if any(_is_line_similar(line, prev) for prev in cleaned):
            continue
        cleaned.append(line)
        if len(cleaned) >= 3:
            return cleaned[:3]

    fallback_candidates = split_summary_to_lines(clean_text(fallback_text or ""), max_lines=3)
    for raw in fallback_candidates:
        line = clean_text(raw)
        if not line:
            continue
        if _is_bad_summary_line(line, title):
            continue
        if any(_is_line_similar(line, prev) for prev in cleaned):
            continue
        cleaned.append(line)
        if len(cleaned) >= 3:
            return cleaned[:3]

    if not cleaned and fallback_candidates:
        first = clean_text(fallback_candidates[0])
        if first:
            return [first]

    return cleaned[:3]


def _summary_redundant_or_title_like(lines: list[str], title: str) -> bool:
    if not lines:
        return True
    normalized = [clean_text(x) for x in lines if clean_text(x)]
    if not normalized:
        return True
    if len(set(normalized)) < len(normalized):
        return True
    if any(_is_line_similar(line, title) for line in normalized):
        return True
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            if _is_line_similar(normalized[i], normalized[j]):
                return True
    return False


def _is_low_info_summary(lines: list[str]) -> bool:
    if not lines:
        return True
    cleaned_lines = [clean_text(x) for x in lines if clean_text(x)]
    if not cleaned_lines:
        return True
    summary_text = " ".join(cleaned_lines)
    if _contains_any(summary_text, _GENERIC_SUMMARY_PHRASES):
        return True
    if len(set(cleaned_lines)) < len(cleaned_lines):
        return True
    if len(cleaned_lines) == 1:
        return len(summary_text) < 80
    if len(summary_text) < 80:
        return True
    return False


def _is_low_info_why(why: str, title: str, summary_lines: list[str]) -> bool:
    why_clean = clean_text(why or "")
    if not why_clean or len(why_clean) < 20:
        return True
    signals = 0
    if _has_entity_overlap(why_clean, title, summary_lines):
        signals += 1
    if _has_number(why_clean):
        signals += 1
    if _has_location(why_clean):
        signals += 1
    if _has_impact_keyword(why_clean):
        signals += 1
    if signals < 2:
        return True
    if _contains_any(why_clean, _GENERIC_WHY_PHRASES) and signals < 3:
        return True
    return False


def _apply_quality_guardrails(
    *,
    quality_label: str,
    quality_reason: str,
    quality_tags: list[str],
    title: str,
    summary_lines: list[str],
    why_important: str,
) -> tuple[str, str, list[str]]:
    label = quality_label or "ok"
    reason = quality_reason or ""

    if label != "low_quality":
        for tag in quality_tags:
            if tag in _QUALITY_TAG_REASON:
                return "low_quality", _QUALITY_TAG_REASON[tag], quality_tags
        if _summary_redundant_or_title_like(summary_lines, title):
            return "low_quality", "요약 중복", quality_tags
        if _is_low_info_summary(summary_lines) or _is_low_info_why(why_important, title, summary_lines):
            return "low_quality", "정보 부족", quality_tags

    if label == "low_quality" and not reason:
        for tag in quality_tags:
            if tag in _QUALITY_TAG_REASON:
                return label, _QUALITY_TAG_REASON[tag], quality_tags
        return label, "정보 부족", quality_tags

    if label == "ok" and not reason:
        reason = "정보성 기사"
    return label, reason, quality_tags

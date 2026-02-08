from __future__ import annotations

import datetime
import email.utils
import json
import os
import re
from typing import Any

from daily_news_digest.processing.ai_enricher import enrich_item_with_ai
from daily_news_digest.processing.scoring import map_topic_to_category
from daily_news_digest.core.config import (
    DEDUPE_HISTORY_PATH,
    DEDUPE_RECENT_DAYS,
    LOW_QUALITY_DOWNGRADE_MAX_IMPORTANCE,
    LOW_QUALITY_DOWNGRADE_RATIONALE,
    LOW_QUALITY_POLICY,
    MIN_TOP_ITEMS,
    TOP_LIMIT,
)
from daily_news_digest.core.constants import (
    ALLOWED_IMPACT_SIGNALS,
    IMPACT_SIGNAL_BASE_LEVELS,
    IMPACT_SIGNAL_LONG_TRIGGERS,
    IMPACT_SIGNALS_MAP,
    MARKET_DEMAND_EVIDENCE_KEYWORDS,
    SANCTIONS_EVIDENCE_KEYWORDS,
    SECURITY_EVIDENCE_KEYWORDS,
    SOURCE_TIER_A,
    SOURCE_TIER_B,
    normalize_source_name,
)
from daily_news_digest.models import DailyDigest
from daily_news_digest.utils import (
    clean_text,
    contains_binary,
    ensure_lines_1_to_3,
    estimate_read_time_seconds,
    jaccard_tokens,
    strip_summary_boilerplate,
)

_ALLOWED_IMPACT_LABELS = set(ALLOWED_IMPACT_SIGNALS)
_IMPACT_LABEL_PRIORITY = ["sanctions", "policy", "security", "capex", "infra", "earnings", "market-demand"]

_SIMPLE_INCIDENT_KEYWORDS = [
    "사고", "화재", "폭발", "추락", "충돌", "교통사고", "정전", "붕괴", "침몰",
    "사망", "부상", "실종", "희생", "피해",
    "범죄", "사건", "살인", "폭행", "강간", "납치", "강도", "테러",
    "체포", "구속", "수사", "기소", "재판", "판결", "징역",
    "부고", "별세", "추모", "장례",
    "인사", "임명", "선임", "취임", "사임", "퇴임", "승진",
    "accident", "crash", "fire", "explosion", "collapse", "sinking",
    "death", "killed", "injured", "missing", "victim",
    "crime", "murder", "assault", "rape", "kidnapping", "robbery", "terror",
    "arrest", "detention", "investigation", "indictment", "trial", "sentence",
    "obituary", "died", "appointed", "resigned", "ceo", "chairman",
]

_POLICY_ACTION_KEYWORDS = [
    "법안", "법률", "규제", "행정명령", "법 개정", "법개정", "정부 요구", "정책 발표",
    "통과", "의결", "시행", "발효", "공포", "가이드라인", "지침",
    "관세", "비관세", "무역 장벽", "trade barrier", "tariff", "non-tariff",
    "policy announcement", "official policy",
]
_POLICY_NEGOTIATION_KEYWORDS = ["협상", "협의", "협정", "회담", "대화", "negotiation", "talks", "summit", "dialogue"]
_POLICY_GOV_KEYWORDS = ["정부", "외교", "국가", "당국", "diplomatic", "government", "state"]
_POLICY_TRADE_ONLY_KEYWORDS = [
    "협상", "협의", "협정", "회담", "대화",
    "관세", "무역", "무역전쟁", "trade", "tariff", "trade talks", "negotiation", "agreement", "summit", "dialogue",
]
_POLICY_STRONG_KEYWORDS = [
    "법안", "법률", "규제", "행정명령", "법 개정", "법개정", "정책 발표",
    "통과", "의결", "시행", "발효", "공포", "가이드라인", "지침", "인허가", "과징금", "감독",
    "policy announcement", "official policy", "regulation", "rule", "guideline", "law", "bill",
]

_SANCTIONS_REQUIRED_KEYWORDS = [
    "제재", "제재 발표", "제재 부과", "제재 확대", "자산 동결", "자산동결", "거래 금지", "거래금지",
    "블랙리스트", "수출통제", "수출 금지", "2차 제재",
    "sanction", "sanctions", "asset freeze", "assets frozen", "export ban", "secondary sanctions",
    "entity list", "export control",
]

_MARKET_VARIABLE_KEYWORDS = [
    "가격", "유가", "환율", "금리", "주가",
    "수요", "주문", "판매", "재고", "출하", "생산", "생산량",
    "price", "oil", "exchange rate", "fx", "interest rate", "stock",
    "demand", "orders", "sales", "inventory", "shipments", "deliveries", "production", "output",
]
_MARKET_CHANGE_KEYWORDS = [
    "상승", "하락", "급등", "급락", "증가", "감소", "확대", "축소", "줄", "늘",
    "rise", "fall", "surge", "plunge", "increase", "decrease", "drop", "decline", "gain", "slump",
]

_EARNINGS_METRIC_KEYWORDS = [
    "매출", "영업이익", "영업익", "순이익", "순손실", "실적",
    "revenue", "operating profit", "operating income", "net income", "net profit", "earnings", "ebit", "ebitda",
]

_CAPEX_ACTION_KEYWORDS = [
    "설비투자", "투자", "투자 계획", "투자계획", "투자 발표",
    "증설", "라인", "공장", "데이터센터", "시설", "건설", "착공",
    "capex", "expansion", "build", "construction", "plant", "factory", "data center",
]
_CAPEX_PLAN_KEYWORDS = [
    "계획", "발표", "착공", "건설", "설립", "확대", "증설", "추진", "예정",
    "plan", "announce", "start", "begin", "expand",
]

_INFRA_KEYWORDS = [
    "장애", "정전", "서비스 중단", "중단", "복구", "전력망", "망 장애", "통신 장애",
    "outage", "downtime", "disruption", "service disruption", "power grid", "network outage",
]

_SECURITY_INCIDENT_KEYWORDS = [
    "무력", "충돌", "공격", "격추", "군사", "군사 행동", "미사일", "드론", "폭격", "교전", "전투",
    "침해", "해킹", "랜섬웨어", "유출", "취약점", "보안 패치", "보안취약점", "제로데이", "cve",
    "해협 봉쇄", "해협봉쇄", "유조선",
    "attack", "strike", "shoot down", "intercept", "missile", "drone", "military", "conflict", "clash",
    "breach", "hack", "ransomware", "vulnerability", "zero-day", "patch", "cve", "blockade", "tanker",
]

_KST = datetime.timezone(datetime.timedelta(hours=9))

def _normalize_for_compare(text: str) -> str:
    """제목/요약 비교를 위한 정규화 텍스트 생성."""
    t = clean_text(text or "").lower()
    t = re.sub(r"[^a-z0-9가-힣]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _is_title_like_summary(title: str, lines: list[str]) -> bool:
    """요약이 제목 반복인지(또는 유사) 판별."""
    if not lines:
        return True
    norm_title = _normalize_for_compare(title)
    if not norm_title:
        return True
    similar = 0
    for line in lines:
        norm_line = _normalize_for_compare(line)
        if not norm_line:
            continue
        if norm_line == norm_title:
            similar += 1
            continue
        if norm_line in norm_title or norm_title in norm_line:
            similar += 1
            continue
        if jaccard_tokens(norm_line, norm_title) >= 0.85:
            similar += 1
            continue
    return similar >= max(1, len(lines))


def _safe_read_json(path: str, default: Any) -> Any:
    """JSON 파일을 안전하게 로드, 실패 시 기본값 반환."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


_IMPACT_LEVEL_SCORE = {"long": 4, "med": 3, "low": 2}

def _impact_base_level(label: str) -> str:
    return IMPACT_SIGNAL_BASE_LEVELS.get(label, "low")

def _has_long_trigger(label: str, evidence: str) -> bool:
    if not label or not evidence:
        return False
    triggers = IMPACT_SIGNAL_LONG_TRIGGERS.get(label, [])
    if not triggers:
        return False
    text = clean_text(evidence).lower()
    return any(kw.lower() in text for kw in triggers)

def _impact_level_for_evidence(label: str, evidence: str) -> str:
    base = _impact_base_level(label)
    if _has_long_trigger(label, evidence):
        return "long"
    return base

def _infer_importance_from_signals(signals: Any) -> int:
    """impactSignals 기반 기본 중요도 점수 산정 (트리거 기반 업그레이드)."""
    if not signals:
        return 1
    best = 1
    if isinstance(signals, list):
        for entry in signals:
            if isinstance(entry, dict):
                label = clean_text(entry.get("label") or "").lower()
                evidence = clean_text(entry.get("evidence") or "")
                if label not in _ALLOWED_IMPACT_LABELS or not evidence:
                    continue
                level = _impact_level_for_evidence(label, evidence)
                best = max(best, _IMPACT_LEVEL_SCORE.get(level, 1))
            else:
                label = clean_text(str(entry)).lower()
                if label not in _ALLOWED_IMPACT_LABELS:
                    continue
                base = _impact_base_level(label)
                best = max(best, _IMPACT_LEVEL_SCORE.get(base, 1))
        return best
    return 1


def _pick_summary_source(
    title: str,
    summary: str,
    summary_raw: str,
) -> str:
    """LLM 요약이 없을 때 RSS 요약만 사용 (본문은 사용하지 않음)."""
    title_clean = clean_text(title)
    base = clean_text(summary_raw or summary)
    if base and base.lower() != title_clean.lower():
        return base
    return ""

def _load_existing_digest(path: str) -> DailyDigest | None:
    """기존 digest 파일 로드."""
    return _safe_read_json(path, None)

def _is_valid_digest(digest: dict) -> bool:
    """MVP 안전장치: 최소 개수 + 핵심 필드 존재 여부만 검사 (엄격하게)."""
    valid, _ = _validate_digest(digest)
    return valid

def _missing_required_fields(item: dict, required: set[str]) -> list[str]:
    missing: list[str] = []
    for key in required:
        if key not in item:
            missing.append(key)
    return missing

def _is_simple_incident_item(item: dict) -> bool:
    title = clean_text(item.get("title") or "")
    summary = item.get("summary") or []
    if isinstance(summary, list):
        summary_text = " ".join([str(s) for s in summary if s])
    else:
        summary_text = str(summary or "")
    text = clean_text(f"{title} {summary_text}").lower()
    if not text:
        return False
    return any(kw in text for kw in _SIMPLE_INCIDENT_KEYWORDS)

def _has_duplicate_impact_labels(impact_signals: Any) -> bool:
    if not isinstance(impact_signals, list):
        return False
    labels: list[str] = []
    for entry in impact_signals:
        if isinstance(entry, dict):
            label = clean_text(entry.get("label") or "").lower()
        else:
            label = clean_text(str(entry)).lower()
        if not label:
            continue
        labels.append(label)
    return len(set(labels)) != len(labels)

def _has_duplicate_impact_evidence(impact_signals: Any) -> bool:
    if not isinstance(impact_signals, list):
        return False
    seen: set[str] = set()
    for entry in impact_signals:
        if not isinstance(entry, dict):
            continue
        evidence = clean_text(entry.get("evidence") or "")
        if not evidence:
            continue
        key = _normalize_evidence_key(evidence)
        if not key:
            continue
        if key in seen:
            return True
        seen.add(key)
    return False

def _has_number_token(text: str) -> bool:
    if not text:
        return False
    t = clean_text(text)
    if not t:
        return False
    if re.search(r"\d", t):
        return True
    return any(unit in t for unit in ["억", "조", "만", "%", "달러", "원", "billion", "million", "trillion", "usd", "$"])

def _parse_datetime(value: str) -> datetime.datetime | None:
    if not value:
        return None
    try:
        dt = datetime.datetime.fromisoformat(value)
    except Exception:
        try:
            dt = email.utils.parsedate_to_datetime(value)
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_KST)
    return dt.astimezone(datetime.timezone.utc)

def _parse_date_base(value: str) -> datetime.datetime | None:
    if not value:
        return None
    try:
        d = datetime.date.fromisoformat(value)
    except Exception:
        return None
    return datetime.datetime(d.year, d.month, d.day, tzinfo=_KST).astimezone(datetime.timezone.utc)

def _policy_evidence_valid(text: str) -> bool:
    t = clean_text(text or "").lower()
    if not t:
        return False
    policy_keywords = [k.lower() for k in IMPACT_SIGNALS_MAP.get("policy", [])]
    if not any(k in t for k in policy_keywords):
        return False
    if any(k in t for k in _POLICY_STRONG_KEYWORDS):
        return True
    if any(k in t for k in _POLICY_GOV_KEYWORDS) and any(k in t for k in _POLICY_NEGOTIATION_KEYWORDS):
        return False
    if any(k in t for k in _POLICY_TRADE_ONLY_KEYWORDS):
        return False
    return False

def _sanctions_evidence_valid(text: str) -> bool:
    t = clean_text(text or "").lower()
    if not t:
        return False
    return any(k.lower() in t for k in SANCTIONS_EVIDENCE_KEYWORDS)

def _market_demand_evidence_valid(text: str) -> bool:
    t = clean_text(text or "").lower()
    if not t:
        return False
    return any(k.lower() in t for k in MARKET_DEMAND_EVIDENCE_KEYWORDS)

def _earnings_evidence_valid(text: str) -> bool:
    t = clean_text(text or "").lower()
    if not t:
        return False
    return any(k in t for k in _EARNINGS_METRIC_KEYWORDS) and _has_number_token(t)

def _capex_evidence_valid(text: str) -> bool:
    t = clean_text(text or "").lower()
    if not t:
        return False
    has_action = any(k in t for k in _CAPEX_ACTION_KEYWORDS)
    has_plan = any(k in t for k in _CAPEX_PLAN_KEYWORDS) or _has_number_token(t)
    return has_action and has_plan

def _infra_evidence_valid(text: str) -> bool:
    t = clean_text(text or "").lower()
    if not t:
        return False
    return any(k in t for k in _INFRA_KEYWORDS)

def _security_evidence_valid(text: str) -> bool:
    t = clean_text(text or "").lower()
    if not t:
        return False
    return any(k.lower() in t for k in SECURITY_EVIDENCE_KEYWORDS)

def _label_evidence_valid(label: str, evidence: str) -> bool:
    if label == "policy":
        return _policy_evidence_valid(evidence)
    if label == "sanctions":
        return _sanctions_evidence_valid(evidence)
    if label == "market-demand":
        return _market_demand_evidence_valid(evidence)
    if label == "earnings":
        return _earnings_evidence_valid(evidence)
    if label == "capex":
        return _capex_evidence_valid(evidence)
    if label == "infra":
        return _infra_evidence_valid(evidence)
    if label == "security":
        return _security_evidence_valid(evidence)
    return False

def _iter_impact_signal_entries(impact_signals: Any) -> list[tuple[str, str]]:
    if not isinstance(impact_signals, list):
        return []
    entries: list[tuple[str, str]] = []
    for entry in impact_signals:
        if isinstance(entry, dict):
            label = clean_text(entry.get("label") or "").lower()
            evidence = clean_text(entry.get("evidence") or "")
        else:
            label = clean_text(str(entry)).lower()
            evidence = ""
        if not label:
            continue
        entries.append((label, evidence))
    return entries

def _normalize_evidence_key(text: str) -> str:
    t = clean_text(text or "").lower()
    if not t:
        return ""
    t = re.sub(r"[^a-z0-9가-힣]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def _split_sentences_for_evidence(text: str) -> list[str]:
    if not text:
        return []
    cleaned = clean_text(text)
    if not cleaned:
        return []
    parts = [
        p.strip()
        for p in re.split(r"(?<=[\.\!\?。])\s+|(?<=다\.)\s+", cleaned)
        if p.strip()
    ]
    return parts if parts else [cleaned]

def _is_evidence_too_short(text: str) -> bool:
    t = clean_text(text or "")
    if not t:
        return True
    if len(t) < 20:
        return True
    if len(t.split()) < 6:
        return True
    return False

def _evidence_keyword_hits(label: str, evidence: str) -> int:
    t = clean_text(evidence or "").lower()
    if not t:
        return 0
    if label == "sanctions":
        keywords = SANCTIONS_EVIDENCE_KEYWORDS
    elif label == "market-demand":
        keywords = MARKET_DEMAND_EVIDENCE_KEYWORDS
    elif label == "security":
        keywords = SECURITY_EVIDENCE_KEYWORDS
    elif label == "policy":
        keywords = IMPACT_SIGNALS_MAP.get("policy", [])
    elif label == "capex":
        keywords = _CAPEX_ACTION_KEYWORDS
    elif label == "earnings":
        keywords = _EARNINGS_METRIC_KEYWORDS
    elif label == "infra":
        keywords = _INFRA_KEYWORDS
    else:
        keywords = []
    return sum(1 for kw in keywords if kw.lower() in t)

def _evidence_specificity_score(label: str, evidence: str) -> tuple[int, int, int]:
    length_score = len(clean_text(evidence or ""))
    number_score = 1 if _has_number_token(evidence or "") else 0
    keyword_score = _evidence_keyword_hits(label, evidence)
    return (length_score, number_score, keyword_score)

def _remap_label_by_evidence(evidence: str) -> str:
    for label in _IMPACT_LABEL_PRIORITY:
        if _label_evidence_valid(label, evidence):
            return label
    return ""

def _extract_evidence_sentence(label: str, text: str) -> str:
    if not label or not text:
        return ""
    candidates = []
    for sentence in _split_sentences_for_evidence(text):
        if not sentence:
            continue
        if _is_evidence_too_short(sentence):
            continue
        if not _label_evidence_valid(label, sentence):
            continue
        candidates.append(sentence)
    if not candidates:
        return ""
    candidates.sort(key=lambda s: _evidence_specificity_score(label, s), reverse=True)
    return candidates[0]

def _sanitize_impact_signals(raw: Any, full_text: str, summary_text: str) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    source_text = full_text or summary_text or ""
    source_norm = clean_text(source_text or "").lower()
    if not source_norm:
        return []
    candidates: dict[str, tuple[tuple[int, int, int], str]] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        label = clean_text(entry.get("label") or "").lower()
        evidence = clean_text(entry.get("evidence") or "")
        if label not in _ALLOWED_IMPACT_LABELS:
            continue
        if not evidence:
            continue
        if _is_evidence_too_short(evidence):
            continue
        if clean_text(evidence).lower() not in source_norm:
            continue
        if not _label_evidence_valid(label, evidence):
            continue
        score = _evidence_specificity_score(label, evidence)
        best = candidates.get(label)
        if not best or score > best[0]:
            candidates[label] = (score, evidence)

    ordered: list[tuple[str, tuple[int, int, int], str]] = [
        (label, payload[0], payload[1]) for label, payload in candidates.items()
    ]
    ordered.sort(key=lambda x: x[1], reverse=True)

    cleaned: list[dict[str, str]] = []
    seen_evidence: set[str] = set()
    for label, _score, evidence in ordered:
        evidence_key = _normalize_evidence_key(evidence)
        if not evidence_key or evidence_key in seen_evidence:
            continue
        cleaned.append({"label": label, "evidence": evidence})
        seen_evidence.add(evidence_key)
    return cleaned

def _collect_item_errors(
    item: dict,
    *,
    full_text: str,
    summary_text: str,
) -> list[str]:
    errors: list[str] = []
    impact_signals = item.get("impactSignals")
    if not isinstance(impact_signals, list):
        errors.append("ERROR: INVALID_IMPACT_SIGNAL_FORMAT")
        impact_signals = []
    elif any(not isinstance(entry, dict) for entry in impact_signals):
        errors.append("ERROR: INVALID_IMPACT_SIGNAL_FORMAT")
    if _has_duplicate_impact_labels(impact_signals):
        errors.append("ERROR: DUPLICATE_IMPACT_SIGNAL_LABEL")
    if _has_duplicate_impact_evidence(impact_signals):
        errors.append("ERROR: DUPLICATE_IMPACT_SIGNAL_EVIDENCE")

    source_text = full_text or summary_text or ""
    source_lower = clean_text(source_text).lower()

    for label, evidence in _iter_impact_signal_entries(impact_signals):
        if label not in _ALLOWED_IMPACT_LABELS:
            errors.append("ERROR: INVALID_IMPACT_LABEL")
            continue
        if not evidence:
            errors.append("ERROR: IMPACT_EVIDENCE_REQUIRED")
            continue
        if _is_evidence_too_short(evidence):
            errors.append("ERROR: IMPACT_EVIDENCE_TOO_SHORT")
            continue
        if source_lower and clean_text(evidence).lower() not in source_lower:
            errors.append("ERROR: IMPACT_EVIDENCE_REQUIRED")
            continue
        if label == "policy" and not _policy_evidence_valid(evidence):
            errors.append("ERROR: INVALID_POLICY_LABEL")
        if label == "sanctions" and not _sanctions_evidence_valid(evidence):
            errors.append("ERROR: INVALID_SANCTIONS_LABEL")
        if label == "market-demand" and not _market_demand_evidence_valid(evidence):
            errors.append("ERROR: INVALID_MARKET_DEMAND_LABEL")
        if label == "earnings" and not _earnings_evidence_valid(evidence):
            errors.append("ERROR: INVALID_EARNINGS_LABEL")
        if label == "capex" and not _capex_evidence_valid(evidence):
            errors.append("ERROR: INVALID_CAPEX_LABEL")
        if label == "infra" and not _infra_evidence_valid(evidence):
            errors.append("ERROR: INVALID_INFRA_LABEL")
        if label == "security" and not _security_evidence_valid(evidence):
            errors.append("ERROR: INVALID_SECURITY_LABEL")

    if item.get("qualityLabel") == "low_quality":
        if item.get("status") != "dropped" and not _low_quality_exception_ok(item):
            errors.append("ERROR: LOW_QUALITY_MISMATCH")

    try:
        importance = int(item.get("importance") or 0)
    except Exception:
        importance = 0
    if importance >= 3 and isinstance(impact_signals, list) and len(impact_signals) == 0:
        errors.append("ERROR: IMPACT_SIGNALS_REQUIRED")

    published_at = _parse_datetime(str(item.get("publishedAt") or ""))
    base_date = _parse_date_base(str(item.get("date") or ""))
    if published_at and base_date:
        diff_hours = abs((published_at - base_date).total_seconds()) / 3600.0
        if diff_hours > 72 and item.get("isCarriedOver") is not True:
            errors.append("ERROR: OUTDATED_ITEM")

    return list(dict.fromkeys(errors))

def classify_errors(errors: list[str]) -> dict[str, list[str]]:
    s1 = {"ERROR: DUPLICATE_DEDUPE_KEY", "ERROR: OUTDATED_ITEM"}
    s2 = {
        "ERROR: INVALID_POLICY_LABEL",
        "ERROR: INVALID_SANCTIONS_LABEL",
        "ERROR: INVALID_MARKET_DEMAND_LABEL",
        "ERROR: INVALID_EARNINGS_LABEL",
        "ERROR: INVALID_CAPEX_LABEL",
        "ERROR: INVALID_INFRA_LABEL",
        "ERROR: INVALID_SECURITY_LABEL",
        "ERROR: INVALID_IMPACT_LABEL",
        "ERROR: IMPACT_EVIDENCE_REQUIRED",
        "ERROR: IMPACT_EVIDENCE_TOO_SHORT",
        "ERROR: INVALID_IMPACT_SIGNAL_FORMAT",
        "ERROR: DUPLICATE_IMPACT_SIGNAL_EVIDENCE",
        "ERROR: DUPLICATE_IMPACT_SIGNAL_LABEL",
    }
    s3 = {"ERROR: LOW_QUALITY_MISMATCH"}
    return {
        "s1": [e for e in errors if e in s1],
        "s2": [e for e in errors if e in s2],
        "s3": [e for e in errors if e in s3],
        "unknown": [e for e in errors if e not in s1 | s2 | s3],
    }

def _normalize_impact_signal_format(item: dict) -> list[dict[str, str]]:
    impact_signals = item.get("impactSignals")
    if not isinstance(impact_signals, list):
        item["impactSignals"] = []
        return []
    cleaned = [entry for entry in impact_signals if isinstance(entry, dict)]
    item["impactSignals"] = cleaned
    return cleaned

def _dedupe_signals_by_evidence(signals: list[dict[str, str]]) -> list[dict[str, str]]:
    if not signals:
        return []
    priority = {label: idx for idx, label in enumerate(_IMPACT_LABEL_PRIORITY)}
    grouped: dict[str, list[dict[str, str]]] = {}
    for signal in signals:
        evidence = clean_text(signal.get("evidence") or "")
        key = _normalize_evidence_key(evidence)
        if not key:
            continue
        grouped.setdefault(key, []).append(signal)
    deduped: list[dict[str, str]] = []
    for entries in grouped.values():
        entries.sort(key=lambda s: priority.get(s.get("label") or "", 99))
        deduped.append(entries[0])
    return deduped

def _auto_fix_impact_signals(item: dict) -> list[str]:
    auto_fixed: list[str] = []
    signals = _normalize_impact_signal_format(item)
    if not signals:
        return auto_fixed
    source_text = (item.get("_fullText") or "") or (item.get("_summaryText") or "")
    source_lower = clean_text(source_text).lower()
    cleaned: list[dict[str, str]] = []
    for entry in signals:
        label = clean_text(entry.get("label") or "").lower()
        evidence = clean_text(entry.get("evidence") or "")
        if label not in _ALLOWED_IMPACT_LABELS:
            new_label = _remap_label_by_evidence(evidence)
            if new_label:
                entry["label"] = new_label
                label = new_label
                auto_fixed.append("remap_label")
            else:
                continue

        if not evidence or _is_evidence_too_short(evidence) or (source_lower and clean_text(evidence).lower() not in source_lower):
            extracted = _extract_evidence_sentence(label, source_text)
            if extracted:
                entry["evidence"] = extracted
                evidence = extracted
                auto_fixed.append("replace_evidence")
            else:
                continue

        if not _label_evidence_valid(label, evidence):
            new_label = _remap_label_by_evidence(evidence)
            if new_label and _label_evidence_valid(new_label, evidence):
                entry["label"] = new_label
                label = new_label
                auto_fixed.append("remap_label")
            else:
                continue

        cleaned.append({"label": label, "evidence": evidence})

    by_label: dict[str, dict[str, str]] = {}
    by_label_score: dict[str, tuple[int, int, int]] = {}
    for entry in cleaned:
        label = entry.get("label") or ""
        evidence = entry.get("evidence") or ""
        score = _evidence_specificity_score(label, evidence)
        if label not in by_label or score > by_label_score.get(label, (0, 0, 0)):
            by_label[label] = entry
            by_label_score[label] = score
    cleaned = list(by_label.values())
    cleaned = _dedupe_signals_by_evidence(cleaned)
    item["impactSignals"] = cleaned
    return auto_fixed

def apply_auto_fixes(item: dict) -> list[str]:
    auto_fixed: list[str] = []
    auto_fixed.extend(_auto_fix_impact_signals(item))
    return auto_fixed

def revalidate(item: dict, *, full_text: str, summary_text: str) -> list[str]:
    return _collect_item_errors(item, full_text=full_text, summary_text=summary_text)

def handle_hard_fails(item: dict, errors: list[str]) -> None:
    if "ERROR: OUTDATED_ITEM" in errors:
        item["status"] = "dropped"
        item["dropReason"] = "outdated"
    if "ERROR: DUPLICATE_DEDUPE_KEY" in errors:
        item["status"] = "dropped"
        item["dropReason"] = "duplicate"
    if any(e for e in errors if e not in {"ERROR: LOW_QUALITY_MISMATCH"}):
        if item.get("status") not in {"dropped"}:
            item["status"] = "dropped"
            item["dropReason"] = item.get("dropReason") or "validation_error"

def apply_soft_warnings(item: dict, errors: list[str]) -> None:
    if "ERROR: LOW_QUALITY_MISMATCH" not in errors:
        return
    item["qualityLabel"] = "low_quality"
    if not item.get("qualityReason"):
        item["qualityReason"] = "정보 부족"
    if "qualityTags" not in item:
        item["qualityTags"] = []
    try:
        importance = int(item.get("importance") or 0)
    except Exception:
        importance = 0
    if importance > 2:
        item["importance"] = 2
    elif importance > 1:
        item["importance"] = max(1, importance - 1)

def handle_validation_errors(item: dict, errors: list[str]) -> dict:
    log = {
        "item_id": item.get("id") or "",
        "original_errors": list(errors),
        "auto_fixed": [],
        "remaining_errors": [],
        "final_action": "kept",
    }
    classified = classify_errors(errors)
    if classified["s2"]:
        auto_fixed = apply_auto_fixes(item)
        log["auto_fixed"] = auto_fixed
    new_errors = revalidate(
        item,
        full_text=item.get("_fullText") or "",
        summary_text=item.get("_summaryText") or "",
    )
    log["remaining_errors"] = new_errors
    classified_after = classify_errors(new_errors)
    if classified_after["s1"] or classified_after["s2"] or classified_after["unknown"]:
        handle_hard_fails(
            item,
            classified_after["s1"] + classified_after["s2"] + classified_after["unknown"],
        )
    if classified_after["s3"]:
        apply_soft_warnings(item, classified_after["s3"])
    if item.get("status") == "dropped":
        log["final_action"] = "dropped"
    elif log["auto_fixed"]:
        log["final_action"] = "modified"
    return log

def _source_tier_rank(source_name: str | None) -> int:
    normalized = normalize_source_name(source_name or "").lower()
    if not normalized:
        return 0
    if normalized in {normalize_source_name(s).lower() for s in SOURCE_TIER_A if s}:
        return 2
    if normalized in {normalize_source_name(s).lower() for s in SOURCE_TIER_B if s}:
        return 1
    return 0

def _resolve_duplicate_dedupe_items(items: list[dict]) -> tuple[list[dict], list[dict]]:
    by_key: dict[str, list[dict]] = {}
    for it in items:
        key = clean_text(it.get("dedupeKey") or "")
        if not key:
            continue
        by_key.setdefault(key, []).append(it)
    if not by_key:
        return items, []
    logs: list[dict] = []
    kept_items: list[dict] = []
    for key, group in by_key.items():
        if len(group) == 1:
            kept_items.append(group[0])
            continue
        def _rank(item: dict) -> tuple[int, int, int, int]:
            quality_ok = 1 if item.get("qualityLabel") == "ok" else 0
            tier = _source_tier_rank(item.get("sourceName"))
            try:
                importance = int(item.get("importance") or 0)
            except Exception:
                importance = 0
            published = _parse_datetime(str(item.get("publishedAt") or ""))
            published_ts = int(published.timestamp()) if published else 0
            return (quality_ok, tier, importance, published_ts)

        winner = sorted(group, key=_rank, reverse=True)[0]
        kept_items.append(winner)
        for it in group:
            if it is winner:
                continue
            it["status"] = "dropped"
            it["dropReason"] = "duplicate"
            logs.append({
                "item_id": it.get("id") or "",
                "original_errors": ["ERROR: DUPLICATE_DEDUPE_KEY"],
                "auto_fixed": ["dedupe:drop"],
                "remaining_errors": [],
                "final_action": "dropped",
            })
    return [it for it in kept_items if it.get("status") != "dropped"], logs

def _low_quality_exception_ok(item: dict) -> bool:
    why = clean_text(item.get("whyImportant") or "")
    rationale = clean_text(item.get("importanceRationale") or "")
    try:
        importance = int(item.get("importance") or 0)
    except Exception:
        importance = 0
    if (
        why == "판단 근거 부족"
        and rationale == "근거 부족으로 영향 판단 불가"
        and importance == 1
    ):
        return True
    if item.get("qualityLabel") == "low_quality":
        return bool(item.get("qualityReason")) and importance <= 2
    return False

def _has_duplicate_dedupe_key(items: list[dict]) -> bool:
    seen: set[str] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        status = it.get("status")
        if status not in {"kept", "published"}:
            continue
        key = clean_text(it.get("dedupeKey") or "")
        if not key:
            continue
        if key in seen:
            return True
        seen.add(key)
    return False

def _validate_digest(digest: dict) -> tuple[bool, str]:
    """MVP 안전장치: 최소 개수 + 필수 필드 존재 여부 검사."""
    if not isinstance(digest, dict):
        return False, "INVALID_DIGEST"
    items = digest.get("items")
    if not isinstance(items, list) or len(items) < MIN_TOP_ITEMS or len(items) > TOP_LIMIT:
        return False, "INVALID_DIGEST"
    if _has_duplicate_dedupe_key(items):
        return False, "ERROR: DUPLICATE_DEDUPE_KEY"

    required_item_keys = {
        "id",
        "date",
        "category",
        "title",
        "summary",
        "whyImportant",
        "importanceRationale",
        "impactSignals",
        "dedupeKey",
        "sourceName",
        "sourceUrl",
        "publishedAt",
        "status",
        "importance",
        "qualityLabel",
        "qualityReason",
    }
    for it in items:
        if not isinstance(it, dict):
            return False, "INVALID_DIGEST"
        if _missing_required_fields(it, required_item_keys):
            return False, "VALIDATION_ERROR: MISSING_FIELD"
        published_at = _parse_datetime(str(it.get("publishedAt") or ""))
        base_date = _parse_date_base(str(it.get("date") or ""))
        if published_at and base_date:
            diff_hours = abs((published_at - base_date).total_seconds()) / 3600.0
            if diff_hours > 72:
                status = it.get("status")
                if status in {"kept", "published"} and it.get("isCarriedOver") is not True:
                    return False, "ERROR: OUTDATED_ITEM"
        if it.get("qualityLabel") == "low_quality":
            if it.get("status") != "dropped" and not _low_quality_exception_ok(it):
                return False, "ERROR: LOW_QUALITY_MISMATCH"
        impact_signals_value = it.get("impactSignals")
        if not isinstance(impact_signals_value, list):
            return False, "ERROR: INVALID_IMPACT_SIGNAL_FORMAT"
        if any(not isinstance(entry, dict) for entry in impact_signals_value):
            return False, "ERROR: INVALID_IMPACT_SIGNAL_FORMAT"
        if _has_duplicate_impact_labels(it.get("impactSignals")):
            return False, "ERROR: DUPLICATE_IMPACT_SIGNAL_LABEL"
        if _has_duplicate_impact_evidence(it.get("impactSignals")):
            return False, "ERROR: DUPLICATE_IMPACT_SIGNAL_EVIDENCE"
        for label, evidence in _iter_impact_signal_entries(it.get("impactSignals")):
            if label not in _ALLOWED_IMPACT_LABELS:
                return False, "ERROR: INVALID_IMPACT_LABEL"
            if not evidence:
                return False, "ERROR: IMPACT_EVIDENCE_REQUIRED"
            if _is_evidence_too_short(evidence):
                return False, "ERROR: IMPACT_EVIDENCE_TOO_SHORT"
            if label == "policy" and not _policy_evidence_valid(evidence):
                return False, "ERROR: INVALID_POLICY_LABEL"
            if label == "sanctions" and not _sanctions_evidence_valid(evidence):
                return False, "ERROR: INVALID_SANCTIONS_LABEL"
            if label == "market-demand" and not _market_demand_evidence_valid(evidence):
                return False, "ERROR: INVALID_MARKET_DEMAND_LABEL"
            if label == "earnings" and not _earnings_evidence_valid(evidence):
                return False, "ERROR: INVALID_EARNINGS_LABEL"
            if label == "capex" and not _capex_evidence_valid(evidence):
                return False, "ERROR: INVALID_CAPEX_LABEL"
            if label == "infra" and not _infra_evidence_valid(evidence):
                return False, "ERROR: INVALID_INFRA_LABEL"
            if label == "security" and not _security_evidence_valid(evidence):
                return False, "ERROR: INVALID_SECURITY_LABEL"
        try:
            importance = int(it.get("importance") or 0)
        except Exception:
            importance = 0
        if importance >= 3:
            impact_signals = it.get("impactSignals")
            if isinstance(impact_signals, list) and len(impact_signals) == 0:
                return False, "ERROR: IMPACT_SIGNALS_REQUIRED"
        if not it.get("title") or not it.get("sourceUrl"):
            return False, "INVALID_DIGEST"
        summary = it.get("summary")
        if not isinstance(summary, list) or len(summary) == 0:
            return False, "INVALID_DIGEST"
    return True, ""

def _atomic_write_json(path: str, payload: dict) -> None:
    """임시 파일로 저장 후 원자적 교체."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

def _load_dedupe_history(path: str) -> dict:
    """중복 제거 히스토리를 안전하게 로드."""
    data = _safe_read_json(path, {"version": 1, "by_date": {}})
    if not isinstance(data, dict):
        return {"version": 1, "by_date": {}}
    by_date = data.get("by_date")
    if not isinstance(by_date, dict):
        data["by_date"] = {}
    if "version" not in data:
        data["version"] = 1
    return data

def _update_dedupe_history(digest: dict, path: str, days: int) -> None:
    """최근 N일치 digest로 중복 제거 히스토리를 갱신."""
    date_str = digest.get("date")
    if not date_str:
        return
    history = _load_dedupe_history(path)
    by_date = history.get("by_date", {})
    items_out: list[dict] = []
    for it in digest.get("items", []) or []:
        if not isinstance(it, dict):
            continue
        if it.get("status") not in {"published", "kept"}:
            continue
        summary = it.get("summary")
        summary_text = " ".join(summary) if isinstance(summary, list) else str(summary or "")
        items_out.append({
            "id": it.get("id"),
            "dedupeKey": it.get("dedupeKey"),
            "clusterKey": it.get("clusterKey"),
            "title": it.get("title") or "",
            "summary": summary_text,
        })
    by_date[date_str] = items_out
    history["by_date"] = by_date

    if days > 0:
        try:
            base = datetime.date.fromisoformat(date_str)
        except Exception:
            base = None
        if base:
            keep = {(base - datetime.timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(0, days + 1)}
            for d in list(by_date.keys()):
                if d not in keep:
                    by_date.pop(d, None)

    _atomic_write_json(path, history)

def export_daily_digest_json(top_items: list[dict], output_path: str, config: dict) -> DailyDigest:
    """MVP 스키마로 변환해 JSON으로 저장."""
    now_kst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    date_str = now_kst.strftime("%Y-%m-%d")
    last_updated_at = now_kst.isoformat()

    items_out: list[dict[str, Any]] = []
    validation_logs: list[dict[str, Any]] = []
    out_index = 0
    kept_dedupe_keys: set[str] = set()
    for item in top_items[:TOP_LIMIT]:
        title = (item.get("title") or "").strip()
        link = (item.get("link") or "").strip()
        summary = (item.get("summary") or "").strip()
        summary_raw = (item.get("summaryRaw") or "").strip()
        full_text = (item.get("fullText") or "").strip()
        topic = (item.get("topic") or "").strip()
        source_name = (item.get("source") or "").strip()
        published = item.get("published")
        published_at = clean_text(str(published)) if published is not None else ""
        is_carried_over = item.get("isCarriedOver") is True

        drop_reason = item.get("dropReason") or ""
        status_value = item.get("status") or ("dropped" if drop_reason else "kept")
        is_merged = status_value == "merged"
        published_dt = _parse_datetime(published_at)
        base_dt = _parse_date_base(date_str)
        if published_dt and base_dt:
            diff_hours = abs((published_dt - base_dt).total_seconds()) / 3600.0
            if diff_hours > 72 and not is_carried_over:
                drop_reason = drop_reason or "outdated"
                status_value = "dropped"
        full_text_len = len(clean_text(full_text))
        if full_text_len < 50 and not drop_reason and not is_merged:
            drop_reason = "policy:full_text_missing"
            status_value = "dropped"
        if drop_reason or status_value == "dropped":
            continue

        ai_result = item.get("ai")
        should_skip_ai = bool(is_merged or drop_reason or status_value == "dropped" or full_text_len < 50)
        if not should_skip_ai and (not isinstance(ai_result, dict) or not ai_result):
            ai_result = enrich_item_with_ai(item) or {}
        elif not isinstance(ai_result, dict):
            ai_result = {}
        title_ko = clean_text(ai_result.get("title_ko") or "")
        if title_ko:
            title = title_ko
        ai_lines_raw = ai_result.get("summary_lines") or []
        summary_source = _pick_summary_source(
            title,
            summary,
            summary_raw,
        )
        summary_source = strip_summary_boilerplate(summary_source)
        summary_lines = ensure_lines_1_to_3(ai_lines_raw, summary_source)

        quality_label = ai_result.get("quality_label") or item.get("aiQuality") or "ok"
        quality_reason = clean_text(ai_result.get("quality_reason") or item.get("quality_reason") or "")
        if quality_label == "low_quality" and not quality_reason:
            quality_reason = "정보 부족"

        force_low_quality_downgrade = False
        if quality_label == "low_quality" and not drop_reason and status_value != "dropped" and not is_merged:
            if LOW_QUALITY_POLICY == "drop":
                drop_reason = f"ai_low_quality:{quality_reason}"
            else:
                force_low_quality_downgrade = True

        def _drop_reason_message(reason: str) -> str:
            if not reason:
                return "본문 확보 실패"
            if "full_text_missing" in reason or reason.startswith("fetch_failed:"):
                return "본문 확보 실패로 판단 불가입니다."
            if "summary_binary" in reason:
                return "요약/본문 데이터에 바이너리 또는 손상 텍스트가 포함되어 있습니다."
            if "summary_title_only" in reason:
                return "요약이 제목 반복으로 편집 기준을 충족하지 못합니다."
            if reason.startswith("ai_low_quality:"):
                return reason.split(":", 1)[1]
            return reason

        policy_drop_reason = ""
        if not is_merged and (
            any(contains_binary(line) for line in summary_lines)
            or contains_binary(summary_raw)
            or contains_binary(summary)
        ):
            policy_drop_reason = "summary_binary"
        elif drop_reason:
            policy_drop_reason = ""
        elif not is_merged and _is_title_like_summary(title, summary_lines):
            policy_drop_reason = "summary_title_only"

        if policy_drop_reason and not drop_reason and not is_merged:
            drop_reason = f"policy:{policy_drop_reason}"
        if drop_reason and not is_merged:
            status_value = "dropped"
            summary_lines = [f"요약 불가: {_drop_reason_message(drop_reason)}"]

        def _fallback_why() -> str:
            if drop_reason or status_value == "dropped":
                return f"요약 불가: {_drop_reason_message(drop_reason)}"
            return "기사 본문을 바탕으로 한 중요성 설명이 충분히 제공되지 않았습니다."

        def _fallback_importance_rationale() -> str:
            if drop_reason or status_value == "dropped":
                return f"근거: {_drop_reason_message(drop_reason)}"
            return "근거: 기사 텍스트에서 명확한 수치나 범위 근거를 찾지 못했습니다."

        why_important = clean_text(ai_result.get("why_important") or item.get("whyImportant") or "")
        importance_rationale = clean_text(ai_result.get("importance_rationale") or item.get("importanceRationale") or "")
        if not why_important:
            why_important = _fallback_why()
        if not importance_rationale:
            importance_rationale = _fallback_importance_rationale()
        if full_text_len < 50 and not is_merged:
            why_important = "본문 확보 실패로 판단 불가입니다."
            importance_rationale = "근거: 본문 확보 실패로 판단 불가입니다."
        if (drop_reason or status_value == "dropped") and not is_merged:
            why_important = _fallback_why()
            importance_rationale = _fallback_importance_rationale()

        dedupe_key = ai_result.get("dedupe_key") or item.get("dedupeKey", "")
        cluster_key = item.get("clusterKey") or ""
        raw_signals = ai_result.get("impact_signals")
        impact_signals_detail: list[dict[str, str]] = []
        if isinstance(raw_signals, list) and raw_signals and isinstance(raw_signals[0], dict):
            for entry in raw_signals:
                if not isinstance(entry, dict):
                    continue
                label = clean_text(entry.get("label") or "").lower()
                evidence = clean_text(entry.get("evidence") or "")
                if not label or not evidence:
                    continue
                impact_signals_detail.append({"label": label, "evidence": evidence})
        else:
            impact_signals = raw_signals or item.get("impactSignals", [])
            evidence_map = ai_result.get("impact_signals_evidence") or {}
            for label in impact_signals:
                evidence = clean_text(evidence_map.get(label) or "")
                if not evidence:
                    continue
                impact_signals_detail.append({"label": label, "evidence": evidence})

        summary_text = " ".join(summary_lines) if summary_lines else summary
        impact_signals_detail = _sanitize_impact_signals(impact_signals_detail, full_text, summary_text)
        impact_signals = [d["label"] for d in impact_signals_detail]
        importance = ai_result.get("importance_score")
        if not importance:
            importance = _infer_importance_from_signals(impact_signals_detail)
        if not impact_signals_detail:
            importance = min(int(importance or 1), 2)
        if quality_label == "low_quality":
            low_quality_item = {
                "whyImportant": why_important,
                "importanceRationale": importance_rationale,
                "importance": importance,
            }
            if not _low_quality_exception_ok(low_quality_item):
                status_value = "dropped"
        if force_low_quality_downgrade and not is_merged:
            try:
                max_importance = int(LOW_QUALITY_DOWNGRADE_MAX_IMPORTANCE)
            except Exception:
                max_importance = 1
            importance = min(int(importance or 1), max(0, max_importance))
            importance_rationale = f"근거: {LOW_QUALITY_DOWNGRADE_RATIONALE}"
        read_time_sec = item.get("readTimeSec")
        if not read_time_sec:
            read_time_sec = estimate_read_time_seconds(" ".join(summary_lines) if summary_lines else summary)

        category = item.get("aiCategory") or map_topic_to_category(topic)

        if (drop_reason or status_value == "dropped") and not is_merged:
            quality_label = "low_quality"
            if not quality_reason:
                quality_reason = _drop_reason_message(drop_reason)
        if not quality_reason:
            quality_reason = "정보성 기사"
        if status_value == "dropped":
            continue
        if status_value in {"kept", "published"}:
            key = clean_text(dedupe_key)
            if key:
                if key in kept_dedupe_keys:
                    continue
                kept_dedupe_keys.add(key)

        out_index += 1
        out_item = {
            "id": f"{date_str}_{out_index}",
            "date": date_str,
            "category": category,
            "title": title,
            "summary": summary_lines if summary_lines else [summary],
            "whyImportant": why_important,
            "importanceRationale": importance_rationale,
            "impactSignals": impact_signals_detail,
            "dedupeKey": dedupe_key,
            "clusterKey": cluster_key,
            "matchedTo": item.get("matchedTo"),
            "sourceName": source_name,
            "sourceUrl": link,
            "publishedAt": published_at,
            "readTimeSec": read_time_sec,
            "status": status_value,
            "importance": importance,
            "qualityLabel": quality_label,
            "qualityReason": quality_reason,
            "isBriefing": False,
        }
        if is_carried_over:
            out_item["isCarriedOver"] = True
        if drop_reason and not is_merged:
            out_item["dropReason"] = drop_reason
        out_item["_fullText"] = full_text
        out_item["_summaryText"] = summary_text
        item_errors = _collect_item_errors(
            out_item,
            full_text=full_text,
            summary_text=summary_text,
        )
        log_entry = handle_validation_errors(out_item, item_errors)
        validation_logs.append(log_entry)
        out_item.pop("_fullText", None)
        out_item.pop("_summaryText", None)
        if out_item.get("status") == "dropped":
            continue
        items_out.append(out_item)

    # merged 기사 matchedTo를 대표 기사 id로 보정
    cluster_rep: dict[str, str] = {}
    for it in items_out:
        if it.get("status") in {"published", "kept"}:
            ck = it.get("clusterKey") or ""
            if ck and ck not in cluster_rep:
                cluster_rep[ck] = it.get("id", "")
    for it in items_out:
        if it.get("status") == "merged":
            ck = it.get("clusterKey") or ""
            rep_id = cluster_rep.get(ck)
            if rep_id:
                it["matchedTo"] = rep_id

    items_out, dedupe_logs = _resolve_duplicate_dedupe_items(items_out)
    validation_logs.extend(dedupe_logs)

    digest = {
        "date": date_str,
        "selectionCriteria": config["selection_criteria"],
        "editorNote": config["editor_note"],
        "question": config["question"],
        "lastUpdatedAt": last_updated_at,
        "items": items_out,
    }

    for log_entry in validation_logs:
        try:
            print(json.dumps(log_entry, ensure_ascii=False))
        except Exception:
            pass

    valid, error = _validate_digest(digest)
    if not valid:
        if error == "INVALID_DIGEST" and 0 < len(items_out) < MIN_TOP_ITEMS:
            print(f"⚠️ 최소 개수({MIN_TOP_ITEMS}) 미달로 {len(items_out)}개만 저장합니다.")
            _atomic_write_json(output_path, digest)
            try:
                _update_dedupe_history(digest, DEDUPE_HISTORY_PATH, DEDUPE_RECENT_DAYS)
            except Exception:
                pass
            return digest
        if error == "VALIDATION_ERROR: MISSING_FIELD":
            raise RuntimeError(error)
        if error == "ERROR: IMPACT_SIGNALS_REQUIRED":
            raise RuntimeError(error)
        if error == "ERROR: DUPLICATE_IMPACT_SIGNAL_LABEL":
            raise RuntimeError(error)
        if error in {
            "ERROR: INVALID_POLICY_LABEL",
            "ERROR: INVALID_SANCTIONS_LABEL",
            "ERROR: INVALID_MARKET_DEMAND_LABEL",
            "ERROR: INVALID_EARNINGS_LABEL",
            "ERROR: INVALID_CAPEX_LABEL",
            "ERROR: INVALID_INFRA_LABEL",
            "ERROR: INVALID_SECURITY_LABEL",
            "ERROR: INVALID_IMPACT_LABEL",
            "ERROR: IMPACT_EVIDENCE_REQUIRED",
            "ERROR: IMPACT_EVIDENCE_TOO_SHORT",
            "ERROR: INVALID_IMPACT_SIGNAL_FORMAT",
            "ERROR: DUPLICATE_IMPACT_SIGNAL_EVIDENCE",
            "ERROR: LOW_QUALITY_MISMATCH",
            "ERROR: DUPLICATE_DEDUPE_KEY",
            "ERROR: OUTDATED_ITEM",
        }:
            raise RuntimeError(error)
        existing = _load_existing_digest(output_path)
        if existing and _is_valid_digest(existing):
            print(f"⚠️ 오늘 digest 생성이 불완전하여 기존 파일({output_path})을 유지합니다.")
            return existing
        raise RuntimeError(
            f"digest 생성 실패: 유효한 {MIN_TOP_ITEMS}~{TOP_LIMIT}개 뉴스가 생성되지 않았고 기존 파일도 없습니다."
        )

    _atomic_write_json(output_path, digest)
    try:
        _update_dedupe_history(digest, DEDUPE_HISTORY_PATH, DEDUPE_RECENT_DAYS)
    except Exception:
        pass
    return digest

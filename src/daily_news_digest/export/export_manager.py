from __future__ import annotations

import datetime
import json
import os
import re
from typing import Any

from daily_news_digest.processing.ai_enricher import enrich_item_with_ai
from daily_news_digest.processing.scoring import map_topic_to_category
from daily_news_digest.processing.dedupe import DedupeEngine
from daily_news_digest.core.config import (
    DEDUPE_HISTORY_PATH,
    DEDUPE_RECENT_DAYS,
    DEDUPKEY_NGRAM_N,
    DEDUPKEY_NGRAM_SIM,
    LOW_QUALITY_DOWNGRADE_MAX_IMPORTANCE,
    LOW_QUALITY_DOWNGRADE_RATIONALE,
    LOW_QUALITY_POLICY,
    METRICS_JSON,
    MIN_TOP_ITEMS,
    TOP_LIMIT,
    TITLE_DEDUPE_JACCARD,
)
from daily_news_digest.core.constants import (
    ALLOWED_IMPACT_SIGNALS,
    DEDUPE_CLUSTER_DOMAINS,
    DEDUPE_CLUSTER_EVENT_LABELS,
    DEDUPE_CLUSTER_MAX_ENTITIES,
    DEDUPE_CLUSTER_MAX_TOKENS,
    DEDUPE_CLUSTER_RELATIONS,
    DEDUPE_EVENT_GROUPS,
    DEDUPE_EVENT_TOKENS,
    DEDUPE_NOISE_WORDS,
    IMPACT_SIGNAL_BASE_LEVELS,
    IMPACT_SIGNAL_LONG_TRIGGERS,
    IMPACT_SIGNALS_MAP,
    MARKET_DEMAND_EVIDENCE_KEYWORDS,
    MEDIA_SUFFIXES,
    SANCTIONS_EVIDENCE_KEYWORDS,
    SECURITY_EVIDENCE_KEYWORDS,
    SOURCE_TIER_A,
    SOURCE_TIER_B,
    MONTH_TOKENS,
    STOPWORDS,
    normalize_source_name,
)
from daily_news_digest.models import DailyDigest
from daily_news_digest.utils import (
    clean_text,
    contains_binary,
    ensure_lines_1_to_3,
    estimate_read_time_seconds,
    jaccard_tokens,
    jaccard,
    normalize_title_for_dedupe,
    normalize_token_for_dedupe,
    parse_date_base_utc,
    parse_datetime_utc,
    strip_summary_boilerplate,
)

# 분리된 모듈에서 import
from daily_news_digest.export.constants import (
    ALLOWED_IMPACT_LABELS,
    ALIGNMENT_TRIGGERS,
    CAPEX_ACTION_KEYWORDS,
    CAPEX_PLAN_KEYWORDS,
    EARNINGS_METRIC_KEYWORDS,
    IMPACT_LABEL_PRIORITY,
    IMPACT_LEVEL_SCORE,
    INFRA_KEYWORDS,
    KST,
    POLICY_GOV_KEYWORDS,
    POLICY_NEGOTIATION_KEYWORDS,
    POLICY_STRONG_KEYWORDS,
    POLICY_TRADE_ONLY_KEYWORDS,
    SIMPLE_INCIDENT_KEYWORDS,
)
from daily_news_digest.export.validators.evidence import (
    has_number_token,
    label_evidence_valid,
    policy_evidence_valid,
    sanctions_evidence_valid,
    evidence_keyword_hits,          # NEW
    evidence_specificity_score,     # NEW
)
from daily_news_digest.export.validators.impact_signal import (
    has_duplicate_impact_evidence,
    has_duplicate_impact_labels,
    is_evidence_too_short,
    normalize_evidence_key,
    sanitize_impact_signals,
)

# 하위 호환성을 위한 alias (분리된 모듈에서 import됨)
_ALLOWED_IMPACT_LABELS = ALLOWED_IMPACT_LABELS
_IMPACT_LABEL_PRIORITY = IMPACT_LABEL_PRIORITY
_SIMPLE_INCIDENT_KEYWORDS = SIMPLE_INCIDENT_KEYWORDS
_POLICY_STRONG_KEYWORDS = POLICY_STRONG_KEYWORDS
_POLICY_GOV_KEYWORDS = POLICY_GOV_KEYWORDS
_POLICY_NEGOTIATION_KEYWORDS = POLICY_NEGOTIATION_KEYWORDS
_POLICY_TRADE_ONLY_KEYWORDS = POLICY_TRADE_ONLY_KEYWORDS
_EARNINGS_METRIC_KEYWORDS = EARNINGS_METRIC_KEYWORDS
_CAPEX_ACTION_KEYWORDS = CAPEX_ACTION_KEYWORDS
_CAPEX_PLAN_KEYWORDS = CAPEX_PLAN_KEYWORDS
_INFRA_KEYWORDS = INFRA_KEYWORDS
_KST = KST
_IMPACT_LEVEL_SCORE = IMPACT_LEVEL_SCORE
_ALIGNMENT_TRIGGERS = ALIGNMENT_TRIGGERS

# 함수 alias (하위 호환성)
_has_number_token = has_number_token
_label_evidence_valid = label_evidence_valid
_policy_evidence_valid = policy_evidence_valid
_sanctions_evidence_valid = sanctions_evidence_valid

# Impact Signal 관련함수 alias
_sanitize_impact_signals = sanitize_impact_signals
_normalize_evidence_key = normalize_evidence_key
_is_evidence_too_short = is_evidence_too_short
_has_duplicate_impact_labels = has_duplicate_impact_labels
_has_duplicate_impact_evidence = has_duplicate_impact_evidence
_evidence_keyword_hits = evidence_keyword_hits
_evidence_specificity_score = evidence_specificity_score






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
_DEDUPE_ENGINE: DedupeEngine | None = None

_ALIGNMENT_TRIGGERS = [
    "policy",
    "sanctions",
    "capex",
    "earnings",
    "tariff",
    "제재",
    "법안",
    "실적",
    "투자",
    "증설",
]


def _get_dedupe_engine() -> DedupeEngine:
    global _DEDUPE_ENGINE
    if _DEDUPE_ENGINE is not None:
        return _DEDUPE_ENGINE
    _DEDUPE_ENGINE = DedupeEngine(
        stopwords=STOPWORDS,
        dedupe_noise_words=DEDUPE_NOISE_WORDS,
        month_tokens=MONTH_TOKENS,
        media_suffixes=MEDIA_SUFFIXES,
        title_dedupe_jaccard=TITLE_DEDUPE_JACCARD,
        dedupe_ngram_n=DEDUPKEY_NGRAM_N,
        dedupe_ngram_sim=DEDUPKEY_NGRAM_SIM,
        dedupe_event_tokens=DEDUPE_EVENT_TOKENS,
        dedupe_event_groups=DEDUPE_EVENT_GROUPS,
        cluster_event_labels=DEDUPE_CLUSTER_EVENT_LABELS,
        cluster_domains=DEDUPE_CLUSTER_DOMAINS,
        cluster_relations=DEDUPE_CLUSTER_RELATIONS,
        cluster_max_tokens=DEDUPE_CLUSTER_MAX_TOKENS,
        cluster_max_entities=DEDUPE_CLUSTER_MAX_ENTITIES,
        normalize_title_for_dedupe_func=normalize_title_for_dedupe,
        normalize_token_for_dedupe_func=normalize_token_for_dedupe,
        clean_text_func=clean_text,
        jaccard_func=jaccard,
    )
    return _DEDUPE_ENGINE


def _normalize_text_tokens(text: str) -> set[str]:
    t = clean_text(text or "").lower()
    t = re.sub(r"[^a-z0-9가-힣\s]", " ", t)
    tokens: set[str] = set()
    for tok in t.split():
        if re.search(r"[가-힣]", tok):
            if len(tok) < 2:
                continue
        else:
            if len(tok) < 3:
                continue
        tokens.add(tok)
    return tokens


def _regenerate_keys_from_title_summary(item: dict) -> None:
    title = clean_text(item.get("title") or "")
    summary_list = item.get("summary")
    if isinstance(summary_list, list):
        summary_text = " ".join([str(x) for x in summary_list if x])
    else:
        summary_text = clean_text(str(summary_list or ""))

    input_text = f"{title} {summary_text}".strip()

    engine = _get_dedupe_engine()

    dedupe_key = engine.build_dedupe_key(title, summary_text)
    cluster_key = engine.build_cluster_key(dedupe_key, hint_text=input_text)
    if dedupe_key:
        item["dedupeKey"] = dedupe_key
    if cluster_key:
        item["clusterKey"] = cluster_key

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
    if label in {"policy", "sanctions"}:
        return "low"
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





def _parse_datetime(value: str) -> datetime.datetime | None:
    return parse_datetime_utc(value, default_tz=_KST)

def _parse_date_base(value: str) -> datetime.datetime | None:
    return parse_date_base_utc(value, base_tz=_KST)



def _iter_impact_signal_entries(impact_signals: Any) -> list[tuple[str, str]]:
    if not isinstance(impact_signals, list):
        return []
    entries: list[tuple[str, str]] = []
    for entry in impact_signals:
        if isinstance(entry, dict):
            label = clean_text(entry.get("label") or "").lower()
        else:
            label = clean_text(str(entry)).lower()
            evidence = ""
        if isinstance(entry, dict):
            evidence = clean_text(entry.get("evidence") or "")
        
        if not label:
            continue
        entries.append((label, evidence))
    return entries

def _remap_label_by_evidence(evidence: str) -> str:
    for label in _IMPACT_LABEL_PRIORITY:
        if _label_evidence_valid(label, evidence):
            return label
    return ""

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


def _build_impact_signals_detail(
    item: dict,
    ai_result: dict,
    *,
    full_text: str,
    summary_text: str,
) -> list[dict[str, str]]:
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

    impact_signals_detail = _sanitize_impact_signals(impact_signals_detail, full_text, summary_text)
    return impact_signals_detail


def _retry_impact_signals(
    item: dict,
    source_item: dict | None,
    *,
    full_text: str,
    summary_text: str,
) -> bool:
    if source_item is None:
        return False
    ai_result = enrich_item_with_ai(source_item) or {}
    if not isinstance(ai_result, dict) or not ai_result:
        return False
    impact_signals_detail = _build_impact_signals_detail(
        source_item,
        ai_result,
        full_text=full_text,
        summary_text=summary_text,
    )
    if not impact_signals_detail:
        return False
    item["impactSignals"] = impact_signals_detail
    try:
        importance_score = int(ai_result.get("importance_score") or 0)
    except Exception:
        importance_score = 0
    if importance_score:
        item["importance"] = max(1, min(5, importance_score))
    return True

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

    title_text = clean_text(item.get("title") or "")
    summary_clean = clean_text(summary_text or "")
    align_text = f"{title_text} {summary_clean}".strip()
    norm_tokens = _normalize_text_tokens(align_text)
    dedupe_tokens = _normalize_text_tokens((item.get("dedupeKey") or "").replace("-", " "))
    cluster_tokens = _normalize_text_tokens((item.get("clusterKey") or "").replace("/", " "))
    if len(dedupe_tokens & norm_tokens) < 2:
        errors.append("ERROR: DEDUPE_KEY_NOT_ALIGNED")
    if len(cluster_tokens & norm_tokens) < 1:
        errors.append("ERROR: CLUSTER_KEY_NOT_ALIGNED")

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

    if importance >= 3 and isinstance(impact_signals, list) and len(impact_signals) == 0:
        if any(t in align_text.lower() for t in _ALIGNMENT_TRIGGERS):
            errors.append("ERROR: IMPACT_SIGNALS_MISSING_FOR_HIGH_IMPORTANCE")

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
        "ERROR: DEDUPE_KEY_NOT_ALIGNED",
        "ERROR: CLUSTER_KEY_NOT_ALIGNED",
        "ERROR: IMPACT_SIGNALS_MISSING_FOR_HIGH_IMPORTANCE",
    }
    s3 = {"ERROR: LOW_QUALITY_MISMATCH", "ERROR: IMPACT_SIGNALS_REQUIRED"}
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
    if "ERROR: LOW_QUALITY_MISMATCH" in errors:
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
    if "ERROR: IMPACT_SIGNALS_REQUIRED" in errors:
        try:
            importance = int(item.get("importance") or 0)
        except Exception:
            importance = 0
        if importance >= 3:
            item["importance"] = 2

def handle_validation_errors(
    item: dict,
    errors: list[str],
    *,
    source_item: dict | None = None,
) -> dict:
    log = {
        "item_id": item.get("_itemId") or item.get("id") or "",
        "dedupe_input_hash": item.get("_dedupeInputHash") or "",
        "original_errors": list(errors),
        "auto_fixed": [],
        "remaining_errors": [],
        "final_action": "kept",
    }
    classified = classify_errors(errors)
    if classified["s2"]:
        auto_fixed = apply_auto_fixes(item)
        log["auto_fixed"] = auto_fixed
        sanitized = _sanitize_impact_signals(
            item.get("impactSignals"),
            item.get("_fullText") or "",
            item.get("_summaryText") or "",
        )
        if sanitized != item.get("impactSignals"):
            item["impactSignals"] = sanitized
            log["auto_fixed"].append("sanitize_impact_signals")
        if "ERROR: DEDUPE_KEY_NOT_ALIGNED" in errors or "ERROR: CLUSTER_KEY_NOT_ALIGNED" in errors:
            _regenerate_keys_from_title_summary(item)
            log["auto_fixed"].append("regen_dedupe_cluster")
        if "ERROR: IMPACT_SIGNALS_MISSING_FOR_HIGH_IMPORTANCE" in errors:
            if not item.get("_impactRetryDone"):
                item["_impactRetryDone"] = True
                retried = _retry_impact_signals(
                    item,
                    source_item,
                    full_text=item.get("_fullText") or "",
                    summary_text=item.get("_summaryText") or "",
                )
                if retried:
                    log["auto_fixed"].append("retry_llm_impact_signals")
    new_errors = revalidate(
        item,
        full_text=item.get("_fullText") or "",
        summary_text=item.get("_summaryText") or "",
    )
    log["remaining_errors"] = new_errors
    classified_after = classify_errors(new_errors)
    if "ERROR: IMPACT_SIGNALS_MISSING_FOR_HIGH_IMPORTANCE" in new_errors:
        item["qualityLabel"] = "low_quality"
        if not item.get("qualityReason"):
            item["qualityReason"] = "impact_signals_missing"
        item["status"] = "dropped"
        item["dropReason"] = "impact_signals_missing"
        log["final_action"] = "dropped"
        return log
    if classified_after["s2"] and not classified_after["s1"] and not classified_after["unknown"]:
        if item.get("impactSignals"):
            item["impactSignals"] = []
            log["auto_fixed"].append("drop_invalid_impact_signals")
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

def export_daily_digest_json(
    top_items: list[dict],
    output_path: str,
    config: dict,
    *,
    metrics_extra: dict[str, Any] | None = None,
) -> DailyDigest:
    """MVP 스키마로 변환해 JSON으로 저장."""
    now_kst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    date_str = now_kst.strftime("%Y-%m-%d")
    last_updated_at = now_kst.isoformat()

    items_out: list[dict[str, Any]] = []
    validation_logs: list[dict[str, Any]] = []
    metrics = {
        "total_in": 0,
        "total_out": 0,
        "dropped": 0,
        "dropReasons": {},
        "impactLabels": {},
        "sources": {},
        "categories": {},
        "importanceDistribution": {},
        "importanceBySignals": {},
    }
    out_index = 0
    kept_dedupe_keys: set[str] = set()
    for item in top_items[:TOP_LIMIT]:
        metrics["total_in"] += 1
        title = (item.get("title") or "").strip()
        link = (item.get("link") or "").strip()
        summary = (item.get("summary") or "").strip()
        summary_raw = (item.get("summaryRaw") or "").strip()
        full_text = (item.get("fullText") or "").strip()
        topic = (item.get("topic") or "").strip()
        source_name = (item.get("source") or "").strip()
        published = item.get("updatedAtUtc") or item.get("publishedAtUtc") or item.get("published")
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
            metrics["dropped"] += 1
            label = drop_reason or "dropped"
            metrics["dropReasons"][label] = metrics["dropReasons"].get(label, 0) + 1
            continue

        ai_result = item.get("ai")
        should_skip_ai = bool(is_merged or drop_reason or status_value == "dropped" or full_text_len < 50)
        if not should_skip_ai and (not isinstance(ai_result, dict) or not ai_result):
            ai_result = enrich_item_with_ai(item) or {}
        elif not isinstance(ai_result, dict):
            ai_result = {}
        title_ko = clean_text(ai_result.get("title_ko") or "")
        title_from_ai = False
        if title_ko:
            title = title_ko
            title_from_ai = True
        ai_lines_raw = ai_result.get("summary_lines") or []
        summary_from_ai = any(clean_text(str(x)) for x in ai_lines_raw) if isinstance(ai_lines_raw, list) else False
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
            
        engine = _get_dedupe_engine()
        summary_text = " ".join(summary_lines) if summary_lines else summary
        if title_from_ai or summary_from_ai:
            dedupe_key = engine.build_dedupe_key(title, summary_text)
            cluster_key = engine.build_cluster_key(dedupe_key, hint_text=f"{title} {summary_text}")
        else:
            dedupe_key = item.get("dedupeKeyRule") or item.get("dedupeKey") or ""
            dedupe_input_summary = clean_text(summary or summary_raw)
            if not dedupe_key:
                dedupe_key = engine.build_dedupe_key(title, dedupe_input_summary)
            cluster_key = item.get("clusterKey") or ""
            if not cluster_key and dedupe_key:
                cluster_key = engine.build_cluster_key(dedupe_key, hint_text=f"{title} {dedupe_input_summary}")
        impact_signals_detail = _build_impact_signals_detail(
            item,
            ai_result,
            full_text=full_text,
            summary_text=summary_text,
        )
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
        if int(importance or 0) >= 3 and not impact_signals_detail:
            importance = 2
            if "근거부족" not in quality_reason:
                quality_reason = "근거부족" if quality_reason == "정보성 기사" else f"{quality_reason} / 근거부족"
        if status_value == "dropped":
            continue
        if status_value in {"kept", "published"}:
            key = clean_text(dedupe_key)
            if key:
                if key in kept_dedupe_keys:
                    print(
                        "EXPORT_SKIP_DUPLICATE_DEDUPE "
                        f"id={item.get('itemId') or ''} key={key[:80]} title={(title or '')[:60]}"
                    )
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
        out_item["_itemId"] = item.get("itemId")
        out_item["_dedupeInputHash"] = item.get("dedupeInputHash")
        item_errors = _collect_item_errors(
            out_item,
            full_text=full_text,
            summary_text=summary_text,
        )
        log_entry = handle_validation_errors(out_item, item_errors, source_item=item)
        validation_logs.append(log_entry)
        out_item.pop("_fullText", None)
        out_item.pop("_summaryText", None)
        out_item.pop("_itemId", None)
        out_item.pop("_dedupeInputHash", None)
        out_item.pop("_impactRetryDone", None)
        if out_item.get("status") == "dropped":
            continue
        if int(out_item.get("importance") or 0) >= 3 and not out_item.get("impactSignals"):
            out_item["importance"] = 2
            if "근거부족" not in (out_item.get("qualityReason") or ""):
                if out_item.get("qualityReason") == "정보성 기사":
                    out_item["qualityReason"] = "근거부족"
                else:
                    out_item["qualityReason"] = f"{out_item.get('qualityReason') or ''} / 근거부족".strip(" /")
        items_out.append(out_item)

        metrics["total_out"] += 1
        src_label = out_item.get("sourceName") or ""
        if src_label:
            metrics["sources"][src_label] = metrics["sources"].get(src_label, 0) + 1
        cat_label = out_item.get("category") or ""
        if cat_label:
            metrics["categories"][cat_label] = metrics["categories"].get(cat_label, 0) + 1
        try:
            importance_val = int(out_item.get("importance") or 0)
        except Exception:
            importance_val = 0
        imp_key = str(importance_val)
        metrics["importanceDistribution"][imp_key] = metrics["importanceDistribution"].get(imp_key, 0) + 1
        imp_bucket = metrics["importanceBySignals"].setdefault(imp_key, {"total": 0, "withSignals": 0})
        imp_bucket["total"] += 1
        if out_item.get("impactSignals"):
            imp_bucket["withSignals"] += 1
        for sig in out_item.get("impactSignals") or []:
            label = clean_text(sig.get("label") or "").lower()
            if not label:
                continue
            metrics["impactLabels"][label] = metrics["impactLabels"].get(label, 0) + 1

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

    if metrics.get("sources"):
        max_source = max(metrics["sources"].values()) if metrics["sources"] else 0
    else:
        max_source = 0
    metrics["topDiversity"] = {
        "uniqueSources": len(metrics.get("sources", {})),
        "uniqueCategories": len(metrics.get("categories", {})),
        "maxPerSource": max_source,
    }

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
    metrics_payload: dict[str, Any] = {"type": "metrics_summary", "date": date_str, **metrics}
    if metrics_extra:
        metrics_payload.update(metrics_extra)
    try:
        print(json.dumps(metrics_payload, ensure_ascii=False))
    except Exception:
        pass
    try:
        _atomic_write_json(METRICS_JSON, metrics_payload)
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

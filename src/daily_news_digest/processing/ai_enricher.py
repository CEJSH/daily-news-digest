from __future__ import annotations

import os

from daily_news_digest.utils import clean_text, sanitize_text
from daily_news_digest.processing.enrichment_utils import (
    _apply_quality_guardrails,
    _fallback_why_important,
    _filter_impact_signal_objects,
    _normalize_category_label,
    _normalize_dedupe_key,
    _normalize_importance_score,
    _normalize_impact_signal_objects,
    _normalize_quality_label,
    _normalize_quality_tags,
    _normalize_summary_lines,
    _rule_based_impact_signals,
)
from daily_news_digest.processing.llm_client import (
    gemini_generate_json,
    log_ai_unavailable,
    parse_json as _parse_json,
)
from daily_news_digest.processing.prompts.digest_prompt import SYSTEM_PROMPT

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv:
    from pathlib import Path

    _repo_root = Path(__file__).resolve().parents[3]
    load_dotenv(dotenv_path=_repo_root / ".env")

AI_INPUT_MAX_CHARS = int(os.getenv("AI_INPUT_MAX_CHARS", "4000"))
AI_IMPACT_EVIDENCE_MIN_CHARS = int(os.getenv("AI_IMPACT_EVIDENCE_MIN_CHARS", "400"))

# Backwards-compatible alias
_log_ai_unavailable = log_ai_unavailable


def _pick_ai_input_text(full_text: str) -> str:
    # 본문만 사용. 없으면 빈 문자열 반환.
    text = full_text or ""
    if len(text) > AI_INPUT_MAX_CHARS:
        text = text[:AI_INPUT_MAX_CHARS]
    return text


def enrich_item_with_ai(item: dict) -> dict:
    # 기사 아이템을 AI로 요약/분류/중요도 평가
    title = clean_text(item.get("title") or "")
    summary_raw = clean_text(
        sanitize_text(item.get("summaryRaw") or item.get("summary") or "")
    )
    full_text = clean_text(
        sanitize_text(item.get("fullText") or "")
    )
    if full_text and len(full_text) > 6000:
        full_text = full_text[:6000]
    source = clean_text(item.get("source") or "")
    published = clean_text(item.get("published") or "")
    impact_signals = item.get("impactSignals") or []

    # 모델 입력 구성
    input_text = _pick_ai_input_text(full_text)
    if not input_text:
        _log_ai_unavailable(f"본문 없음: {item.get('link') or item.get('title','')[:60]}")
        return {}
    full_text_clean = clean_text(full_text or "")
    if len(full_text_clean) < AI_IMPACT_EVIDENCE_MIN_CHARS:
        candidates = []
    else:
        candidates = _rule_based_impact_signals(full_text_clean)
    user_prompt = (
        f"Title: {title}\n"
        f"ImpactSignalsHint: {', '.join(impact_signals)}\n"
        f"ImpactSignalCandidates: {', '.join(candidates)}\n"
        f"Text: {input_text}\n"
        "Return only JSON."
    )

    # Gemini만 사용 (OpenAI 폴백 없음)
    payload = gemini_generate_json(SYSTEM_PROMPT, user_prompt)
    if not isinstance(payload, dict):
        return {}

    # 요약/중요도/라벨 결과를 정규화
    title_ko = clean_text(payload.get("title_ko") or "")
    if not title_ko:
        title_ko = title
    summary_fallback = summary_raw
    summary_lines = _normalize_summary_lines(
        payload.get("summary_lines") or [],
        title_ko or title,
        summary_fallback,
    )
    why_important = clean_text(payload.get("why_important") or "")
    if not why_important:
        why_important = _fallback_why_important(impact_signals)
    importance_rationale = clean_text(payload.get("importance_rationale") or "")

    dedupe_key = _normalize_dedupe_key(payload.get("dedupe_key") or "")
    if not dedupe_key:
        dedupe_key = _normalize_dedupe_key(item.get("dedupeKey") or title or summary_raw)

    impact_signals_obj = _normalize_impact_signal_objects(payload.get("impact_signals"))
    evidence_text = full_text_clean
    if len(evidence_text) < AI_IMPACT_EVIDENCE_MIN_CHARS:
        candidates = []
    else:
        candidates = _rule_based_impact_signals(evidence_text)
    impact_signals_ai, evidence_map = _filter_impact_signal_objects(
        impact_signals_obj,
        candidates,
        evidence_text,
    )
    importance_score = _normalize_importance_score(
        payload.get("importance_score") or payload.get("importance"),
        impact_signals,
    )
    if not importance_rationale:
        if importance_score <= 2:
            importance_rationale = "근거가 충분히 드러나지 않아 중요도가 낮습니다."
        else:
            importance_rationale = why_important or "중요도 근거가 요약에 충분히 나타나지 않습니다."
    quality_label = _normalize_quality_label(payload.get("quality_label") or payload.get("quality"))
    quality_reason = clean_text(payload.get("quality_reason") or "")
    quality_tags = _normalize_quality_tags(payload.get("quality_tags"))
    category_label = _normalize_category_label(payload.get("category_label") or payload.get("category"))

    quality_label, quality_reason, quality_tags = _apply_quality_guardrails(
        quality_label=quality_label,
        quality_reason=quality_reason,
        quality_tags=quality_tags,
        title=title_ko or title,
        summary_lines=summary_lines,
        why_important=why_important,
    )

    return {
        "title_ko": title_ko,
        "summary_lines": summary_lines,
        "why_important": why_important,
        "importance_rationale": importance_rationale,
        "dedupe_key": dedupe_key,
        "importance_score": importance_score,
        "impact_signals": impact_signals_ai,
        "impact_signals_evidence": {s: evidence_map.get(s, "") for s in impact_signals_ai},
        "quality_label": quality_label,
        "quality_reason": quality_reason,
        "quality_tags": quality_tags,
        "category_label": category_label,
    }

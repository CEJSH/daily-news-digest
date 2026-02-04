from __future__ import annotations

import datetime
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
from daily_news_digest.models import DailyDigest
from daily_news_digest.utils import (
    clean_text,
    contains_binary,
    ensure_lines_1_to_3,
    estimate_read_time_seconds,
    jaccard_tokens,
    normalize_summary_lines_for_focus,
)

_LONG_IMPACT = {"policy", "sanctions"}  # 장기 영향 신호로 간주하는 카테고리
_MED_IMPACT = {"capex", "infra", "security"}  # 중간 영향 신호 카테고리
_LOW_IMPACT = {"earnings", "market-demand"}  # 단기 영향 신호 카테고리

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


def _infer_importance_from_signals(signals: set[str]) -> int:
    """impactSignals 기반 기본 중요도 점수 산정."""
    if signals & _LONG_IMPACT:
        return 4
    if signals & _MED_IMPACT:
        return 3
    if signals & _LOW_IMPACT:
        return 2
    return 1


def _pick_summary_source(title: str, summary: str, summary_raw: str, full_text: str) -> str:
    """요약 후보 중 제목과 중복되지 않는 가장 긴 본문 선택."""
    title_clean = clean_text(title)
    candidates = [
        clean_text(full_text),
        clean_text(summary_raw),
        clean_text(summary),
    ]
    filtered = [c for c in candidates if c and c.lower() != title_clean.lower()]
    if filtered:
        return max(filtered, key=len)
    return clean_text(summary_raw or summary or full_text or title_clean)

def _load_existing_digest(path: str) -> DailyDigest | None:
    """기존 digest 파일 로드."""
    return _safe_read_json(path, None)

def _is_valid_digest(digest: dict) -> bool:
    """MVP 안전장치: 최소 개수 + 핵심 필드 존재 여부만 검사 (엄격하게)."""
    if not isinstance(digest, dict):
        return False
    items = digest.get("items")
    if not isinstance(items, list) or len(items) < MIN_TOP_ITEMS or len(items) > TOP_LIMIT:
        return False

    required_item_keys = {"id", "date", "category", "title", "summary", "sourceName", "sourceUrl", "status", "importance"}
    for it in items:
        if not isinstance(it, dict):
            return False
        if not required_item_keys.issubset(it.keys()):
            return False
        if not it.get("title") or not it.get("sourceUrl"):
            return False
        summary = it.get("summary")
        if not isinstance(summary, list) or len(summary) == 0:
            return False
    return True

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
    for i, item in enumerate(top_items[:TOP_LIMIT], start=1):
        title = (item.get("title") or "").strip()
        link = (item.get("link") or "").strip()
        summary = (item.get("summary") or "").strip()
        summary_raw = (item.get("summaryRaw") or "").strip()
        full_text = (item.get("fullText") or "").strip()
        topic = (item.get("topic") or "").strip()
        source_name = (item.get("source") or "").strip()
        published = item.get("published")

        drop_reason = item.get("dropReason") or ""
        status_value = item.get("status") or ("dropped" if drop_reason else "kept")
        full_text_len = len(clean_text(full_text))
        if full_text_len < 80 and not drop_reason:
            drop_reason = "policy:full_text_missing"
            status_value = "dropped"

        ai_result = item.get("ai")
        should_skip_ai = bool(drop_reason or status_value == "dropped" or full_text_len < 80)
        if not should_skip_ai and (not isinstance(ai_result, dict) or not ai_result):
            ai_result = enrich_item_with_ai(item) or {}
        elif not isinstance(ai_result, dict):
            ai_result = {}
        title_ko = clean_text(ai_result.get("title_ko") or "")
        if title_ko:
            title = title_ko
        summary_source = _pick_summary_source(title, summary, summary_raw, full_text)
        summary_lines = ensure_lines_1_to_3(ai_result.get("summary_lines") or [], summary_source)
        summary_lines, is_briefing = normalize_summary_lines_for_focus(
            summary_lines,
            title,
            summary_source,
        )

        quality_label = ai_result.get("quality_label") or item.get("aiQuality") or "ok"
        quality_reason = clean_text(ai_result.get("quality_reason") or item.get("quality_reason") or "")
        if quality_label == "low_quality" and not quality_reason:
            quality_reason = "정보 부족"

        force_low_quality_downgrade = False
        if quality_label == "low_quality" and not drop_reason and status_value != "dropped":
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
        if any(contains_binary(line) for line in summary_lines) or contains_binary(summary_raw) or contains_binary(summary):
            policy_drop_reason = "summary_binary"
        elif drop_reason:
            policy_drop_reason = ""
        elif _is_title_like_summary(title, summary_lines):
            policy_drop_reason = "summary_title_only"

        if policy_drop_reason and not drop_reason:
            drop_reason = f"policy:{policy_drop_reason}"
        if drop_reason:
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
        if full_text_len < 80:
            why_important = "본문 확보 실패로 판단 불가입니다."
            importance_rationale = "근거: 본문 확보 실패로 판단 불가입니다."
        if drop_reason or status_value == "dropped":
            why_important = _fallback_why()
            importance_rationale = _fallback_importance_rationale()

        dedupe_key = ai_result.get("dedupe_key") or item.get("dedupeKey", "")
        impact_signals = ai_result.get("impact_signals") or item.get("impactSignals", [])
        importance = ai_result.get("importance_score")
        if not importance:
            signals = set(item.get("impactSignals") or [])
            importance = _infer_importance_from_signals(signals)
        if force_low_quality_downgrade:
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

        if drop_reason or status_value == "dropped":
            quality_label = "low_quality"
            if not quality_reason:
                quality_reason = _drop_reason_message(drop_reason)
        if not quality_reason:
            quality_reason = "정보성 기사"

        out_item = {
            "id": f"{date_str}_{i}",
            "date": date_str,
            "category": category,
            "title": title,
            "summary": summary_lines if summary_lines else [summary],
            "whyImportant": why_important,
            "importanceRationale": importance_rationale,
            "impactSignals": impact_signals,
            "dedupeKey": dedupe_key,
            "matchedTo": item.get("matchedTo"),
            "sourceName": source_name,
            "sourceUrl": link,
            "publishedAt": published,
            "readTimeSec": read_time_sec,
            "status": status_value,
            "importance": importance,
            "qualityLabel": quality_label,
            "qualityReason": quality_reason,
            "isBriefing": is_briefing,
        }
        if drop_reason:
            out_item["dropReason"] = drop_reason
        items_out.append(out_item)

    digest = {
        "date": date_str,
        "selectionCriteria": config["selection_criteria"],
        "editorNote": config["editor_note"],
        "question": config["question"],
        "lastUpdatedAt": last_updated_at,
        "items": items_out,
    }

    if not _is_valid_digest(digest):
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

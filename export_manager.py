import datetime
import json
import os
import re
from typing import Any
from ai_enricher import enrich_item_with_ai
from config import TOP_LIMIT, MIN_TOP_ITEMS, DEDUPE_HISTORY_PATH, DEDUPE_RECENT_DAYS
from utils import clean_text, ensure_lines_1_to_3, estimate_read_time_seconds

_LONG_IMPACT = {"policy", "sanctions"}
_MED_IMPACT = {"capex", "infra", "security"}
_LOW_IMPACT = {"earnings", "market-demand"}

_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_PNG_SIGNS = ("PNG", "IHDR", "IDAT", "IEND")


def _contains_binary(text: str) -> bool:
    if not text:
        return False
    if _CONTROL_RE.search(text):
        return True
    if text.count("�") / max(1, len(text)) > 0.01:
        return True
    upper = text.upper()
    return any(sign in upper for sign in _PNG_SIGNS)


def _normalize_for_compare(text: str) -> str:
    t = clean_text(text or "").lower()
    t = re.sub(r"[^a-z0-9가-힣]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _jaccard(a: str, b: str) -> float:
    toks_a = set(a.split())
    toks_b = set(b.split())
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def _is_title_like_summary(title: str, lines: list[str]) -> bool:
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
        if _jaccard(norm_line, norm_title) >= 0.85:
            similar += 1
            continue
    return similar >= max(1, len(lines))


def _safe_read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _infer_importance_from_signals(signals: set[str]) -> int:
    if signals & _LONG_IMPACT:
        return 4
    if signals & _MED_IMPACT:
        return 3
    if signals & _LOW_IMPACT:
        return 2
    return 1


def _pick_summary_source(title: str, summary: str, summary_raw: str, full_text: str) -> str:
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

def _load_existing_digest(path: str) -> dict | None:
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
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

def _load_dedupe_history(path: str) -> dict:
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

def export_daily_digest_json(top_items: list[dict], output_path: str, config: dict) -> dict:
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

        def _drop_reason_message(reason: str) -> str:
            if not reason:
                return "본문 확보 실패"
            if "summary_binary" in reason:
                return "요약/본문 데이터에 바이너리 또는 손상 텍스트가 포함되어 있습니다."
            if "summary_title_only" in reason:
                return "요약이 제목 반복으로 편집 기준을 충족하지 못합니다."
            if reason.startswith("fetch_failed:"):
                return reason.split(":", 1)[1]
            if reason.startswith("ai_low_quality:"):
                return reason.split(":", 1)[1]
            return reason

        policy_drop_reason = ""
        if any(_contains_binary(line) for line in summary_lines) or _contains_binary(summary_raw) or _contains_binary(summary):
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

        dedupe_key = ai_result.get("dedupe_key") or item.get("dedupeKey", "")
        impact_signals = ai_result.get("impact_signals") or item.get("impactSignals", [])
        importance = ai_result.get("importance_score")
        if not importance:
            signals = set(item.get("impactSignals") or [])
            importance = _infer_importance_from_signals(signals)
        read_time_sec = item.get("readTimeSec")
        if not read_time_sec:
            read_time_sec = estimate_read_time_seconds(" ".join(summary_lines) if summary_lines else summary)

        from news_digest_exporter import map_topic_to_category
        category = item.get("aiCategory") or map_topic_to_category(topic)

        quality_label = ai_result.get("quality_label") or item.get("aiQuality") or "ok"
        quality_reason = clean_text(ai_result.get("quality_reason") or item.get("quality_reason") or "")
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
            print("⚠️ 오늘 digest 생성이 불완전하여 기존 daily_digest.json을 유지합니다.")
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

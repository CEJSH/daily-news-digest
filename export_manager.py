import datetime
import json
import os
from ai_enricher import enrich_item_with_ai
from utils import clean_text, ensure_three_lines, estimate_read_time_seconds

_LONG_IMPACT = {"policy", "sanctions"}
_MED_IMPACT = {"capex", "infra", "security"}
_LOW_IMPACT = {"earnings", "market-demand"}

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
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _is_valid_digest(digest: dict) -> bool:
    """MVP 안전장치: 5개 고정 + 핵심 필드 존재 여부만 검사 (엄격하게)."""
    if not isinstance(digest, dict):
        return False
    items = digest.get("items")
    if not isinstance(items, list) or len(items) != 5:
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

def export_daily_digest_json(top_items: list[dict], output_path: str, config: dict) -> dict:
    """MVP 스키마로 변환해 JSON으로 저장."""
    now_kst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    date_str = now_kst.strftime("%Y-%m-%d")
    last_updated_at = now_kst.isoformat()

    items_out: list[dict] = []
    for i, item in enumerate(top_items[:5], start=1):
        title = (item.get("title") or "").strip()
        link = (item.get("link") or "").strip()
        summary = (item.get("summary") or "").strip()
        summary_raw = (item.get("summaryRaw") or "").strip()
        full_text = (item.get("fullText") or "").strip()
        topic = (item.get("topic") or "").strip()
        source_name = (item.get("source") or "").strip()
        published = item.get("published")

        ai_result = item.get("ai") or enrich_item_with_ai(item)
        summary_source = _pick_summary_source(title, summary, summary_raw, full_text)
        summary_lines = ensure_three_lines(ai_result.get("summary_lines") or [], summary_source)
        why_important = ai_result.get("why_important") or ""
        dedupe_key = ai_result.get("dedupe_key") or item.get("dedupeKey", "")
        impact_signals = ai_result.get("impact_signals") or item.get("impactSignals", [])
        importance = ai_result.get("importance_score")
        if not importance:
            signals = set(item.get("impactSignals") or [])
            if signals & _LONG_IMPACT:
                importance = 4
            elif signals & _MED_IMPACT:
                importance = 3
            elif signals & _LOW_IMPACT:
                importance = 2
            else:
                importance = 1
        read_time_sec = item.get("readTimeSec")
        if not read_time_sec:
            read_time_sec = estimate_read_time_seconds(" ".join(summary_lines) if summary_lines else summary)

        from news_digest_exporter import map_topic_to_category
        category = item.get("aiCategory") or map_topic_to_category(topic)
        out_item = {
            "id": f"{date_str}_{i}",
            "date": date_str,
            "category": category,
            "title": title,
            "summary": summary_lines if summary_lines else [summary],
            "whyImportant": why_important,
            "impactSignals": impact_signals,
            "dedupeKey": dedupe_key,
            "matchedTo": item.get("matchedTo"),
            "sourceName": source_name,
            "sourceUrl": link,
            "publishedAt": published,
            "readTimeSec": read_time_sec,
            "status": "kept",
            "importance": importance,
        }
        if item.get("dropReason"):
            out_item["dropReason"] = item.get("dropReason")
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
        raise RuntimeError("digest 생성 실패: 유효한 5개 뉴스가 생성되지 않았고 기존 파일도 없습니다.")

    _atomic_write_json(output_path, digest)
    return digest
